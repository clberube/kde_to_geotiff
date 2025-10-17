#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import KernelDensity
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from scipy.spatial import cKDTree


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute KDE over a grid from point shapefile and export to GeoTIFF with transparency outside the kernel range."
    )
    p.add_argument("shapefile", type=Path, help="Path to input shapefile (points).")
    p.add_argument("output", type=Path, help="Path to output GeoTIFF.")
    p.add_argument(
        "--kernel-size",
        type=float,
        required=True,
        help="Kernel bandwidth (same units as the shapefile CRS; e.g., meters).",
    )
    p.add_argument(
        "--pixel-size",
        type=float,
        required=True,
        help="Grid pixel size (same units as the shapefile CRS; e.g., meters).",
    )
    p.add_argument(
        "--data-column",
        type=str,
        default=None,
        help="Name of a column to use for filtering points. Rows with missing values in this column are ignored.",
    )
    p.add_argument(
        "--to-crs",
        type=str,
        default=None,
        help="Optional target CRS (e.g., 'EPSG:3857'). If provided, data are reprojected before KDE.",
    )
    p.add_argument(
        "--clip-padding",
        type=float,
        default=None,
        help="Optional extra padding around bounds (defaults to kernel-size).",
    )
    p.add_argument(
        "--kernel",
        type=str,
        default="gaussian",
        choices=[
            "gaussian",
            "tophat",
            "epanechnikov",
            "exponential",
            "linear",
            "cosine",
        ],
        help="Kernel type for KDE (sklearn). Default: gaussian.",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=1e-12,
        help="Tiny additive for numerical stability when exponentiating log-density.",
    )
    return p.parse_args()


def read_points(
    shp_path: Path,
    to_crs: str | None,
    data_column: str | None,
) -> tuple[np.ndarray, CRS, gpd.GeoDataFrame]:
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("The shapefile contains no features.")
    if gdf.geometry.is_empty.all():
        raise ValueError("All geometries are empty.")

    # Keep points; if lines/polygons sneak in, use their centroid to avoid failing hard.
    geom = gdf.geometry
    if not geom.geom_type.isin(["Point"]).all():
        geom = geom.centroid

    if to_crs is not None:
        gdf = gdf.set_geometry(geom)
        gdf = gdf.to_crs(to_crs)
        geom = gdf.geometry
        crs = CRS.from_string(to_crs)
    else:
        if gdf.crs is None:
            raise ValueError("Input shapefile has no CRS. Provide --to-crs.")
        crs = CRS.from_user_input(gdf.crs)

    # Filter by data column if provided
    if data_column is not None:
        if data_column not in gdf.columns:
            raise SystemExit(
                f"Error: column '{data_column}' not found in shapefile attributes."
            )

        # Drop rows with NaN, None, or empty string in that column
        before_count = len(gdf)
        gdf = gdf[gdf[data_column].notnull() & (gdf[data_column] != "")]
        after_count = len(gdf)

        if after_count == 0:
            raise SystemExit(
                f"Error: all rows have missing values in column '{data_column}'."
            )
        else:
            print(
                f"Filtered {before_count - after_count} rows with missing values in '{data_column}'."
            )

    coords = np.vstack([(pt.x, pt.y) for pt in geom if isinstance(pt, Point)])
    if coords.size == 0:
        raise ValueError("No point coordinates found after geometry handling.")
    return coords, crs, gdf.set_geometry(geom)


def make_grid(
    bounds, pixel_size: float, pad: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = bounds
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad

    # Align to pixel grid (top-left aligned)
    width = int(math.ceil((maxx - minx) / pixel_size))
    height = int(math.ceil((maxy - miny) / pixel_size))

    # Recompute exact max edges from integer dims
    maxx = minx + width * pixel_size
    maxy = miny + height * pixel_size

    xs = np.linspace(minx + pixel_size / 2.0, maxx - pixel_size / 2.0, width)
    ys = np.linspace(
        maxy - pixel_size / 2.0, miny + pixel_size / 2.0, height
    )  # top to bottom
    Xc, Yc = np.meshgrid(xs, ys)
    return Xc, Yc, xs, (minx, maxy, width, height)


def kde_on_grid(
    points_xy: np.ndarray,
    bandwidth: float,
    kernel: str,
    Xc: np.ndarray,
    Yc: np.ndarray,
    atol: float,
) -> np.ndarray:
    # sklearn expects shape (n_samples, n_features)
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(points_xy)

    # Evaluate at grid centers
    grid_pts = np.column_stack([Xc.ravel(), Yc.ravel()])
    log_density = kde.score_samples(grid_pts)  # log p(x)
    density = np.exp(
        log_density - log_density.max()
    )  # scale to [0,1] range for numerical stability
    density = density + atol
    return density.reshape(Xc.shape)


def mask_outside_kernel(
    points_xy: np.ndarray, Xc: np.ndarray, Yc: np.ndarray, kernel_size: float
) -> np.ndarray:
    # Build KDTree for nearest neighbor distances
    tree = cKDTree(points_xy)
    grid_pts = np.column_stack([Xc.ravel(), Yc.ravel()])
    dists, _ = tree.query(grid_pts, k=1)
    dists = dists.reshape(Xc.shape)
    mask = (dists <= 3 * kernel_size).astype(bool)  # 1 inside, 0 outside
    return mask


def main():
    args = parse_args()

    if args.kernel_size <= 0 or args.pixel_size <= 0:
        print("ERROR: --kernel-size and --pixel-size must be > 0.", file=sys.stderr)
        sys.exit(2)

    points_xy, crs, gdf = read_points(args.shapefile, args.to_crs, args.data_column)
    pad = args.clip_padding if args.clip_padding is not None else args.kernel_size

    # Bounds in current CRS
    minx, miny, maxx, maxy = gdf.total_bounds
    Xc, Yc, xs, (origin_minx, origin_maxy, width, height) = make_grid(
        (minx, miny, maxx, maxy), pixel_size=args.pixel_size, pad=pad
    )

    # KDE
    dens = kde_on_grid(points_xy, args.kernel_size, args.kernel, Xc, Yc, args.atol)

    # Mask when density is below 1%
    threshold = 0.01 * np.nanmax(dens)
    in_mask = dens >= threshold

    # Alternative mask: transparent when farther than kernel-size from any input point
    # in_mask = mask_outside_kernel(points_xy, Xc, Yc, args.kernel_size)

    nodata_value = -9999.0
    out = dens.copy()
    out[~in_mask] = nodata_value
    # alpha = (in_mask * 255).astype(
    #     np.uint8
    # )  # 255 opaque where within kernel range; 0 transparent outside

    # GeoTIFF transform (top-left origin)
    transform = from_origin(origin_minx, origin_maxy, args.pixel_size, args.pixel_size)

    # Write GeoTIFF with alpha mask
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": nodata_value,
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(out.astype(np.float32), 1)
        # Optional : Export an explicit alpha mask (0 transparent, 255 opaque)
        # dst.write_mask(alpha)

    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
