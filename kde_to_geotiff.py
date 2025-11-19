#!/usr/bin/env python3
import argparse
import math
import time
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# from sklearn.neighbors import KernelDensity
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

# from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute KDE over a grid from point shapefile and export to GeoTIFF with transparency outside the kernel range."
    )
    p.add_argument("shapefile", type=Path, help="Path to input shapefile (points).")
    p.add_argument("output", type=Path, help="Path to output GeoTIFF.")
    p.add_argument(
        "--kernel-size",
        type=float,
        default=0.1,
        help="Kernel bandwidth as a fraction of Scott's rule bandwidth (Default: 0.1)).",
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
        "--clip-to-hull",
        action="store_true",
        help="If set, clip the KDE to the convex hull of the input geometries (expanded by kernel_size).",
    )
    p.add_argument(
        "--horizon-column",
        type=str,
        default=None,
        help="Name of the column indicating soil horizons (e.g., 'HORIZON').",
    )
    p.add_argument(
        "--horizon-keep",
        type=str,
        nargs="+",
        default=None,
        help="List of horizon labels to keep (e.g., A, B, C). If None, all samples are kept.",
    )
    p.add_argument(
        "--remove-negative",
        action="store_true",
        help="If set, remove rows where the value in --data-column is negative (<0).",
    )
    p.add_argument(
        "--remove-positive",
        action="store_true",
        help="If set, remove rows where the value in --data-column is positive (>0).",
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
    horizon_column: str | None,
    horizon_keep: list[str] | None,
    remove_negative: bool,
    remove_positive: bool,
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

    # Remove negative or positive values if requested
    if data_column is not None:
        col = gdf[data_column]

        if remove_negative:
            before = len(gdf)
            gdf = gdf[col >= 0]
            print(
                f"Removed {before - len(gdf)} rows with negative values in '{data_column}'."
            )

        if remove_positive:
            before = len(gdf)
            gdf = gdf[col <= 0]
            print(
                f"Removed {before - len(gdf)} rows with positive values in '{data_column}'."
            )

        if len(gdf) == 0:
            raise SystemExit(
                f"Error: No rows remain after applying value filtering (--remove-negative / --remove-positive)."
            )

    # Filter by horizon column if provided
    if horizon_column is not None and horizon_keep is not None:
        if horizon_column not in gdf.columns:
            raise SystemExit(
                f"Error: column '{horizon_column}' not found in shapefile attributes."
            )
        before_count = len(gdf)
        gdf = gdf[gdf[horizon_column].isin(horizon_keep)]
        after_count = len(gdf)
        print(
            f"Filtered {before_count - after_count} rows not in horizons {horizon_keep}."
        )
        if after_count == 0:
            raise SystemExit(
                f"Error: no rows remain after filtering horizons {horizon_keep}."
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
    return Xc, Yc, xs, ys, (minx, maxy, width, height)


def evaluate_kde_batched(kde, positions, batch_size=1000, show_progress=True):
    """Evaluate scipy.stats.gaussian_kde in batches to reduce memory use."""
    n = positions.shape[1]  # positions is (2, N)
    results = np.empty(n, dtype=np.float64)
    batches = range(0, n, batch_size)

    iterator = tqdm(batches, disable=not show_progress)
    for start in iterator:
        stop = min(start + batch_size, n)
        results[start:stop] = kde.logpdf(positions[:, start:stop])
    return results


def kde_on_grid(
    points_xy: np.ndarray,
    bandwidth: float,
    Xc: np.ndarray,
    Yc: np.ndarray,
    atol: float,
) -> np.ndarray:

    values = np.vstack([points_xy[:, 0], points_xy[:, 1]])
    print("Fitting KDE...")
    kde = gaussian_kde(values, bw_method=bandwidth)
    print("Done\n")

    grid_pts = np.vstack([Xc.ravel().astype(np.float32), Yc.ravel().astype(np.float32)])
    print("Evaluating KDE...")
    # log_density = kde.logpdf(grid_pts)
    log_density = evaluate_kde_batched(kde, grid_pts)
    print("Done\n")

    density = np.exp(
        log_density - log_density.max()
    )  # scale to [0,1] range for numerical stability
    density = density + atol
    return density.reshape(Xc.shape).astype(np.float32)

    # # sklearn expects shape (n_samples, n_features)
    # kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    # kde.fit(points_xy)

    # # Evaluate at grid centers
    # grid_pts = np.column_stack([Xc.ravel(), Yc.ravel()])
    # log_density = kde.score_samples(grid_pts)  # log p(x)
    # density = np.exp(
    #     log_density - log_density.max()
    # )  # scale to [0,1] range for numerical stability
    # density = density + atol
    # return density.reshape(Xc.shape)


# def mask_outside_kernel(
#     points_xy: np.ndarray, Xc: np.ndarray, Yc: np.ndarray, kernel_size: float
# ) -> np.ndarray:
#     # Build KDTree for nearest neighbor distances
#     tree = cKDTree(points_xy)
#     grid_pts = np.column_stack([Xc.ravel(), Yc.ravel()])
#     dists, _ = tree.query(grid_pts, k=1)
#     dists = dists.reshape(Xc.shape)
#     mask = (dists <= 3 * kernel_size).astype(bool)  # 1 inside, 0 outside
#     return mask


def clip_outside_convex_hull(
    density, xs, ys, gdf, buffer_distance=0.0, nodata_value=-9999.0
):
    """
    Set density values to nodata outside the convex hull of geometries in a GeoDataFrame.

    Parameters
    ----------
    density : np.ndarray
        2D array of KDE values (height x width).
    xs, ys : np.ndarray
        1D arrays of X and Y coordinates corresponding to pixel centers.
    gdf : geopandas.GeoDataFrame
        Input geometries (any type).
    buffer_distance : float, optional
        Distance (in CRS units) to expand the hull before masking (default 0).
    nodata_value : float, optional
        Value to assign to masked pixels (default -9999.0).

    Returns
    -------
    np.ndarray
        The density array with values outside the hull set to nodata.
    """
    # Compute convex hull + buffer
    try:
        merged = gdf.union_all()
    except AttributeError:  # for Shapely < 2.0
        merged = gdf.unary_union

    hull = merged.convex_hull.buffer(buffer_distance)

    # Build meshgrid of pixel centers
    X, Y = np.meshgrid(xs, ys)
    points = np.column_stack([X.ravel(), Y.ravel()])

    # Vectorized point-in-polygon test if Shapely 2.x is installed
    try:
        from shapely import contains

        mask = contains(hull, points)
        mask = mask.reshape(density.shape)
    except Exception:
        # Fallback (slower) loop for Shapely < 2.0
        mask = np.zeros_like(density, dtype=bool)
        for i in range(density.shape[0]):
            for j in range(density.shape[1]):
                if hull.contains(Point(X[i, j], Y[i, j])):
                    mask[i, j] = True

    # Apply mask
    density_masked = density.copy()
    density_masked[~mask] = nodata_value

    return density_masked


def main():
    args = parse_args()

    if args.kernel_size <= 0 or args.pixel_size <= 0:
        print("ERROR: --kernel-size and --pixel-size must be > 0.", file=sys.stderr)
        sys.exit(2)

    print("Loading shapefile...")
    points_xy, crs, gdf = read_points(
        args.shapefile,
        args.to_crs,
        args.data_column,
        args.horizon_column,
        args.horizon_keep,
        args.remove_negative,
        args.remove_positive,
    )
    print("Done\n")

    # Bounds in current CRS
    minx, miny, maxx, maxy = gdf.total_bounds
    Xc, Yc, xs, ys, (origin_minx, origin_maxy, width, height) = make_grid(
        (minx, miny, maxx, maxy), pixel_size=args.pixel_size, pad=0.0
    )

    # KDE
    density = kde_on_grid(points_xy, args.kernel_size, Xc, Yc, args.atol)
    nodata_value = -9999.0

    # Mask outside convex hull
    if args.clip_to_hull:
        print("Clipping shapefile...")
        density = clip_outside_convex_hull(
            density,
            xs,
            ys,
            gdf,
            buffer_distance=args.kernel_size,
            nodata_value=nodata_value,
        )
        print("Done \n")

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
        dst.write(density.astype(np.float32), 1)

    print(f"Wrote: {args.output}\n")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Runtime: {(time.time() - start_time):.2f} seconds\n")
