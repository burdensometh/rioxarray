"""
Lazy, window-by-window reprojection of dask-backed DataArrays.

The destination grid (affine, shape, CRS, nodata) is computed once, up front,
without reading any pixels. Each destination block is then warped on its own
from only the source window that overlaps it. Every block uses a *sub-window*
of the single, global destination transform (``dst_affine * translation``),
so blocks cannot disagree about where pixels live and there are no seams.

The approach is the same one used by GDAL's warper (``GDALWarpOperation``) and
:class:`rasterio.vrt.WarpedVRT`: densify the destination block bounds to
reproject them into source coordinates, inflate the resulting source window by
a resampling-kernel dependent halo, fill the parts of the window that fall
outside the source with nodata/mask, and warp the window into the block.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy
import rasterio
import rasterio.warp
import xarray
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window

from rioxarray.exceptions import RioXarrayError

# Number of source pixels that every non-trivial resampling kernel can reach
# beyond the pixel a destination sample maps to.  This mirrors the default
# ``SRC_EXTRA`` padding used by GDAL's warper, which rasterio itself drives.
_RESAMPLING_HALO = {
    Resampling.nearest: 1,
    Resampling.mode: 2,
}
_DEFAULT_HALO = 2

# Bounds densification when reprojecting a destination window back into source
# pixel coordinates (21 intermediate points per edge, rasterio's default).
_DENSIFY_POINTS = 21


def _resampling_halo(resampling: Resampling) -> int:
    """Return the source halo (in pixels) needed by a resampling kernel."""
    return _RESAMPLING_HALO.get(resampling, _DEFAULT_HALO)


def _normalize_dst_chunks(
    chunks: Any, *, dst_height: int, dst_width: int, dtype: Any
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Normalize the user requested output ``chunks`` to (y chunks, x chunks)."""
    import dask.array.core

    if isinstance(chunks, dict):
        y_chunks = chunks.get("y", chunks.get(1, "auto"))
        x_chunks = chunks.get("x", chunks.get(2, "auto"))
        chunks = (y_chunks, x_chunks)
    elif isinstance(chunks, (int, float, str)):
        chunks = (chunks, chunks)
    elif len(chunks) == 1:
        chunks = (chunks[0], chunks[0])
    normalized = dask.array.core.normalize_chunks(
        chunks, shape=(dst_height, dst_width), dtype=dtype
    )
    return tuple(tuple(c) for c in normalized)  # type: ignore[return-value]


def _source_window_for_block(
    *,
    dst_window: Window,
    dst_affine: Affine,
    dst_crs: Any,
    src_affine: Affine,
    src_crs: Any,
    src_height: int,
    src_width: int,
    halo: int,
) -> Optional[Window]:
    """
    Determine the (halo-inclusive) source window overlapping a destination
    block, or ``None`` when the block does not overlap the source at all.
    """
    left, bottom, right, top = rasterio.windows.bounds(dst_window, dst_affine)
    src_left, src_bottom, src_right, src_top = rasterio.warp.transform_bounds(
        dst_crs,
        src_crs,
        left,
        bottom,
        right,
        top,
        densify_pts=_DENSIFY_POINTS,
    )
    # Blocks whose bounds cannot be represented in the source CRS (e.g.
    # outside the projection's valid area) do not overlap the source.
    if not all(
        numpy.isfinite(value)
        for value in (src_left, src_bottom, src_right, src_top)
    ):
        return None
    src_window = rasterio.windows.from_bounds(
        src_left,
        src_bottom,
        src_right,
        src_top,
        src_affine,
    )
    # expand by the halo before any clipping so edge pixels keep the full
    # resampling neighbourhood available, then snap to whole source pixels.
    src_window = Window(
        col_off=src_window.col_off - halo,
        row_off=src_window.row_off - halo,
        width=src_window.width + 2 * halo,
        height=src_window.height + 2 * halo,
    ).round_offsets().round_lengths()
    full_source = Window(0, 0, src_width, src_height)
    try:
        return src_window.intersection(full_source)
    except rasterio.errors.WindowError:
        return None


def _pad_source_window(
    source: numpy.ndarray,
    *,
    edge_pads: tuple[int, int, int, int],
    src_nodata: Optional[float],
) -> numpy.ndarray:
    """
    Pad a realized source window back to its halo-inclusive requested size.

    ``edge_pads`` gives the number of missing pixels on the
    ``(top, left, bottom, right)`` edges. Missing pixels lie outside the
    source raster and are either filled with ``src_nodata`` or, when no
    nodata value exists, masked so GDAL treats them as missing.
    """
    top, left, bottom, right = edge_pads
    if top == left == bottom == right == 0:
        if src_nodata is None and not numpy.ma.is_masked(source):
            return numpy.ma.asarray(source)
        return source
    pad_spec = [(0, 0)] * (source.ndim - 2) + [
        (top, bottom),
        (left, right),
    ]
    if src_nodata is not None:
        return numpy.pad(
            source,
            pad_spec,
            mode="constant",
            constant_values=src_nodata,
        )
    padded = numpy.ma.masked_all(
        (
            *source.shape[:-2],
            source.shape[-2] + top + bottom,
            source.shape[-1] + left + right,
        ),
        dtype=source.dtype,
    )
    padded[
        ...,
        top : top + source.shape[-2],
        left : left + source.shape[-1],
    ] = numpy.ma.asarray(source)
    return padded


def _warp_dst_block(
    *,
    source_buffer: numpy.ndarray,
    buffer_transform: Affine,
    dst_window: Window,
    src_crs: Any,
    src_nodata: Optional[float],
    dst_affine: Affine,
    dst_crs: Any,
    dst_nodata: Optional[float],
    resampling: Resampling,
    warp_kwargs: dict[str, Any],
) -> numpy.ndarray:
    """Warp one destination block from its overlapping source window."""
    # Sub-window of the single global destination transform: blocks never
    # derive their own transform, so they cannot produce seams.
    block_transform = dst_affine * Affine.translation(
        dst_window.col_off, dst_window.row_off
    )
    block_shape = (
        *source_buffer.shape[:-2],
        int(dst_window.height),
        int(dst_window.width),
    )
    if dst_nodata is not None:
        destination = numpy.full(
            block_shape, dst_nodata, dtype=source_buffer.dtype
        )
    else:
        destination = numpy.zeros(block_shape, dtype=source_buffer.dtype)

    rasterio.warp.reproject(
        source=source_buffer,
        destination=destination,
        src_transform=buffer_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=block_transform,
        dst_crs=dst_crs,
        dst_nodata=dst_nodata,
        resampling=resampling,
        **warp_kwargs,
    )
    return destination


def _warp_realized_window(
    source: numpy.ndarray,
    *,
    requested_src_window: Window,
    edge_pads: tuple[int, int, int, int],
    dst_window: Window,
    src_affine: Affine,
    src_crs: Any,
    src_nodata: Optional[float],
    dst_affine: Affine,
    dst_crs: Any,
    dst_nodata: Optional[float],
    resampling: Resampling,
    warp_kwargs: dict[str, Any],
) -> numpy.ndarray:
    """
    Warp a realized source window into a destination block.

    ``source`` holds the in-bounds pixels of the halo-inclusive
    ``requested_src_window`` because the dask graph was sliced to the window
    before execution; pixels spilling past the raster edges are re-added as
    nodata/mask padding via ``edge_pads``.
    """
    source_buffer = _pad_source_window(
        source, edge_pads=edge_pads, src_nodata=src_nodata
    )
    # Transform of the full requested window's top-left corner, which is the
    # origin of the padded buffer.
    buffer_transform = rasterio.windows.transform(
        requested_src_window, src_affine
    )
    return _warp_dst_block(
        source_buffer=source_buffer,
        buffer_transform=buffer_transform,
        dst_window=dst_window,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_affine=dst_affine,
        dst_crs=dst_crs,
        dst_nodata=dst_nodata,
        resampling=resampling,
        warp_kwargs=warp_kwargs,
    )


def supports_lazy_path(
    src_data_array: xarray.DataArray, kwargs: dict[str, Any]
) -> bool:
    """
    The windowed path requires an ordinary affine source transform. Sources
    georeferenced with GCPs, RPCs or a geolocation array are left on the
    existing eager path.
    """
    if any(key in kwargs for key in ("gcps", "rpcs", "src_geoloc_array")):
        return False
    try:
        import dask
    except ImportError:  # pragma: no cover - dask is a test dependency
        return False
    return dask.is_dask_collection(src_data_array.data)


def reproject_lazy(
    src_data_array: xarray.DataArray,
    *,
    dst_crs: Any,
    dst_affine: Affine,
    dst_height: int,
    dst_width: int,
    dst_nodata: Optional[float],
    src_affine: Affine,
    src_nodata: Optional[float],
    resampling: Resampling,
    chunks: Any,
    warp_kwargs: dict[str, Any],
) -> dask.array.Array:
    """
    Build a lazy, block-wise reprojection graph for a dask-backed DataArray.

    No source pixels are read here. For every destination block the
    overlapping source window is determined up front and used to slice the
    dask array *in graph space*, so each output task depends only on the
    source chunks its window actually overlaps.
    """
    try:
        import dask
        import dask.array
    except ImportError as error:  # pragma: no cover - dask is a test dep
        raise RioXarrayError(
            "Lazy reprojection requires the optional 'dask' dependency."
        ) from error

    src_dask = src_data_array.data
    src_height, src_width = src_data_array.rio.shape
    src_crs = src_data_array.rio.crs
    halo = _resampling_halo(resampling)
    y_chunks, x_chunks = _normalize_dst_chunks(
        chunks,
        dst_height=dst_height,
        dst_width=dst_width,
        dtype=src_data_array.dtype,
    )
    leading_shape = src_dask.shape[:-2]
    delayed_warp = dask.delayed(_warp_realized_window, pure=True)

    block_arrays: list[list[Any]] = []
    y_offset = 0
    for block_height in y_chunks:
        row_blocks: list[Any] = []
        x_offset = 0
        for block_width in x_chunks:
            dst_window = Window(
                col_off=x_offset,
                row_off=y_offset,
                width=int(block_width),
                height=int(block_height),
            )
            src_window = _source_window_for_block(
                dst_window=dst_window,
                dst_affine=dst_affine,
                dst_crs=dst_crs,
                src_affine=src_affine,
                src_crs=src_crs,
                src_height=src_height,
                src_width=src_width,
                halo=halo,
            )
            block_shape = (
                *leading_shape,
                int(block_height),
                int(block_width),
            )
            if src_window is None:
                # No overlap: a nodata-only block with no dependency on any
                # source chunk.
                if dst_nodata is not None:
                    block_data = dask.array.full(
                        block_shape, dst_nodata, dtype=src_dask.dtype
                    )
                else:
                    block_data = dask.array.zeros(
                        block_shape, dtype=src_dask.dtype
                    )
            else:
                # Slice the source graph first: only chunks overlapping this
                # window become dependencies of the block's warp task.
                row0 = max(int(src_window.row_off), 0)
                col0 = max(int(src_window.col_off), 0)
                row1 = min(
                    int(src_window.row_off + src_window.height), src_height
                )
                col1 = min(
                    int(src_window.col_off + src_window.width), src_width
                )
                # Pixels missing because the window spills over each edge of
                # the source; re-added as nodata/mask inside the worker.
                edge_pads = (
                    row0 - int(src_window.row_off),  # top
                    col0 - int(src_window.col_off),  # left
                    int(src_window.row_off + src_window.height)
                    - row1,  # bottom
                    int(src_window.col_off + src_window.width)
                    - col1,  # right
                )
                source_slice = src_dask[..., row0:row1, col0:col1]
                delayed_block = delayed_warp(
                    source_slice,
                    requested_src_window=src_window,
                    edge_pads=edge_pads,
                    dst_window=dst_window,
                    src_affine=src_affine,
                    src_crs=src_crs,
                    src_nodata=src_nodata,
                    dst_affine=dst_affine,
                    dst_crs=dst_crs,
                    dst_nodata=dst_nodata,
                    resampling=resampling,
                    warp_kwargs=warp_kwargs,
                )
                block_data = dask.array.from_delayed(
                    delayed_block,
                    shape=block_shape,
                    dtype=src_dask.dtype,
                )
            row_blocks.append(block_data)
            x_offset += int(block_width)
        block_arrays.append(row_blocks)
        y_offset += int(block_height)

    return dask.array.block(block_arrays)
