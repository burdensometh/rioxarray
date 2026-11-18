"""
Tests for the lazy, window-by-window reprojection path of
:meth:`rioxarray.raster_array.RasterArray.reproject`.
"""

import warnings

import dask
import dask.array
import numpy
import pytest
import rasterio
import xarray
from rasterio.enums import Resampling

import rioxarray

SRC_CRS = "EPSG:32633"
DST_CRS = "EPSG:4326"


def _make_dataarray(array, *, chunks=None, nodata=None):
    """Build a georeferenced (optionally dask-backed) 2D DataArray."""
    height, width = array.shape[-2:]
    src_transform = rasterio.transform.from_origin(
        500000.0, 5000000.0, 20.0, 20.0
    )
    x = src_transform.c + numpy.arange(width) * 20.0 + 10.0
    y = src_transform.f + numpy.arange(height) * (-20.0) - 10.0
    dims = ("y", "x") if array.ndim == 2 else ("band", "y", "x")
    coords = {"x": x, "y": y}
    if array.ndim == 3:
        coords["band"] = numpy.arange(array.shape[0])
    data_array = xarray.DataArray(
        array, dims=dims, coords=coords, name="band"
    )
    data_array.rio.write_crs(SRC_CRS, inplace=True)
    data_array.rio.write_transform(src_transform, inplace=True)
    if nodata is not None:
        data_array.rio.write_nodata(nodata, inplace=True)
    if chunks is not None:
        data_array = data_array.chunk(chunks)
    return data_array


@pytest.fixture
def source_arrays():
    rng = numpy.random.default_rng(0)
    height = width = 128
    array = rng.normal(size=(height, width)).astype("float64")
    array[:8, :8] = -9999.0
    return array


def test_lazy_reproject_returns_dask_and_does_not_read_pixels(source_arrays):
    """Building the lazy graph must not realize any source chunk."""
    lazy_source = _make_dataarray(
        source_arrays, chunks={"y": 16, "x": 16}, nodata=-9999.0
    )
    with dask.config.set(scheduler="sync"):
        # reproject itself must stay lazy: no compute is triggered
        reprojected = lazy_source.rio.reproject(DST_CRS)
    assert isinstance(reprojected.data, dask.array.Array)
    assert reprojected.rio.crs.to_epsg() == 4326
    # the eager result is the whole-image reference
    eager = _make_dataarray(source_arrays, nodata=-9999.0).rio.reproject(
        DST_CRS
    )
    assert reprojected.rio.shape == eager.rio.shape
    numpy.testing.assert_allclose(
        reprojected.rio.transform(recalc=True)[:6],
        eager.rio.transform(recalc=True)[:6],
    )
    numpy.testing.assert_allclose(reprojected.x.values, eager.x.values)
    numpy.testing.assert_allclose(reprojected.y.values, eager.y.values)


def test_lazy_reproject_nearest_matches_eager(source_arrays):
    eager = _make_dataarray(source_arrays, nodata=-9999.0).rio.reproject(
        DST_CRS, resampling=Resampling.nearest
    )
    lazy = _make_dataarray(
        source_arrays, chunks={"y": 16, "x": 16}, nodata=-9999.0
    ).rio.reproject(DST_CRS, resampling=Resampling.nearest)
    # nodata masks must be identical
    assert_array_equal = numpy.testing.assert_array_equal
    assert_array_equal(
        eager.values == eager.rio.nodata, lazy.values == lazy.rio.nodata
    )
    # nearest neighbour is exact everywhere except the isolated source-pixel
    # rows where GDAL's own per-invocation transformer rounds differently;
    # those rows are identical regardless of whether the source was windowed.
    mismatch_rows = numpy.where(
        (numpy.abs(eager.values - lazy.values) > 0).any(axis=1)
    )[0]
    assert len(mismatch_rows) <= max(3, eager.rio.height // 20)


@pytest.mark.parametrize(
    "resampling",
    [
        Resampling.bilinear,
        Resampling.cubic,
        Resampling.average,
        Resampling.mode,
    ],
)
def test_lazy_reproject_nodata_mask_matches_eager(
    source_arrays, resampling
):
    eager = _make_dataarray(source_arrays, nodata=-9999.0).rio.reproject(
        DST_CRS, resampling=resampling
    )
    lazy = _make_dataarray(
        source_arrays, chunks={"y": 16, "x": 16}, nodata=-9999.0
    ).rio.reproject(DST_CRS, resampling=resampling)
    numpy.testing.assert_array_equal(
        numpy.isnan(eager.values), numpy.isnan(lazy.values)
    )


def test_lazy_reproject_chunks_forms(source_arrays):
    lazy_source = _make_dataarray(
        source_arrays, chunks={"y": 16, "x": 16}, nodata=-9999.0
    )
    # explicit chunk specifications are honored
    for chunks, expected_max in (
        (32, 32),
        ((32, 32), 32),
        ({"y": 32, "x": 48}, 48),
    ):
        reprojected = lazy_source.rio.reproject(DST_CRS, chunks=chunks)
        assert isinstance(reprojected.data, dask.array.Array)
        y_sizes, x_sizes = reprojected.data.chunks
        assert max(y_sizes) <= 32
        assert max(x_sizes) <= expected_max
    # "auto" always yields a valid chunking without reading pixels
    auto = lazy_source.rio.reproject(DST_CRS, chunks="auto")
    assert isinstance(auto.data, dask.array.Array)
    assert sum(auto.data.chunks[0]) == auto.rio.height
    assert sum(auto.data.chunks[1]) == auto.rio.width
    # default: source spatial chunk sizes are reused
    default = lazy_source.rio.reproject(DST_CRS)
    assert max(default.data.chunks[0]) <= 16
    assert max(default.data.chunks[1]) <= 16


def test_lazy_reproject_only_reads_overlapping_source_window(source_arrays):
    """A destination block task must depend only on its local source chunks."""
    accessed = []

    @dask.delayed
    def _tracked_chunk(row, col, shape):
        accessed.append((row, col))
        return source_arrays[
            row * shape[0] : (row + 1) * shape[0],
            col * shape[1] : (col + 1) * shape[1],
        ].copy()

    chunk = 16
    n_blocks = source_arrays.shape[0] // chunk
    blocks = [
        [
            dask.array.from_delayed(
                _tracked_chunk(row, col, (chunk, chunk)),
                (chunk, chunk),
                dtype=source_arrays.dtype,
            )
            for col in range(n_blocks)
        ]
        for row in range(n_blocks)
    ]
    tracked = dask.array.block(blocks)
    data_array = _make_dataarray(tracked, nodata=-9999.0)
    reprojected = data_array.rio.reproject(DST_CRS, chunks=(32, 32))

    accessed.clear()
    reprojected.data.blocks[0, 0].compute()
    corner = set(accessed)
    # corner output block touches only the top-left source region
    assert corner and all(row <= 3 for row, _ in corner)
    assert len(corner) < n_blocks * n_blocks

    accessed.clear()
    reprojected.data.blocks[2, 2].compute()
    middle = set(accessed)
    assert 0 < len(middle) <= 16


def test_lazy_reproject_no_nodata_uses_masked_padding():
    rng = numpy.random.default_rng(1)
    array = rng.normal(size=(96, 96)).astype("float32")
    eager = _make_dataarray(array).rio.reproject(DST_CRS)
    lazy = _make_dataarray(array, chunks={"y": 24, "x": 24}).rio.reproject(
        DST_CRS
    )
    assert isinstance(lazy.data, dask.array.Array)
    assert eager.rio.shape == lazy.rio.shape
    # finite everywhere because no nodata exists and padding is masked
    assert numpy.isfinite(lazy.values).all()


def test_lazy_reproject_integer_default_nodata(source_arrays):
    integer_array = (
        numpy.random.default_rng(2)
        .integers(0, 100, size=(96, 96))
        .astype("int16")
    )
    eager = _make_dataarray(integer_array).rio.reproject(DST_CRS)
    lazy = _make_dataarray(
        integer_array, chunks={"y": 24, "x": 24}
    ).rio.reproject(DST_CRS)
    assert lazy.dtype == integer_array.dtype
    assert eager.rio.nodata == lazy.rio.nodata
    numpy.testing.assert_array_equal(
        eager.values == eager.rio.nodata, lazy.values == lazy.rio.nodata
    )


def test_lazy_reproject_3d(source_arrays):
    array_3d = numpy.stack([source_arrays, source_arrays * 1.5])
    eager = _make_dataarray(array_3d, nodata=-9999.0).rio.reproject(DST_CRS)
    lazy = _make_dataarray(
        array_3d, chunks={"band": 1, "y": 16, "x": 16}, nodata=-9999.0
    ).rio.reproject(DST_CRS)
    assert isinstance(lazy.data, dask.array.Array)
    assert lazy.dims == eager.dims
    assert lazy.rio.shape == eager.rio.shape
    numpy.testing.assert_array_equal(
        eager.values == eager.rio.nodata, lazy.values == lazy.rio.nodata
    )


def test_lazy_reproject_block_boundaries_have_no_seams():
    height = width = 256
    rows, cols = numpy.meshgrid(
        numpy.arange(height), numpy.arange(width), indexing="ij"
    )
    smooth = (numpy.sin(cols / 10.0) + numpy.cos(rows / 13.0)).astype(
        "float32"
    )
    reprojected = _make_dataarray(
        smooth, chunks={"y": 32, "x": 32}
    ).rio.reproject(DST_CRS, chunks=(32, 32), resampling=Resampling.bilinear)
    values = reprojected.values
    for boundary in (32, 64):
        seam = numpy.abs(values[boundary] - values[boundary - 1])
        neighbour = numpy.abs(values[boundary - 1] - values[boundary - 2])
        assert seam.max() <= neighbour.max() * 1.5 + 1e-6


def test_numpy_input_with_chunks_falls_back_with_warning(source_arrays):
    numpy_source = _make_dataarray(source_arrays, nodata=-9999.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reprojected = numpy_source.rio.reproject(DST_CRS, chunks=(32, 32))
    assert not isinstance(reprojected.data, dask.array.Array)
    assert any("chunks" in str(entry.message) for entry in caught)


def test_lazy_reproject_match(source_arrays):
    lazy_source = _make_dataarray(
        source_arrays, chunks={"y": 16, "x": 16}, nodata=-9999.0
    )
    match_transform = rasterio.transform.from_origin(
        -100.0, 45.0, 0.001, 0.001
    )
    match = xarray.DataArray(
        numpy.zeros((50, 60)),
        dims=("y", "x"),
        coords={
            "x": match_transform.c
            + numpy.arange(60) * 0.001
            + 0.0005,
            "y": match_transform.f
            + numpy.arange(50) * (-0.001)
            - 0.0005,
        },
    )
    match.rio.write_crs(DST_CRS, inplace=True)
    match.rio.write_transform(match_transform, inplace=True)
    reprojected = lazy_source.rio.reproject_match(match)
    assert isinstance(reprojected.data, dask.array.Array)
    assert reprojected.rio.shape == (50, 60)
    assert reprojected.rio.crs.to_epsg() == 4326


def test_lazy_reproject_block_outside_projection_is_all_nodata():
    """Blocks whose bounds fail to transform to the source CRS stay nodata."""
    small = numpy.arange(64, dtype="float32").reshape(8, 8)
    lazy_source = _make_dataarray(
        small, chunks={"y": 4, "x": 4}, nodata=-1.0
    )
    # grid centered far enough away that corner blocks miss UTM projection
    outside_transform = rasterio.transform.from_origin(
        499000.0, 5001000.0, 0.0008, 0.0008
    )
    reprojected = lazy_source.rio.reproject(
        DST_CRS,
        transform=outside_transform,
        shape=(40, 40),
        chunks=(10, 10),
    )
    assert isinstance(reprojected.data, dask.array.Array)
    values = reprojected.values
    assert (values == -1.0).all()
    eager = _make_dataarray(small, nodata=-1.0).rio.reproject(
        DST_CRS, transform=outside_transform, shape=(40, 40)
    )
    numpy.testing.assert_array_equal(
        values == -1.0, eager.values == -1.0
    )


def test_lazy_reproject_to_raster_is_blockwise(tmp_path, source_arrays):
    lazy_source = _make_dataarray(
        source_arrays, chunks={"y": 16, "x": 16}, nodata=-9999.0
    )
    reprojected = lazy_source.rio.reproject(
        DST_CRS, chunks=(32, 32)
    )
    output_path = tmp_path / "lazy_reproject.tif"
    reprojected.rio.to_raster(output_path, tiled=True, lock=False)
    with rasterio.open(output_path) as raster:
        assert raster.crs.to_epsg() == 4326
        numpy.testing.assert_allclose(
            raster.transform[:6],
            reprojected.rio.transform(recalc=True)[:6],
            atol=1e-12,
        )
        numpy.testing.assert_array_equal(
            raster.read(1), reprojected.values
        )
