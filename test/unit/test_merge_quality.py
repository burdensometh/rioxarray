"""Tests for quality-priority merging of multiple scenes."""

import numpy
import pytest
import rasterio
import xarray
from rasterio.transform import from_origin

import rioxarray  # noqa: F401
from rioxarray.exceptions import QualityLayerError
from rioxarray.merge import merge_datasets, merge_datasets_quality


def _make_dataset(
    quality,
    band_a,
    band_b,
    *,
    x0=0.0,
    y0=2.0,
    res=1.0,
    quality_nodata=-1,
    nodata=-9999.0,
    dims=("y", "x"),
    crs="EPSG:32633",
):
    """Build a small dataset with two bands and a quality layer."""
    quality = numpy.asarray(quality)
    height, width = quality.shape
    x = x0 + res / 2 + numpy.arange(width) * res
    y = y0 - res / 2 - numpy.arange(height) * res
    y_dim, x_dim = dims
    dataset = xarray.Dataset(
        {
            "a": (
                (y_dim, x_dim),
                numpy.asarray(band_a, dtype="float32"),
                {"_FillValue": nodata},
            ),
            "b": (
                (y_dim, x_dim),
                numpy.asarray(band_b, dtype="float32"),
                {"_FillValue": nodata},
            ),
            "quality": (
                (y_dim, x_dim),
                numpy.asarray(quality, dtype="int16"),
                {"_FillValue": quality_nodata},
            ),
        },
        coords={y_dim: y, x_dim: x},
    )
    dataset = dataset.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
    dataset.rio.write_crs(crs, inplace=True)
    dataset.rio.write_transform(from_origin(x0, y0, res, res))
    return dataset


def test_merge_quality__single_source_per_pixel():
    """Every band at a pixel must come from the winning source."""
    first = _make_dataset(
        [[0, 0], [5, 5]],
        numpy.full((2, 2), 10.0),
        numpy.full((2, 2), 20.0),
    )
    second = _make_dataset(
        [[2, 2], [2, 2]],
        numpy.full((2, 2), 11.0),
        numpy.full((2, 2), 21.0),
    )
    merged = merge_datasets_quality([first, second], "quality")

    assert merged["source_index"].values.tolist() == [[0, 0], [1, 1]]
    assert merged["source_valid"].values.tolist() == [[1, 1], [1, 1]]
    # both bands are taken from the same source in each pixel
    assert merged["a"].values.tolist() == [[10.0, 10.0], [11.0, 11.0]]
    assert merged["b"].values.tolist() == [[20.0, 20.0], [21.0, 21.0]]
    assert merged["quality"].values.tolist() == [[0, 0], [2, 2]]


def test_merge_quality__higher_is_better():
    first = _make_dataset(
        numpy.zeros((2, 2)), numpy.full((2, 2), 10.0), numpy.full((2, 2), 1.0)
    )
    second = _make_dataset(
        numpy.full((2, 2), 7), numpy.full((2, 2), 11.0), numpy.full((2, 2), 1.0)
    )

    merged = merge_datasets_quality([first, second], "quality", higher_is_better=True)
    assert (merged["source_index"].values == 1).all()
    assert (merged["a"].values == 11.0).all()


def test_merge_quality__invalid_required_band_cannot_win():
    """A source whose band is invalid at a pixel must not win there."""
    best = _make_dataset(
        [[0, 0], [0, 0]],
        numpy.full((2, 2), 10.0),
        [[20.0, -9999.0], [20.0, 20.0]],
    )
    other = _make_dataset(
        numpy.full((2, 2), 5),
        numpy.full((2, 2), 11.0),
        numpy.full((2, 2), 21.0),
    )
    merged = merge_datasets_quality([best, other], "quality")

    # the best-quality source is excluded only where band 'b' is invalid
    assert merged["source_index"].values.tolist() == [[0, 1], [0, 0]]
    assert merged["b"].values.tolist() == [[20.0, 21.0], [20.0, 20.0]]


def test_merge_quality__invalid_quality_cannot_win():
    first = _make_dataset(
        [[-1, 0], [0, 0]], numpy.full((2, 2), 10.0), numpy.full((2, 2), 20.0)
    )
    second = _make_dataset(
        numpy.full((2, 2), 4), numpy.full((2, 2), 11.0), numpy.full((2, 2), 21.0)
    )
    merged = merge_datasets_quality([first, second], "quality")
    assert merged["source_index"].values.tolist() == [[1, 0], [0, 0]]


def test_merge_quality__ties_are_stable():
    first = _make_dataset(
        numpy.full((2, 2), 3), numpy.full((2, 2), 10.0), numpy.full((2, 2), 1.0)
    )
    second = _make_dataset(
        numpy.full((2, 2), 3), numpy.full((2, 2), 11.0), numpy.full((2, 2), 1.0)
    )

    merged = merge_datasets_quality([first, second], "quality")
    assert (merged["source_index"].values == 0).all()

    declared = merge_datasets_quality([first, second], "quality", priority=[5.0, 1.0])
    assert (declared["source_index"].values == 1).all()


def test_merge_quality__no_valid_source():
    invalid = _make_dataset(
        numpy.full((2, 2), -1),
        numpy.full((2, 2), -9999.0),
        numpy.full((2, 2), -9999.0),
    )
    merged = merge_datasets_quality([invalid], "quality")
    assert (merged["source_index"].values < 0).all()
    assert (merged["source_valid"].values < 0).all()


def test_merge_quality__quality_is_not_interpolated():
    """Quality classes must use nearest neighbour, never blended values."""
    coarse = _make_dataset(
        [[0, 1], [2, 3]], numpy.full((2, 2), 10.0), numpy.full((2, 2), 20.0)
    )
    fine = _make_dataset(
        numpy.arange(4).reshape(2, 2) + 10,
        numpy.full((2, 2), 11.0),
        numpy.full((2, 2), 21.0),
        res=0.5,
    )
    merged = merge_datasets_quality([coarse, fine], "quality", res=0.5)
    assert set(numpy.unique(merged["quality"].values)).issubset(
        {0, 1, 2, 3, 10, 11, 12, 13}
    )


def test_merge_quality__quality_and_data_grid_must_match():
    """A band on a different grid than its quality layer is rejected."""
    x = numpy.arange(2) + 0.5
    y = 2 - numpy.arange(2) - 0.5
    dataset = xarray.Dataset(
        {
            "a": (("lat", "lon"), numpy.ones((2, 2), dtype="float32")),
            "quality": (("y", "x"), numpy.zeros((2, 2), dtype="int16")),
        },
        coords={
            "lat": ("lat", y, {"standard_name": "latitude"}),
            "lon": ("lon", x, {"standard_name": "longitude"}),
            "y": y,
            "x": x,
        },
    )
    dataset.rio.write_crs("EPSG:32633", inplace=True)
    dataset.rio.write_transform(from_origin(0, 2, 1, 1))
    with pytest.raises(QualityLayerError, match="not on the same grid"):
        merge_datasets_quality([dataset], "quality")


def test_merge_quality__cropped_result_is_consistent(tmp_path):
    first = _make_dataset(
        [[0, 0], [5, 5]], numpy.full((2, 2), 10.0), numpy.full((2, 2), 20.0)
    )
    second = _make_dataset(
        [[2, 2], [2, 2]],
        numpy.full((2, 2), 11.0),
        numpy.full((2, 2), 21.0),
        x0=2.0,
    )
    merged = merge_datasets_quality([first, second], "quality")
    geometry = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1.5, 0], [1.5, 2], [0, 2], [0, 0]]],
    }
    clipped = merged.rio.clip([geometry], drop=False)

    valid = clipped["source_valid"].values == 1
    numpy.testing.assert_array_equal(
        clipped["a"].values[valid], merged["a"].values[valid]
    )
    numpy.testing.assert_array_equal(
        clipped["source_index"].values[valid],
        merged["source_index"].values[valid],
    )
    # nodata is shared by every layer, so band and metadata stay in step
    nodata_values = {layer.rio.nodata for layer in clipped.data_vars.values()}
    assert len(nodata_values) == 1

    raster_path = tmp_path / "merged.tif"
    merged.rio.to_raster(raster_path)
    with rasterio.open(raster_path) as rds:
        assert len(set(rds.nodatavals)) == 1
        bands = rds.read()
    for index, name in enumerate(merged.data_vars, start=1):
        assert bands[index - 1].shape == merged[name].shape


def test_merge_quality__nonstandard_dimension_names():
    first = _make_dataset(
        [[0, 0], [5, 5]],
        numpy.full((2, 2), 10.0),
        numpy.full((2, 2), 20.0),
        dims=("lat", "lon"),
    )
    second = _make_dataset(
        [[2, 2], [2, 2]],
        numpy.full((2, 2), 11.0),
        numpy.full((2, 2), 21.0),
        dims=("lat", "lon"),
    )
    merged = merge_datasets_quality([first, second], "quality")
    assert merged["a"].dims == ("lat", "lon")
    assert merged["source_index"].dims == ("lat", "lon")


def test_merge_quality__reprojects_onto_shared_grid():
    first = _make_dataset(
        [[0, 0], [5, 5]], numpy.full((2, 2), 10.0), numpy.full((2, 2), 20.0)
    )
    second = _make_dataset(
        [[2, 2], [2, 2]],
        numpy.full((2, 2), 11.0),
        numpy.full((2, 2), 21.0),
        res=0.5,
    )
    merged = merge_datasets_quality([first, second], "quality", res=1.0)
    assert merged.rio.shape == (2, 2)
    assert set(numpy.unique(merged["source_index"].values)).issubset({0, 1})


def test_merge_quality__merge_datasets_integration():
    first = _make_dataset(
        [[0, 0], [5, 5]], numpy.full((2, 2), 10.0), numpy.full((2, 2), 20.0)
    )
    second = _make_dataset(
        [[2, 2], [2, 2]], numpy.full((2, 2), 11.0), numpy.full((2, 2), 21.0)
    )
    merged = merge_datasets([first, second], quality="quality")
    assert "source_index" in merged.data_vars
    assert "source_valid" in merged.data_vars
    assert merged["source_index"].values.tolist() == [[0, 0], [1, 1]]
    with pytest.raises(QualityLayerError, match="do not apply"):
        merge_datasets([first, second], quality="quality", method="max")


def test_merge_quality__squeezes_single_band_dimension():
    dataset = _make_dataset(
        numpy.zeros((2, 2)), numpy.full((2, 2), 10.0), numpy.full((2, 2), 20.0)
    )
    stacked = dataset.copy(deep=True)
    stacked["a"] = stacked["a"].expand_dims(band=[1])
    merged = merge_datasets_quality([stacked, dataset], "quality")
    assert merged["a"].dims == ("y", "x")
    assert merged["a"].values.tolist() == [[10.0, 10.0], [10.0, 10.0]]


def test_merge_quality__rejects_multiband_variables():
    dataset = _make_dataset(
        numpy.zeros((2, 2)), numpy.full((2, 2), 10.0), numpy.full((2, 2), 20.0)
    )
    dataset["a"] = dataset["a"].expand_dims(band=[1, 2])
    with pytest.raises(QualityLayerError, match="extra non-spatial dimensions"):
        merge_datasets_quality([dataset], "quality")


def test_merge_quality__boolean_bands_are_written_as_bytes():
    dataset = _make_dataset(
        numpy.zeros((2, 2)), numpy.full((2, 2), 10.0), numpy.full((2, 2), 20.0)
    )
    dataset["a"] = (
        ("y", "x"),
        numpy.array([[True, False], [False, True]]),
    )
    merged = merge_datasets_quality([dataset], "quality")
    assert merged["a"].dtype == numpy.dtype("uint8")
    assert merged["a"].values.tolist() == [[1, 0], [0, 1]]


def test_merge_quality__options_require_quality():
    first = _make_dataset(
        numpy.zeros((2, 2)), numpy.full((2, 2), 10.0), numpy.full((2, 2), 20.0)
    )
    with pytest.raises(QualityLayerError, match="require 'quality'"):
        merge_datasets([first], bands=["a"])


def test_merge_quality__validates_arguments():
    dataset = _make_dataset(
        numpy.zeros((2, 2)), numpy.full((2, 2), 10.0), numpy.full((2, 2), 20.0)
    )
    with pytest.raises(QualityLayerError, match="At least one dataset"):
        merge_datasets_quality([], "quality")
    with pytest.raises(QualityLayerError, match="missing required variable"):
        merge_datasets_quality([dataset], "quality", bands=["nope"])
    with pytest.raises(QualityLayerError, match="reserved"):
        merge_datasets_quality([dataset], "quality", bands=["a"], source_index_name="a")
    with pytest.raises(QualityLayerError, match="cannot also be merged as a band"):
        merge_datasets_quality([dataset], "quality", bands=["quality"])
    with pytest.raises(QualityLayerError, match="one value per dataset"):
        merge_datasets_quality([dataset], "quality", priority=[1.0, 2.0])


def test_merge_quality__nodata_collision_is_rejected():
    dataset = _make_dataset(
        numpy.zeros((2, 2)), numpy.full((2, 2), 10.0), numpy.full((2, 2), 20.0)
    )
    with pytest.raises(QualityLayerError, match="collides"):
        merge_datasets_quality([dataset], "quality", nodata=0)


def test_merge_quality__rejects_impossible_output_grid():
    """Incomparable CRSs must fail clearly instead of exhausting memory."""
    first = _make_dataset(
        numpy.zeros((2, 2)), numpy.full((2, 2), 10.0), numpy.full((2, 2), 20.0)
    )
    other = _make_dataset(
        numpy.ones((2, 2)),
        numpy.full((2, 2), 11.0),
        numpy.full((2, 2), 21.0),
        crs="EPSG:4326",
    )
    with pytest.raises(QualityLayerError, match="too large"):
        merge_datasets_quality([first, other], "quality")
