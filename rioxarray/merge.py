"""
This module allows you to merge xarray Datasets/DataArrays
geospatially with the `rasterio.merge` module.
"""

from collections.abc import Hashable, Sequence
from typing import Callable, Optional, Union

import numpy
import rasterio.warp
from affine import Affine
from rasterio.crs import CRS
from rasterio.dtypes import dtype_rev
from rasterio.enums import Resampling
from rasterio.merge import merge as _rio_merge
from xarray import DataArray, Dataset, IndexVariable

from rioxarray._spatial_utils import (
    _NODATA_DTYPE_MAP,
    _convert_str_to_resampling,
    _get_nonspatial_coords,
    _get_spatial_dims,
    _make_coords,
)
from rioxarray.exceptions import (
    MissingCRS,
    MissingSpatialDimensionError,
    QualityLayerError,
)
from rioxarray.raster_writer import _ensure_nodata_dtype


class RasterioDatasetDuck:
    """
    This class is to provide the attributes and methods necessary
    to make the :func:`rasterio.merge.merge` function think that
    the :obj:`xarray.DataArray` is a :obj:`rasterio.io.DatasetReader`.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, xds: DataArray):
        self._xds = xds
        self.crs = xds.rio.crs
        self.bounds = xds.rio.bounds(recalc=True)
        self.count = int(xds.rio.count)
        self.dtypes = [xds.dtype]
        self.name = xds.name
        self.nodatavals = [xds.rio.nodata]
        res = xds.rio.resolution(recalc=True)
        self.res = (abs(res[0]), abs(res[1]))
        self.transform = xds.rio.transform(recalc=True)
        # profile is only used for writing to a file.
        # This never happens with rioxarray merge.
        self.profile: dict = {}

    def colormap(self, *args, **kwargs) -> None:
        """
        colormap is only used for writing to a file.
        This never happens with rioxarray merge.
        """
        # pylint: disable=unused-argument
        return None

    def read(self, window, out_shape, *args, **kwargs) -> numpy.ma.MaskedArray:
        # pylint: disable=unused-argument
        """
        This method is meant to be used by the rasterio.merge.merge function.
        """
        data_window = self._xds.rio.isel_window(window)
        if data_window.shape != out_shape:
            # in this section, the data is geographically the same
            # however it is not the same dimensions as requested
            # so need to resample to the requested shape
            if len(out_shape) == 3:
                _, out_height, out_width = out_shape
            else:
                out_height, out_width = out_shape
            data_window = self._xds.rio.reproject(
                self._xds.rio.crs,
                transform=self.transform,
                shape=(out_height, out_width),
            )

        nodata = self.nodatavals[0]
        mask = False
        fill_value = None
        if nodata is not None and numpy.isnan(nodata):
            mask = numpy.isnan(data_window)
        elif nodata is not None:
            mask = data_window == nodata
            fill_value = nodata

        # make sure the returned shape matches
        # the expected shape. This can be the case
        # when the xarray dataset was squeezed to 2D beforehand
        if len(out_shape) == 3 and len(data_window.shape) == 2:
            data_window = data_window.values.reshape((1, out_height, out_width))

        return numpy.ma.array(
            data_window, mask=mask, fill_value=fill_value, dtype=self.dtypes[0]
        )


def merge_arrays(
    dataarrays: Sequence[DataArray],
    *,
    bounds: Optional[tuple] = None,
    res: Optional[tuple] = None,
    nodata: Optional[float] = None,
    precision: Optional[float] = None,
    method: Union[str, Callable, None] = None,
    crs: Optional[CRS] = None,
    parse_coordinates: bool = True,
) -> DataArray:
    """
    Merge data arrays geospatially.

    Uses :func:`rasterio.merge.merge`

    .. versionadded:: 0.2 crs

    Parameters
    ----------
    dataarrays: list[xarray.DataArray]
        List of multiple xarray.DataArray with all geo attributes.
        The first one is assumed to have the same
        CRS, dtype, and dimensions as the others in the array.
    bounds: tuple, optional
        Bounds of the output image (left, bottom, right, top).
        If not set, bounds are determined from bounds of input DataArrays.
    res: tuple, optional
        Output resolution in units of coordinate reference system.
        If not set, the resolution of the first DataArray is used.
        If a single value is passed, output pixels will be square.
    nodata: float, optional
        nodata value to use in output file.
        If not set, uses the nodata value in the first input DataArray.
    precision: float, optional
        Number of decimal points of precision when computing inverse transform.
    method: str or callable, optional
        See :func:`rasterio.merge.merge` for details.
    crs: rasterio.crs.CRS, optional
        Output CRS. If not set, the CRS of the first DataArray is used.
    parse_coordinates: bool, optional
        If False, it will disable loading spatial coordinates.

    Returns
    -------
    :obj:`xarray.DataArray`:
        The geospatially merged data.
    """
    input_kwargs = {
        "bounds": bounds,
        "res": res,
        "nodata": nodata,
        "precision": precision,
        "method": method,
    }

    if crs is None:
        crs = dataarrays[0].rio.crs
    if res is None:
        res = tuple(abs(res_val) for res_val in dataarrays[0].rio.resolution())

    # prepare the duck arrays
    rioduckarrays = []
    for dataarray in dataarrays:
        da_res = tuple(abs(res_val) for res_val in dataarray.rio.resolution())
        if da_res != res or dataarray.rio.crs != crs:
            rioduckarrays.append(
                RasterioDatasetDuck(
                    dataarray.rio.reproject(dst_crs=crs, resolution=res)
                )
            )
        else:
            rioduckarrays.append(RasterioDatasetDuck(dataarray))

    # use rasterio to merge
    merged_data, merged_transform = _rio_merge(
        rioduckarrays,
        **{key: val for key, val in input_kwargs.items() if val is not None},
    )
    # generate merged data array
    representative_array = rioduckarrays[0]._xds
    if parse_coordinates:
        coords = _make_coords(
            src_data_array=representative_array,
            dst_affine=merged_transform,
            dst_width=merged_data.shape[-1],
            dst_height=merged_data.shape[-2],
        )
        if (
            representative_array.rio.x_dim != "x"
            and "x" in coords
            and coords["x"].ndim == 1
        ):
            coords[representative_array.rio.x_dim] = IndexVariable(
                representative_array.rio.x_dim, coords.pop("x")
            )
        if (
            representative_array.rio.y_dim != "y"
            and "y" in coords
            and coords["y"].ndim == 1
        ):
            coords[representative_array.rio.y_dim] = IndexVariable(
                representative_array.rio.y_dim, coords.pop("y")
            )
    else:
        coords = _get_nonspatial_coords(representative_array)

    # make sure the output merged data shape is 2D if the
    # original data was 2D. this can happen if the
    # xarray datasarray was squeezed.
    if len(merged_data.shape) == 3 and len(representative_array.shape) == 2:
        merged_data = merged_data.squeeze()

    xda = DataArray(
        name=representative_array.name,
        data=merged_data,
        coords=coords,
        dims=tuple(representative_array.dims),
        attrs=representative_array.attrs,
    )
    xda.encoding = representative_array.encoding.copy()
    xda.rio.write_nodata(
        nodata if nodata is not None else representative_array.rio.nodata, inplace=True
    )
    xda.rio.write_crs(
        representative_array.rio.crs,
        grid_mapping_name=representative_array.rio.grid_mapping,
        inplace=True,
    )
    xda.rio.write_transform(
        merged_transform,
        grid_mapping_name=representative_array.rio.grid_mapping,
        inplace=True,
    )
    return xda


def merge_datasets(
    datasets: Sequence[Dataset],
    *,
    bounds: Optional[tuple] = None,
    res: Optional[tuple] = None,
    nodata: Optional[float] = None,
    precision: Optional[float] = None,
    method: Union[str, Callable, None] = None,
    crs: Optional[CRS] = None,
    quality: Optional[Hashable] = None,
    bands: Optional[Sequence[Hashable]] = None,
    higher_is_better: bool = False,
    priority: Optional[Sequence[float]] = None,
    resampling: Union[Resampling, str] = Resampling.nearest,
    source_index_name: str = "source_index",
    source_valid_name: str = "source_valid",
) -> Dataset:
    """
    Merge datasets geospatially.

    Uses :func:`rasterio.merge.merge`

    .. versionadded:: 0.2 crs

    Parameters
    ----------
    datasets: list[xarray.Dataset]
        List of multiple xarray.Dataset with all geo attributes.
        The first one is assumed to have the same
        CRS, dtype, dimensions, and data_vars as the others in the array.
    bounds: tuple, optional
        Bounds of the output image (left, bottom, right, top).
        If not set, bounds are determined from bounds of input Dataset.
    res: tuple, optional
        Output resolution in units of coordinate reference system.
        If not set, the resolution of the first Dataset is used.
        If a single value is passed, output pixels will be square.
    nodata: float, optional
        nodata value to use in output file.
        If not set, uses the nodata value in the first input Dataset.
    precision: float, optional
        Number of decimal points of precision when computing inverse transform.
    method: str or callable, optional
        See rasterio docs.
    crs: rasterio.crs.CRS, optional
        Output CRS. If not set, the CRS of the first DataArray is used.
    quality: str, optional
        Name of a quality layer. When given, the merge becomes quality
        priority: one source supplies *all* of ``bands`` at each pixel
        instead of each band being merged independently. See
        :func:`merge_datasets_quality` for the full semantics.
    bands: list[str], optional
        Bands merged together by quality priority. Defaults to every data
        variable except ``quality``. Requires ``quality``.
    higher_is_better: bool, optional
        If True the highest quality value wins; otherwise the lowest does.
        Requires ``quality``. Default is False.
    priority: list[float], optional
        Per-source priorities that break quality ties. Requires ``quality``.
    resampling: rasterio.enums.Resampling or str, optional
        Resampling for the data bands in quality priority mode; the quality
        layer itself always uses nearest neighbour. Default is nearest.
    source_index_name, source_valid_name: str, optional
        Names of the quality-priority output layers. Default to
        ``"source_index"`` and ``"source_valid"``.

    Returns
    -------
    :obj:`xarray.Dataset`:
        The geospatially merged data.
    """
    if quality is not None:
        if method is not None or precision is not None:
            raise QualityLayerError(
                "'method' and 'precision' do not apply to quality-priority "
                "merging; the winner is chosen by the quality layer."
            )
        return merge_datasets_quality(
            datasets,
            quality,
            bands=bands,
            bounds=bounds,
            res=res,
            crs=crs,
            nodata=nodata,
            higher_is_better=higher_is_better,
            priority=priority,
            resampling=resampling,
            source_index_name=source_index_name,
            source_valid_name=source_valid_name,
        )
    if any(value is not None for value in (bands, priority)) or higher_is_better:
        raise QualityLayerError(
            "'bands', 'priority' and 'higher_is_better' require 'quality'."
        )

    representative_ds = datasets[0]
    merged_data = {}
    for iii, data_var in enumerate(representative_ds.data_vars):
        merged_data[data_var] = merge_arrays(
            [dataset[data_var] for dataset in datasets],
            bounds=bounds,
            res=res,
            nodata=nodata,
            precision=precision,
            method=method,
            crs=crs,
            parse_coordinates=iii == 0,
        )
    data_var = list(representative_ds.data_vars)[0]
    xds = Dataset(
        merged_data,
        attrs=representative_ds.attrs,
    )
    xds.rio.write_crs(merged_data[data_var].rio.crs, inplace=True)
    return xds


def _as_resampling(resampling: Union[Resampling, str]) -> Resampling:
    """Return the rasterio resampling enum for a name or a value."""
    if isinstance(resampling, str):
        return _convert_str_to_resampling(resampling)
    return Resampling(resampling)


def _valid_mask(values: numpy.ndarray, nodata: Optional[float]) -> numpy.ndarray:
    """Boolean mask of the pixels in ``values`` that are valid observations.

    NaN is never a valid observation, and neither is the layer's own nodata.
    """
    values = numpy.asarray(values)
    if numpy.issubdtype(values.dtype, numpy.floating):
        valid = ~numpy.isnan(values)
    else:
        valid = numpy.ones(values.shape, dtype=bool)
    if nodata is None:
        return valid
    if numpy.isnan(nodata):
        return valid
    return valid & (values != nodata)


def _default_nodata(dtype: numpy.dtype) -> Optional[float]:
    """The rioxarray default nodata for a GDAL-compatible dtype, if any."""
    rasterio_dtype = dtype_rev.get(numpy.dtype(dtype).name)
    if rasterio_dtype is None:
        return None
    value = _NODATA_DTYPE_MAP.get(rasterio_dtype)
    return None if value is None else float(value)


def _resolve_dtype(arrays: Sequence[DataArray], fallback: str) -> numpy.dtype:
    """Output dtype for a layer: the common input dtype or ``fallback``."""
    dtypes = {numpy.dtype(array.dtype) for array in arrays}
    if len(dtypes) == 1:
        out_dtype = dtypes.pop()
    elif dtypes:
        out_dtype = numpy.dtype(numpy.result_type(*dtypes))
    else:
        out_dtype = numpy.dtype(fallback)
    # bool has no GDAL raster equivalent; store the flags as bytes.
    if out_dtype == numpy.dtype("bool"):
        out_dtype = numpy.dtype("uint8")
    if dtype_rev.get(out_dtype.name) is None:
        raise QualityLayerError(
            f"The dtype ({out_dtype}) cannot be represented in a raster. "
            "Choose one of the GDAL supported dtypes."
        )
    return out_dtype


def _shared_nodata(
    requested: Optional[float],
    layers: Sequence[tuple[numpy.dtype, Optional[float]]],
    valid_values: Sequence[numpy.ndarray],
    reserved: Optional[set] = None,
) -> float:
    """
    Pick one nodata value that every output layer can store.

    A GeoTIFF has a single nodata value and
    :meth:`~rioxarray.raster_dataset.RasterDataset.to_raster` requires every
    layer of a dataset to agree on it. Sharing one value keeps ``rio.clip``
    and ``rio.to_raster`` consistent with the merged result.

    ``valid_values`` are the observations the merged layers will hold and
    ``reserved`` are the values the source index/validity layers use; a
    nodata sentinel colliding with either would hide real data, so those
    candidates are skipped. The explicit ``requested`` value is honoured even
    then, since the caller asked for it.
    """
    dtypes = [numpy.dtype(dtype) for dtype, _ in layers]
    if requested is not None:
        return _check_nodata(float(requested), dtypes)

    observed = set() if reserved is None else set(reserved)
    for values in valid_values:
        values = numpy.asarray(values)
        if values.size:
            observed.update(numpy.unique(values).tolist())

    candidates = [float(nodata) for _, nodata in layers if nodata is not None]
    candidates += [
        value
        for value in (_default_nodata(dtype) for dtype in dtypes)
        if value is not None
    ]
    candidates += [-1.0, 0.0]

    seen = set()
    for candidate in candidates:
        key = (candidate, numpy.isnan(candidate))
        if key in seen:
            continue
        seen.add(key)
        if not _fits_nodata(candidate, dtypes):
            continue
        if candidate in observed and not numpy.isnan(candidate):
            continue
        return candidate
    raise QualityLayerError(
        "Unable to find a nodata value representable by every output dtype "
        f"({[str(dtype) for dtype in dtypes]}) and unused by the data. Pass "
        "an explicit 'nodata'."
    )


def _fits_nodata(nodata: float, dtypes: Sequence[numpy.dtype]) -> bool:
    """Whether ``nodata`` can be stored in every dtype without changing value."""
    for dtype in dtypes:
        dtype = numpy.dtype(dtype)
        if numpy.isnan(nodata):
            if not numpy.issubdtype(dtype, numpy.floating):
                return False
            continue
        try:
            _ensure_nodata_dtype(original_nodata=nodata, new_dtype=dtype)
        except (OverflowError, ValueError, TypeError):
            return False
    return True


def _check_nodata(nodata: float, dtypes: Sequence[numpy.dtype]) -> float:
    """Return an explicit nodata value after checking that it is storable."""
    if not _fits_nodata(nodata, dtypes):
        raise QualityLayerError(
            f"The nodata value ({nodata}) cannot be represented by every "
            f"output dtype ({[str(dtype) for dtype in dtypes]})."
        )
    return nodata


def _index_dtype(nodata: float, source_count: int) -> numpy.dtype:
    """Dtype for the source index and validity layers."""
    if numpy.isnan(nodata) or float(nodata) != int(nodata):
        return numpy.dtype("float64")
    needed = [0, 1, max(source_count - 1, 0), int(nodata)]
    for dtype in (
        numpy.dtype("uint8"),
        numpy.dtype("int8"),
        numpy.dtype("uint16"),
        numpy.dtype("int16"),
        numpy.dtype("uint32"),
        numpy.dtype("int32"),
        numpy.dtype("uint64"),
        numpy.dtype("int64"),
    ):
        info = numpy.iinfo(dtype)
        if all(info.min <= value <= info.max for value in needed):
            return dtype
    return numpy.dtype("float64")


def _rename_to_reference(dataarray: DataArray, reference: DataArray) -> DataArray:
    """Rename the inferred x/y dimensions back to the reference names."""
    rename = {}
    if (
        reference.rio.x_dim != "x"
        and dataarray.rio.x_dim == "x"
        and "x" in dataarray.dims
        and reference.rio.x_dim not in dataarray.dims
    ):
        rename["x"] = reference.rio.x_dim
    if (
        reference.rio.y_dim != "y"
        and dataarray.rio.y_dim == "y"
        and "y" in dataarray.dims
        and reference.rio.y_dim not in dataarray.dims
    ):
        rename["y"] = reference.rio.y_dim
    if rename:
        dataarray = dataarray.rename(rename)
        dataarray.rio.set_spatial_dims(
            x_dim=reference.rio.x_dim, y_dim=reference.rio.y_dim, inplace=True
        )
    return dataarray


def _align_layer(
    dataarray: DataArray,
    *,
    reference: DataArray,
    crs: CRS,
    transform: Affine,
    shape: tuple[int, int],
    x_values: numpy.ndarray,
    y_values: numpy.ndarray,
    out_dtype: numpy.dtype,
    resampling: Resampling,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Put a layer on the reference grid and return its values and valid mask.

    Validity comes from the layer's *own* nodata, evaluated before any output
    nodata normalization, so legitimate values are never mistaken for nodata.
    """
    if dataarray.rio.crs is None:
        raise MissingCRS("CRS not found. Please set the CRS with 'rio.write_crs()'.")
    on_reference_grid = (
        dataarray.rio.crs == crs
        and (int(dataarray.rio.height), int(dataarray.rio.width)) == shape
        and dataarray.rio.transform(recalc=True) == transform
        and numpy.array_equal(
            numpy.asarray(dataarray[dataarray.rio.x_dim].values), x_values
        )
        and numpy.array_equal(
            numpy.asarray(dataarray[dataarray.rio.y_dim].values), y_values
        )
    )
    if on_reference_grid:
        aligned = dataarray
    else:
        # When the source declares no nodata, reprojection still needs a value
        # for the uncovered destination pixels; treat that fill as invalid.
        source_nodata = dataarray.rio.nodata
        if source_nodata is None:
            source_nodata = _default_nodata(out_dtype)
        aligned = dataarray.rio.reproject(
            crs,
            transform=transform,
            shape=shape,
            resampling=resampling,
            nodata=source_nodata,
        )
        aligned = _rename_to_reference(aligned, reference)
        if dataarray.rio.nodata is None:
            aligned.rio.write_nodata(source_nodata, inplace=True)

    values = numpy.asarray(aligned.values)
    valid = _valid_mask(values, aligned.rio.nodata)
    return values.astype(out_dtype, copy=True), valid


def _grid_signature(dataarray: DataArray) -> tuple:
    """Everything that identifies the pixel grid of a 2D layer.

    The coordinates are included because the transform is stored per
    ``grid_mapping`` and can be shared by arrays with different coordinates.
    """
    x_values = numpy.asarray(dataarray[dataarray.rio.x_dim].values)
    y_values = numpy.asarray(dataarray[dataarray.rio.y_dim].values)
    return (
        None if dataarray.rio.crs is None else dataarray.rio.crs.to_wkt(),
        dataarray.rio.x_dim,
        dataarray.rio.y_dim,
        (int(dataarray.rio.height), int(dataarray.rio.width)),
        tuple(dataarray.rio.transform(recalc=True))[:6],
        (x_values.dtype.str, x_values.shape, x_values.tobytes()),
        (y_values.dtype.str, y_values.shape, y_values.tobytes()),
    )


def _prepare_layer(dataset: Dataset, name: Hashable) -> DataArray:
    """Return a 2D layer with its spatial dimensions resolved.

    Dims are taken from the variable itself, falling back to the dataset's
    ``rio`` accessor (as elsewhere in rioxarray) so ``lat``/``lon`` datasets
    work without renaming. A single-band 3D variable is squeezed to 2D.
    """
    dataarray = dataset[name]
    try:
        x_dim, y_dim = _get_spatial_dims(dataset, var=name)
    except MissingSpatialDimensionError:
        raise QualityLayerError(
            f"Unable to determine the spatial dimensions of {str(name)!r}. "
            "Set them with 'rio.set_spatial_dims()'."
        ) from None
    extra_dims = [
        dim
        for dim in dataarray.dims
        if dim not in (x_dim, y_dim) and dataarray.sizes[dim] != 1
    ]
    if extra_dims:
        raise QualityLayerError(
            f"{str(name)!r} has extra non-spatial dimensions {extra_dims}; only "
            "2D variables can be quality-merged."
        )
    if x_dim not in dataarray.dims or y_dim not in dataarray.dims:
        raise QualityLayerError(
            f"{str(name)!r} does not use the spatial dimensions "
            f"({x_dim!r}, {y_dim!r}) that the dataset's other layers use. "
            "Align the layers before merging."
        )
    dataarray = dataarray.squeeze(drop=True)
    if dataarray.dims != (y_dim, x_dim):
        dataarray = dataarray.transpose(y_dim, x_dim)
    return dataarray.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=False)


def _reference_grid(
    arrays: Sequence[DataArray],
    *,
    crs: Optional[CRS],
    res: Optional[Union[float, tuple[float, float]]],
    bounds: Optional[tuple],
) -> tuple[CRS, Affine, tuple[int, int]]:
    """Compute the common output grid every source is ranked on."""
    representative = arrays[0]
    if crs is None:
        crs = representative.rio.crs
    if crs is None:
        raise MissingCRS("CRS not found. Please set the CRS with 'rio.write_crs()'.")

    if res is None:
        resolution = tuple(abs(val) for val in representative.rio.resolution())
    elif isinstance(res, (int, float)):
        resolution = (abs(float(res)), abs(float(res)))
    else:
        resolutions = tuple(abs(float(val)) for val in res)
        resolution = (
            (resolutions[0], resolutions[0])
            if len(resolutions) == 1
            else (resolutions[0], resolutions[1])
        )
    if resolution[0] == 0 or resolution[1] == 0:
        raise QualityLayerError("The output resolution must be greater than 0.")

    if bounds is None:
        left = bottom = numpy.inf
        right = top = -numpy.inf
        for array in arrays:
            if array.rio.crs is None:
                raise MissingCRS(
                    "CRS not found. Please set the CRS with 'rio.write_crs()'."
                )
            a_left, a_bottom, a_right, a_top = array.rio.bounds(recalc=True)
            if array.rio.crs != crs:
                a_left, a_bottom, a_right, a_top = rasterio.warp.transform_bounds(
                    array.rio.crs, crs, a_left, a_bottom, a_right, a_top
                )
            left, bottom = min(left, a_left), min(bottom, a_bottom)
            right, top = max(right, a_right), max(top, a_top)
    else:
        left, bottom, right, top = (float(val) for val in bounds)

    width = int(round((right - left) / resolution[0]))
    height = int(round((top - bottom) / resolution[1]))
    if width < 1 or height < 1:
        raise QualityLayerError(
            "The requested bounds and resolution define an empty output grid."
        )
    # Never attempt to allocate an obviously impossible output grid; this
    # usually means incomparable CRSs or a resolution in the wrong units.
    max_pixels = 2**31
    if width * height > max_pixels:
        raise QualityLayerError(
            f"The output grid would be {width} x {height} pixels, which is too "
            "large to build in memory. Check the CRS and 'res' of the inputs."
        )
    transform = Affine.translation(left, top) @ Affine.scale(
        resolution[0], -resolution[1]
    )
    return crs, transform, (height, width)


def _grid_coords(
    *, source: DataArray, transform: Affine, shape: tuple[int, int]
) -> dict:
    """Coordinates of the output grid, keeping the reference dimension names."""
    coords = _make_coords(
        src_data_array=source,
        dst_affine=transform,
        dst_width=shape[1],
        dst_height=shape[0],
    )
    renamed = {}
    if source.rio.x_dim != "x" and "x" in coords:
        renamed[source.rio.x_dim] = IndexVariable(
            source.rio.x_dim, coords.pop("x").values
        )
    if source.rio.y_dim != "y" and "y" in coords:
        renamed[source.rio.y_dim] = IndexVariable(
            source.rio.y_dim, coords.pop("y").values
        )
    renamed.update(coords)
    return renamed


def merge_datasets_quality(
    datasets: Sequence[Dataset],
    quality: Hashable,
    *,
    bands: Optional[Sequence[Hashable]] = None,
    bounds: Optional[tuple] = None,
    res: Optional[Union[float, tuple[float, float]]] = None,
    crs: Optional[CRS] = None,
    nodata: Optional[float] = None,
    higher_is_better: bool = False,
    priority: Optional[Sequence[float]] = None,
    resampling: Union[Resampling, str] = Resampling.nearest,
    source_index_name: str = "source_index",
    source_valid_name: str = "source_valid",
) -> Dataset:
    """
    Merge datasets by a quality layer, one source per pixel for all bands.

    :func:`merge_datasets` merges every data variable independently, so a
    pixel may combine bands that were never observed together. This function
    lets a single source win each pixel and supply *every* band, so each
    output pixel remains one real observation.

    The quality layer and the data are first checked against each other and
    against the output grid. A source can only win a pixel where its quality
    value and all of its required bands are valid; a source with any required
    band invalid at that pixel is excluded. Sources are then ranked by the
    declared priority (the lowest quality wins unless ``higher_is_better``),
    and remaining ties are resolved by ``priority`` and finally by position in
    ``datasets``, which makes the choice deterministic. The quality layer is
    always resampled with nearest neighbour, so quality classes are never
    interpolated.

    The result contains the merged bands, the quality layer of the winning
    sources, ``source_index_name`` (the winning source's position in
    ``datasets``, or the nodata value) and ``source_valid_name`` (1 where a
    source won, nodata elsewhere). All layers share one nodata value, so
    ``rio.clip`` and ``rio.to_raster`` keep the source assignment intact.

    .. versionadded:: 0.25

    Parameters
    ----------
    datasets: list[xarray.Dataset]
        Datasets to merge. Each must contain the quality layer and every band
        listed in ``bands`` on the same grid.
    quality: str
        Name of the quality layer used to rank sources. Lower is better
        unless ``higher_is_better`` is True. Its nodata value marks a source
        invalid at that pixel.
    bands: list[str], optional
        Bands to merge, all taken from the same winning source per pixel.
        Defaults to every data variable except ``quality``.
    bounds: tuple, optional
        Output bounds (left, bottom, right, top) in ``crs``. Defaults to the
        union of the inputs.
    res: float or tuple, optional
        Output resolution in ``crs`` units. Defaults to the first input's.
    crs: rasterio.crs.CRS, optional
        Output CRS. Defaults to the first input's.
    nodata: float, optional
        nodata value used by every output layer. Defaults to the first source
        nodata value that all output dtypes can store, then to the dtype
        defaults.
    higher_is_better: bool, optional
        If True the highest quality value wins; otherwise the lowest does.
        Default is False.
    priority: list[float], optional
        Explicit priority per source, parallel to ``datasets``. Lower values
        win, but only to break quality ties before falling back to
        ``datasets`` order. Defaults to ``datasets`` order.
    resampling: rasterio.enums.Resampling or str, optional
        Resampling for the data bands. The quality layer always uses nearest
        neighbour. Default is ``Resampling.nearest``.
    source_index_name: str, optional
        Name of the output variable holding the winning source index.
        Default is ``"source_index"``.
    source_valid_name: str, optional
        Name of the output variable holding the per-pixel validity.
        Default is ``"source_valid"``.

    Returns
    -------
    :obj:`xarray.Dataset`:
        The merged bands, the quality layer and the source index/validity.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    if not datasets:
        raise QualityLayerError("At least one dataset is required to merge.")
    if str(quality) in (source_index_name, source_valid_name):
        raise QualityLayerError(
            f"The quality layer ({quality!r}) conflicts with a reserved output name."
        )
    if priority is None:
        source_priority = numpy.arange(len(datasets), dtype=numpy.float64)
    else:
        source_priority = numpy.asarray(priority, dtype=numpy.float64)
        if source_priority.shape != (len(datasets),):
            raise QualityLayerError(
                f"'priority' must have one value per dataset "
                f"({len(datasets)}); got {source_priority.size}."
            )
        source_priority = numpy.where(
            numpy.isnan(source_priority), numpy.inf, source_priority
        )

    band_names = (
        [str(name) for name in bands]
        if bands is not None
        else [
            str(var)
            for var in datasets[0].data_vars
            if str(var) != str(quality)
            and str(var) not in (source_index_name, source_valid_name)
        ]
    )
    if not band_names:
        raise QualityLayerError(
            "No bands to merge. Provide 'bands' or add variables besides "
            f"the quality layer ({quality!r})."
        )
    if len(set(band_names)) != len(band_names):
        raise QualityLayerError(f"Duplicate band names requested: {band_names}")
    for reserved in (source_index_name, source_valid_name):
        if reserved in band_names:
            raise QualityLayerError(f"{reserved!r} is reserved for the merge output.")
    if str(quality) in band_names:
        raise QualityLayerError(
            f"The quality layer ({quality!r}) cannot also be merged as a band."
        )

    for dataset in datasets:
        missing = [name for name in [quality, *band_names] if name not in dataset]
        if missing:
            raise QualityLayerError(
                f"Dataset is missing required variable(s) {missing}."
            )

    # Resolve a single spatial grid per source, so every band and the quality
    # layer are compared on the same (x, y) axes.
    sources = [
        {name: _prepare_layer(dataset, name) for name in [quality, *band_names]}
        for dataset in datasets
    ]

    # The quality layer and the bands must describe the same grid, otherwise
    # a source could be ranked by a quality value from another pixel. Within
    # a Dataset the transform is shared, so compare the coordinate values too.
    for source in sources:
        reference = source[quality]
        reference_grid = _grid_signature(reference)
        for name in band_names:
            if _grid_signature(source[name]) != reference_grid:
                raise QualityLayerError(
                    f"The quality layer {str(quality)!r} and band {str(name)!r} "
                    "are not on the same grid. Align them before merging."
                )

    resampling_enum = _as_resampling(resampling)
    dst_crs, dst_transform, dst_shape = _reference_grid(
        [source[quality] for source in sources],
        crs=crs,
        res=res,
        bounds=bounds,
    )
    reference_quality = sources[0][quality]
    out_dims = (reference_quality.rio.y_dim, reference_quality.rio.x_dim)
    dst_coords = _grid_coords(
        source=reference_quality, transform=dst_transform, shape=dst_shape
    )
    if (
        len(dst_coords) < 2
        or out_dims[0] not in dst_coords
        or (out_dims[1] not in dst_coords)
    ):
        raise QualityLayerError(
            "Unable to generate the output coordinates for the merged grid."
        )

    band_dtypes = {
        name: _resolve_dtype([source[name] for source in sources], "float32")
        for name in band_names
    }
    quality_dtype = _resolve_dtype([source[quality] for source in sources], "float32")
    dst_x = numpy.asarray(dst_coords[reference_quality.rio.x_dim].values)
    dst_y = numpy.asarray(dst_coords[reference_quality.rio.y_dim].values)

    aligned_bands: dict[str, list[numpy.ndarray]] = {name: [] for name in band_names}
    band_valid: dict[str, list[numpy.ndarray]] = {name: [] for name in band_names}
    aligned_quality = []
    quality_valid = []
    for source in sources:
        for name in band_names:
            values, valid = _align_layer(
                source[name],
                reference=reference_quality,
                crs=dst_crs,
                transform=dst_transform,
                shape=dst_shape,
                x_values=dst_x,
                y_values=dst_y,
                out_dtype=band_dtypes[name],
                resampling=resampling_enum,
            )
            aligned_bands[name].append(values)
            band_valid[name].append(valid)
        # quality classes are never interpolated
        values, valid = _align_layer(
            source[quality],
            reference=reference_quality,
            crs=dst_crs,
            transform=dst_transform,
            shape=dst_shape,
            x_values=dst_x,
            y_values=dst_y,
            out_dtype=quality_dtype,
            resampling=Resampling.nearest,
        )
        aligned_quality.append(values)
        quality_valid.append(valid)

    reserved = set(range(len(sources))) | {0, 1}
    shared_nodata = _shared_nodata(
        nodata,
        [(band_dtypes[name], sources[0][name].rio.nodata) for name in band_names]
        + [(quality_dtype, reference_quality.rio.nodata)],
        [
            *(
                values[valid]
                for name in band_names
                for values, valid in zip(aligned_bands[name], band_valid[name])
            ),
            *(values[valid] for values, valid in zip(aligned_quality, quality_valid)),
        ],
        reserved=reserved,
    )
    if (
        nodata is not None
        and not numpy.isnan(shared_nodata)
        and (shared_nodata in reserved)
    ):
        raise QualityLayerError(
            f"The nodata value ({shared_nodata}) collides with a source index "
            "or a validity flag; pass another value."
        )
    index_dtype = _index_dtype(shared_nodata, len(sources))
    for one_dtype in (*band_dtypes.values(), quality_dtype, index_dtype):
        if not _fits_nodata(shared_nodata, [one_dtype]):
            raise QualityLayerError(
                f"The nodata value ({shared_nodata}) cannot be represented by "
                f"dtype ({one_dtype})."
            )

    valid_stack = numpy.stack(
        [
            numpy.logical_and.reduce(
                [
                    *[band_valid[name][index] for name in band_names],
                    quality_valid[index],
                ]
            )
            for index in range(len(sources))
        ]
    )
    quality_stack = numpy.stack(
        [numpy.asarray(values, dtype=numpy.float64) for values in aligned_quality]
    )

    # Rank by quality, then declared priority, then source order. The first
    # valid source wins outright, so the choice never depends on how a dtype
    # compares extreme quality values, and the strict comparisons keep the
    # earliest source on a full tie.
    has_winner = numpy.zeros(dst_shape, dtype=bool)
    best_index = numpy.zeros(dst_shape, dtype=numpy.intp)
    best_quality = numpy.zeros(dst_shape, dtype=numpy.float64)
    best_priority = numpy.full(dst_shape, numpy.inf, dtype=numpy.float64)
    for index in range(len(sources)):
        valid = valid_stack[index]
        source_quality_values = quality_stack[index]
        if higher_is_better:
            quality_better = source_quality_values > best_quality
        else:
            quality_better = source_quality_values < best_quality
        quality_tie = source_quality_values == best_quality
        better = valid & (
            ~has_winner
            | quality_better
            | (quality_tie & (source_priority[index] < best_priority))
        )
        best_quality = numpy.where(better, source_quality_values, best_quality)
        best_priority = numpy.where(better, source_priority[index], best_priority)
        best_index = numpy.where(better, index, best_index)
        has_winner |= better

    winner = numpy.where(has_winner, best_index, -1)
    rows, cols = numpy.indices(dst_shape)

    result = Dataset(attrs=dict(datasets[0].attrs))
    for name in band_names:
        stack = numpy.stack(aligned_bands[name]).astype(band_dtypes[name])
        values = numpy.full(dst_shape, shared_nodata, dtype=band_dtypes[name])
        if has_winner.any():
            values[has_winner] = stack[
                winner[has_winner], rows[has_winner], cols[has_winner]
            ]
        layer = DataArray(
            values,
            dims=out_dims,
            coords=dst_coords,
            attrs=dict(sources[0][name].attrs),
            name=name,
        )
        layer.rio.write_nodata(shared_nodata, inplace=True)
        result[name] = layer

    quality_values = numpy.full(dst_shape, shared_nodata, dtype=quality_dtype)
    if has_winner.any():
        cast_quality = numpy.stack(aligned_quality).astype(quality_dtype)
        quality_values[has_winner] = cast_quality[
            winner[has_winner], rows[has_winner], cols[has_winner]
        ]
    merged_quality = DataArray(
        quality_values,
        dims=out_dims,
        coords=dst_coords,
        attrs=dict(reference_quality.attrs),
        name=quality,
    )
    merged_quality.rio.write_nodata(shared_nodata, inplace=True)
    result[str(quality)] = merged_quality

    source_index_values = numpy.full(dst_shape, shared_nodata, dtype=index_dtype)
    if has_winner.any():
        source_index_values[has_winner] = winner[has_winner].astype(index_dtype)
    source_index = DataArray(
        source_index_values,
        dims=out_dims,
        coords=dst_coords,
        attrs={"long_name": "index of the source that won each pixel"},
        name=source_index_name,
    )
    source_index.rio.write_nodata(shared_nodata, inplace=True)
    result[source_index_name] = source_index

    valid_values = numpy.full(dst_shape, shared_nodata, dtype=index_dtype)
    valid_values[has_winner] = numpy.asarray(1, dtype=index_dtype)
    source_valid = DataArray(
        valid_values,
        dims=out_dims,
        coords=dst_coords,
        attrs={"long_name": "1 where a source was valid, nodata elsewhere"},
        name=source_valid_name,
    )
    source_valid.rio.write_nodata(shared_nodata, inplace=True)
    result[source_valid_name] = source_valid

    for name in result.data_vars:
        try:
            result[name].encoding = dict(sources[0][name].encoding)
        except KeyError:
            pass
    # writing the CRS at the Dataset level also tags every data variable
    result.rio.write_crs(dst_crs, inplace=True)
    result.rio.write_transform(dst_transform, inplace=True)
    return result
