rioxarray package
=================

rioxarray.open_rasterio
-----------------------

.. autofunction:: rioxarray.open_rasterio


rioxarray.assets module
-----------------------

Remote assets (for example objects behind short-lived signed URLs) can be
opened through a stable asset identity instead of a URL.  The identity is
re-resolved to a current URL and open options every time a lazy task actually
reads the asset, expired credentials are refreshed a bounded number of times,
and the content version, grid and band schema are pinned so a new asset
version can never be silently mixed into an ongoing computation.

.. automodule:: rioxarray.assets
    :members:
    :undoc-members:
    :show-inheritance:


rioxarray.merge module
----------------------

.. autofunction:: rioxarray.merge.merge_arrays

.. autofunction:: rioxarray.merge.merge_datasets


rioxarray.set_options
-----------------------

.. autoclass:: rioxarray.set_options


rioxarray.show_versions
-----------------------

.. autofunction:: rioxarray.show_versions


rioxarray.enum module
---------------------

.. automodule:: rioxarray.enum
    :members:
    :undoc-members:
    :show-inheritance:


rioxarray `rio` accessors
--------------------------

rioxarray `extends xarray <http://xarray.pydata.org/en/stable/internals/extending-xarray.html>`__
with the `rio` accessor. The `rio` accessor is activated by importing rioxarray like so:

.. code-block:: python

    import rioxarray


.. autoclass:: rioxarray.rioxarray.XRasterBase
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: rioxarray.raster_array.RasterArray
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: rioxarray.raster_dataset.RasterDataset
    :members:
    :undoc-members:
    :show-inheritance:


rioxarray.exceptions module
---------------------------

.. automodule:: rioxarray.exceptions
    :members:
    :undoc-members:
    :show-inheritance:
