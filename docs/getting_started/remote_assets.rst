.. _remote_assets:

Remote assets with re-resolvable identities
============================================

Lazy Dask computations can be executed long after the data was opened.  When
the underlying raster lives behind a short-lived signed URL, the URL embedded
in the Dask graph may have expired by the time a worker task tries to read
pixels.

``rioxarray`` solves this by separating the *stable identity* of an asset
from its *current URL*.  Instead of passing a URL to
:func:`rioxarray.open_rasterio`, pass an
:class:`~rioxarray.assets.AssetIdentity` together with a
:class:`~rioxarray.assets.RemoteAssetResolver`.  The resolver is invoked at
actual read time to obtain the current URL and open options, so expired
credentials are never baked into a lazy graph.

Defining a resolver
-------------------

A resolver turns an identity into a :class:`~rioxarray.assets.ResolvedAsset`
(current URL, open options and content version):

.. code-block:: python

    import rioxarray
    from rioxarray.assets import (
        AssetIdentity,
        RemoteAssetResolver,
        ResolvedAsset,
    )


    class StacResolver(RemoteAssetResolver):
        """Mint a fresh signed URL for a STAC asset on every resolution."""

        def __init__(self, catalog_client):
            self.catalog = catalog_client

        def resolve(self, identity, *, force_refresh=False):
            record = self.catalog.get_asset(
                identity.asset_id,
                version=identity.version,
                force_refresh=force_refresh,
            )
            return ResolvedAsset(
                url=record.signed_url,
                open_options={"headers": record.headers},
                version=record.content_version,
            )


    identity = AssetIdentity("s3://bucket/collection/item/asset.tif")
    da = rioxarray.open_rasterio(
        identity,
        asset_resolver=StacResolver(catalog),
        chunks=(1, 2048, 2048),
    )

    # The resolver is only called here, when the pixels are actually read.
    da.values

For one-off resolvers, :class:`~rioxarray.assets.SimpleResolver` wraps a plain
callable.

Authentication refresh
----------------------

When a read fails with an authentication/authorization error (HTTP 401/403,
expired signatures or tokens, ...), the manager closes the broken handle,
asks the resolver to mint a fresh URL with ``force_refresh=True`` and
re-opens the asset.  The number of refreshes is bounded by
``max_refreshes`` (default 2):

.. code-block:: python

    da = rioxarray.open_rasterio(
        identity,
        asset_resolver=resolver,
        max_refreshes=3,
    )

Refreshes are *single-flight*: if many threads see the same URL expire at
once, only one resolver call is made and every thread re-opens against the
resulting generation.  Resolver-specific exception types can be recognized by
overriding :meth:`~rioxarray.assets.RemoteAssetResolver.is_auth_error`.

Version and schema pinning
--------------------------

The content version and the full grid/band schema (dimensions, transform,
CRS, band count, dtypes, nodata, scales, offsets, descriptions, units)
observed at the first open are pinned for the lifetime of the array.  Every
re-open is checked against them:

* if the refreshed asset reports a different content version,
  :class:`~rioxarray.assets.AssetVersionMismatchError` is raised;
* if the grid or band schema changed,
  :class:`~rioxarray.assets.AssetSchemaMismatchError` is raised.

This guarantees that pixels from a new asset version can never be silently
mixed into the same lazy computation.  Start a new computation by calling
:func:`~rioxarray.open_rasterio` again.  Passing an explicit ``version`` to
:class:`~rioxarray.assets.AssetIdentity` pins the computation from the very
first resolution.

Serialization and resource lifecycle
------------------------------------

Only the re-resolvable identity (never a signed URL or a live rasterio
handle) is stored on the Dask graph, so pickled arrays recompute correctly on
any worker that can construct the resolver.  Metadata and optional
read-window caches (``window_cache_size``) are partitioned by content
version, and closing the array (``da.close()``) releases every handle the
manager opened.

Local paths are unaffected
--------------------------

Calling :func:`~rioxarray.open_rasterio` with a regular path or an open
rasterio dataset follows exactly the same code path as before; the asset
machinery is only used when the argument is an
:class:`~rioxarray.assets.AssetIdentity`.
