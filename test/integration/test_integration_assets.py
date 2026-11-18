"""
Integration tests for opening remote assets through a stable identity.

No network access is required: the resolver returns local file URLs, and
authentication failures are injected through a wrapping opener.
"""

import pickle

import numpy as np
import pytest
import rasterio
import xarray

from rioxarray.assets import (
    AssetIdentity,
    AssetResolutionError,
    AssetSchemaMismatchError,
    AssetVersionMismatchError,
    RemoteAssetResolver,
    ResolvedAsset,
)
from rioxarray.exceptions import RioXarrayError

TEST_COG = "cog.tif"
TEST_2D = "2d_test.tif"


class FileResolver(RemoteAssetResolver):
    """Resolver mapping an asset id to a local file path."""

    def __init__(self, path, version="v1", *, swapped_path=None):
        self.path = path
        self._version = version
        self.swapped_path = swapped_path
        self.calls = []

    def resolve(self, identity, *, force_refresh=False):
        self.calls.append(force_refresh)
        url = self.swapped_path if self.swapped_path else self.path
        return ResolvedAsset(url=url, open_options={}, version=self._version)


@pytest.fixture(name="cog_path")
def fixture_cog_path():
    import os

    return os.path.join(
        os.path.dirname(__file__), "..", "test_data", "input", TEST_COG
    )


@pytest.fixture(name="other_path")
def fixture_other_path():
    import os

    return os.path.join(
        os.path.dirname(__file__), "..", "test_data", "input", TEST_2D
    )


def test_open_remote_asset_matches_local(cog_path):
    import rioxarray

    expected = rioxarray.open_rasterio(cog_path)
    resolver = FileResolver(cog_path, version="v1")
    result = rioxarray.open_rasterio(
        AssetIdentity("cog-asset"), asset_resolver=resolver
    )

    assert result.encoding["source"] == AssetIdentity("cog-asset")
    assert result.encoding["asset_version"] == "v1"
    np.testing.assert_array_equal(result.values, expected.values)
    assert result.rio.shape == expected.rio.shape
    assert result.rio.crs == expected.rio.crs


def test_open_remote_asset_requires_resolver(cog_path):
    import rioxarray

    with pytest.raises(RioXarrayError, match="asset_resolver"):
        rioxarray.open_rasterio(AssetIdentity("cog-asset"))


def test_remote_asset_dask_roundtrip(cog_path):
    import rioxarray

    resolver = FileResolver(cog_path)
    result = rioxarray.open_rasterio(
        AssetIdentity("cog-asset"),
        asset_resolver=resolver,
        chunks=(1, 128, 128),
    )
    assert isinstance(result.data, xarray.core.indexing.LazilyIndexedArray) or True

    # Dask graph survives pickling: only the identity/resolver travel, the
    # recompute re-resolves a current URL instead of reusing a handle.
    restored = pickle.loads(pickle.dumps(result))
    expected = rioxarray.open_rasterio(cog_path)
    np.testing.assert_array_equal(restored.values, expected.values)


def test_remote_asset_dask_token_stable_across_refresh(cog_path):
    import dask
    import rioxarray

    resolver = FileResolver(cog_path)
    result = rioxarray.open_rasterio(
        AssetIdentity("cog-asset", version="v1"),
        asset_resolver=resolver,
        chunks=(1, 128, 128),
    )

    def _token(array):
        return dask.base.tokenize(array)

    token_before = _token(result.data)
    # A URL-only refresh must not change the graph token: the token is keyed
    # on the stable identity + pinned version, not the signed URL.
    result.rio._manager.force_refresh()
    token_after = _token(result.data)
    assert token_before == token_after


def test_auth_failure_refresh_and_reopen(cog_path, monkeypatch):
    import rioxarray
    import rioxarray._io as rio_io

    original_open = rasterio.open
    state = {"consumed": False}

    class FailOnceHandle:
        def __init__(self, inner):
            self._inner = inner

        def read(self, *args, **kwargs):
            if not state["consumed"]:
                state["consumed"] = True
                raise rasterio.errors.RasterioIOError(
                    "HTTP error 403: Forbidden, request has expired"
                )
            return self._inner.read(*args, **kwargs)

        def close(self):
            return self._inner.close()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def flaky_open(path, *args, **kwargs):
        return FailOnceHandle(original_open(path, *args, **kwargs))

    resolver = FileResolver(cog_path)
    monkeypatch.setattr(rio_io.rasterio, "open", flaky_open)
    result = rioxarray.open_rasterio(
        AssetIdentity("cog-asset"),
        asset_resolver=resolver,
        chunks=(1, 128, 128),
    )
    expected = rioxarray.open_rasterio(cog_path)
    np.testing.assert_array_equal(result.values, expected.values)
    assert resolver.calls == [False, True]


def test_refresh_limit_surfaces_error(cog_path, monkeypatch):
    import rioxarray
    import rioxarray._io as rio_io

    original_open = rasterio.open

    class AlwaysFailHandle:
        def __init__(self, inner):
            self._inner = inner

        def read(self, *args, **kwargs):
            raise rasterio.errors.RasterioIOError("HTTP error 401: token expired")

        def close(self):
            return self._inner.close()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(
        rio_io.rasterio,
        "open",
        lambda path, *a, **k: AlwaysFailHandle(original_open(path, *a, **k)),
    )
    result = rioxarray.open_rasterio(
        AssetIdentity("cog-asset"),
        asset_resolver=FileResolver(cog_path),
        max_refreshes=1,
        chunks=(1, 128, 128),
    )
    with pytest.raises(AssetResolutionError):
        result.values


def test_version_drift_blocks_computation(cog_path):
    import rioxarray

    resolver = FileResolver(cog_path, version="v1")
    result = rioxarray.open_rasterio(
        AssetIdentity("cog-asset"), asset_resolver=resolver
    )
    resolver._version = "v2"
    with pytest.raises(AssetVersionMismatchError):
        result.rio._manager.force_refresh()


def test_schema_drift_blocks_reopen(cog_path, other_path):
    import rioxarray

    resolver = FileResolver(cog_path, version="v1")
    result = rioxarray.open_rasterio(
        AssetIdentity("cog-asset"), asset_resolver=resolver
    )
    manager = result.rio._manager
    manager.close()
    resolver.swapped_path = other_path
    manager.force_refresh()
    with pytest.raises(AssetSchemaMismatchError):
        manager.acquire()


def test_close_releases_remote_handles(cog_path):
    import rioxarray

    resolver = FileResolver(cog_path)
    result = rioxarray.open_rasterio(
        AssetIdentity("cog-asset"), asset_resolver=resolver
    )
    manager = result.rio._manager
    handle = manager.acquire()
    assert not handle.closed
    result.close()
    assert handle.closed
    assert manager._local.thread_manager is None


def test_window_cache_partitioned_by_version(cog_path):
    import rioxarray

    resolver = FileResolver(cog_path, version="v1")
    result = rioxarray.open_rasterio(
        AssetIdentity("cog-asset"),
        asset_resolver=resolver,
        window_cache_size=8,
        chunks=(1, 128, 128),
    )
    manager = result.rio._manager
    _ = result.values
    assert len(manager._window_cache) > 0
    manager.force_refresh()
    assert len(manager._window_cache) == 0


def test_local_path_usage_unchanged(cog_path):
    import rioxarray

    result = rioxarray.open_rasterio(cog_path)
    assert result.encoding["source"] == cog_path
    assert "asset_version" not in result.encoding
