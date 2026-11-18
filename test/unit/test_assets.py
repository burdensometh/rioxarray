"""
Unit tests for the remote asset resolution machinery in ``rioxarray.assets``.

The resolvers and openers are fakes; no network access is required.
"""

import pickle
import threading
import concurrent.futures

import pytest
import rasterio

from rioxarray.assets import (
    AssetIdentity,
    AssetResolutionError,
    AssetSchema,
    AssetSchemaMismatchError,
    AssetURIManager,
    AssetVersionMismatchError,
    RemoteAssetResolver,
    ResolvedAsset,
    SimpleResolver,
    get_registered_resolver,
    is_credential_error,
    is_remote_asset,
    register_resolver,
)


class FakeHandle:
    """Minimal rasterio-like handle driven by an opener script."""

    def __init__(self, url, version, read_error=None):
        self.url = url
        self.version = version
        self._read_error = read_error
        self.closed = False
        self.reads = []
        self.name = url
        self.count = 1
        self.height = 4
        self.width = 4
        self.dtypes = ("int16",)
        self.nodatavals = (None,)
        self.indexes = (1,)
        self.scales = (1.0,)
        self.offsets = (0.0,)
        self.descriptions = (None,)
        self.units = (None,)
        self.crs = None
        self.transform = type("T", (), {"a": 1.0, "b": 0.0, "c": 0.0,
                                        "d": 0.0, "e": 1.0, "f": 0.0})()
        self.block_shapes = [(4, 4)]
        self.tags = lambda **_: {}

    def read(self, band_key, window=None, masked=False, **kwargs):
        self.reads.append((band_key, window, masked))
        if self._read_error is not None:
            raise self._read_error
        import numpy as np

        return np.zeros((1, 4, 4), dtype="int16")

    def close(self):
        self.closed = True


class ScriptedResolver(RemoteAssetResolver):
    """Resolver returning scripted resolutions and recording calls."""

    def __init__(self, resolutions, *, auth_exc=None):
        self._resolutions = list(resolutions)
        self._cursor = -1
        self.calls = []  # list of force_refresh flags
        self.auth_exc = auth_exc

    def resolve(self, identity, *, force_refresh=False):
        self.calls.append(force_refresh)
        if force_refresh or self._cursor < 0:
            self._cursor = min(self._cursor + 1, len(self._resolutions) - 1)
        return self._resolutions[self._cursor]


def _module_level_opener(url, mode="r", **kwargs):
    return FakeHandle(url, "v1")


def _make_manager(resolver, identity=None, *, opener=None, **kwargs):
    identity = identity or AssetIdentity("asset-1")
    opener = opener or _module_level_opener
    return AssetURIManager(identity, resolver, opener, **kwargs)


# ---------------------------------------------------------------------------
# Identity / value objects
# ---------------------------------------------------------------------------


def test_asset_identity_requires_id():
    with pytest.raises(ValueError):
        AssetIdentity("")


def test_asset_identity_extra_normalised_to_tuple():
    identity = AssetIdentity.from_mapping("a", version="v1", extra={"z": 1, "a": 2})
    assert identity.extra == (("a", 2), ("z", 1))
    assert identity.extra_mapping() == {"a": 2, "z": 1}
    # frozen and hashable
    assert hash(identity) == hash(pickle.loads(pickle.dumps(identity)))


def test_asset_identity_pinned():
    identity = AssetIdentity("a")
    pinned = identity.pinned("v7")
    assert identity.version is None
    assert pinned.version == "v7"


def test_resolved_asset_defaults():
    resolved = ResolvedAsset(url="https://example.test/a.tif")
    assert resolved.open_options == {}
    assert resolved.version is None


# ---------------------------------------------------------------------------
# Credential error heuristic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "HTTP error 403: Forbidden",
        "HTTP error 401: Unauthorized",
        "SignatureDoesNotMatch",
        "The request has expired",
        "provided token is expired",
        "access denied",
    ],
)
def test_is_credential_error_messages(message):
    assert is_credential_error(rasterio.errors.RasterioIOError(message))


def test_is_credential_error_class_name():
    class ExpiredTokenError(Exception):
        pass

    assert is_credential_error(ExpiredTokenError("boom"))


def test_is_credential_error_negative():
    assert not is_credential_error(IOError("no such file"))


def test_resolver_custom_auth_predicate():
    class Marker(Exception):
        pass

    resolver = SimpleResolver(
        lambda identity, force_refresh=False: ResolvedAsset("u"),
        auth_error_callable=lambda exc: isinstance(exc, Marker),
    )
    assert resolver.is_auth_error(Marker())
    assert not resolver.is_auth_error(RuntimeError("x"))


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------


def test_acquire_resolves_once_and_caches_handle():
    resolver = ScriptedResolver([ResolvedAsset("url-1", version="v1")])
    manager = _make_manager(resolver)
    handle = manager.acquire()
    assert manager.acquire() is handle
    assert resolver.calls == [False]
    assert manager.pinned_version == "v1"
    assert manager.schema is not None
    manager.close()
    assert handle.closed


def test_close_all_reclaims_every_handle():
    resolver = ScriptedResolver([ResolvedAsset("url-1", version="v1")])
    manager = _make_manager(resolver)
    handles = []

    def grab():
        handles.append(manager.acquire())

    threads = [threading.Thread(target=grab) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(handles) == 4
    manager.close_all()
    assert all(handle.closed for handle in handles)


def test_pickle_carries_identity_not_handles():
    resolver = ScriptedResolver([ResolvedAsset("url-1", version="v1")])
    manager = _make_manager(resolver)
    handle = manager.acquire()
    restored = pickle.loads(pickle.dumps(manager))
    assert restored.identity == manager.identity
    assert restored._resolved is None
    assert restored.schema is None
    new_handle = restored.acquire()
    assert new_handle is not handle
    assert new_handle.url == "url-1"


def test_is_remote_asset():
    assert is_remote_asset(AssetIdentity("a"))
    assert not is_remote_asset("/local/path.tif")


# ---------------------------------------------------------------------------
# Refresh behavior
# ---------------------------------------------------------------------------


def _auth_error():
    return rasterio.errors.RasterioIOError("HTTP error 403: expired signature")


def test_read_recovers_after_single_refresh():
    handles = []

    def opener(url, mode="r", **kwargs):
        handle = FakeHandle(
            url,
            "v1",
            read_error=_auth_error() if len(handles) == 0 else None,
        )
        handles.append(handle)
        return handle

    resolver = ScriptedResolver(
        [ResolvedAsset("url-1", version="v1"), ResolvedAsset("url-2", version="v1")]
    )
    manager = _make_manager(resolver, opener=opener, max_refreshes=2)
    result = manager.read_window(1, ((0, 4), (0, 4)), masked=False)
    assert result.shape == (1, 4, 4)
    assert resolver.calls == [False, True]
    assert handles[0].closed
    assert len(handles) == 2


def test_refresh_budget_is_bounded():
    def opener(url, mode="r", **kwargs):
        return FakeHandle(url, "v1", read_error=_auth_error())

    resolver = ScriptedResolver(
        [ResolvedAsset(f"url-{i}", version="v1") for i in range(6)]
    )
    manager = _make_manager(resolver, opener=opener, max_refreshes=2)
    with pytest.raises(AssetResolutionError):
        manager.read_window(1, ((0, 4), (0, 4)), masked=False)
    # one initial resolution + two forced refreshes
    assert resolver.calls == [False, True, True]


def test_non_auth_error_is_not_retried():
    def opener(url, mode="r", **kwargs):
        return FakeHandle(url, "v1", read_error=OSError("disk on fire"))

    resolver = ScriptedResolver([ResolvedAsset("url-1", version="v1")])
    manager = _make_manager(resolver, opener=opener)
    with pytest.raises(OSError, match="disk on fire"):
        manager.read_window(1, ((0, 4), (0, 4)), masked=False)
    assert resolver.calls == [False]


def test_concurrent_auth_failures_share_single_refresh():
    import time

    state = {"generation": 0}

    class GenerationalHandle(FakeHandle):
        def __init__(self, url, opened_generation):
            super().__init__(url, "v1")
            self._opened_generation = opened_generation

        def read(self, band_key, window=None, masked=False, **kwargs):
            if state["generation"] == self._opened_generation == 0:
                raise _auth_error()
            return super().read(band_key, window=window, masked=masked, **kwargs)

    def opener(url, mode="r", **kwargs):
        return GenerationalHandle(url, state["generation"])

    class SlowRefreshResolver(ScriptedResolver):
        def resolve(self, identity, *, force_refresh=False):
            if force_refresh:
                # Widen the race window so all threads pile up on the lock.
                time.sleep(0.2)
                state["generation"] += 1
            return super().resolve(identity, force_refresh=force_refresh)

    resolver = SlowRefreshResolver(
        [ResolvedAsset("url-1", version="v1"), ResolvedAsset("url-2", version="v1")]
    )
    manager = _make_manager(resolver, opener=opener, max_refreshes=3)

    def read():
        return manager.read_window(1, ((0, 4), (0, 4)), masked=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: read(), range(8)))

    assert all(result.shape == (1, 4, 4) for result in results)
    assert resolver.calls.count(True) == 1


# ---------------------------------------------------------------------------
# Version / schema consistency
# ---------------------------------------------------------------------------


def test_version_drift_on_refresh_rejected():
    resolver = ScriptedResolver(
        [ResolvedAsset("url-1", version="v1"), ResolvedAsset("url-2", version="v2")]
    )
    manager = _make_manager(resolver)
    manager.acquire()
    with pytest.raises(AssetVersionMismatchError):
        manager.force_refresh()
    # manager stays pinned to the original version
    assert manager.pinned_version == "v1"


def test_pre_pinned_version_enforced_on_first_resolve():
    resolver = ScriptedResolver([ResolvedAsset("url-1", version="v9")])
    manager = _make_manager(resolver, identity=AssetIdentity("a", version="v1"))
    with pytest.raises(AssetVersionMismatchError):
        manager.acquire()


def test_schema_mismatch_on_reopen_rejected():
    schema_a = AssetSchema(
        count=1, height=4, width=4, transform=(1, 0, 0, 0, 1, 0), crs=None,
        dtypes=("int16",), nodatavals=(None,), indexes=(1,), scales=(1.0,),
        offsets=(0.0,), descriptions=(None,), units=(None,),
    )
    schema_b = AssetSchema(
        count=2, height=4, width=4, transform=(1, 0, 0, 0, 1, 0), crs=None,
        dtypes=("int16",), nodatavals=(None,), indexes=(1, 2), scales=(1.0, 1.0),
        offsets=(0.0, 0.0), descriptions=(None, None), units=(None, None),
    )
    diffs = schema_a.diff(schema_b)
    assert "count" in diffs[0]
    with pytest.raises(AssetSchemaMismatchError):
        schema_a.assert_matches(schema_b, asset_id="a")


def test_schema_drift_detected_on_acquire():
    handles = []

    def opener(url, mode="r", **kwargs):
        handle = FakeHandle(url, "v1")
        if handles:
            handle.width = 8  # grid change on the second open
        handles.append(handle)
        return handle

    resolver = ScriptedResolver(
        [ResolvedAsset("url-1", version="v1"), ResolvedAsset("url-2", version="v1")]
    )
    manager = _make_manager(resolver, opener=opener)
    manager.acquire()
    manager.close()
    manager.force_refresh()
    with pytest.raises(AssetSchemaMismatchError, match="width"):
        manager.acquire()


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------


def test_window_cache_namespaced_by_version():
    resolver = ScriptedResolver([ResolvedAsset("url-1", version="v1")])
    manager = _make_manager(resolver, window_cache_size=4)
    window = ((0, 4), (0, 4))
    first = manager.read_window(1, window, masked=False)
    second = manager.read_window(1, window, masked=False)
    assert first is second
    # different version namespace never hits
    assert manager._window_cache.get("v2", ((1,), window, False, ())) is None


def test_window_cache_hit_avoids_repeated_reads():
    resolver = ScriptedResolver([ResolvedAsset("url-1", version="v1")])
    manager = _make_manager(resolver, window_cache_size=4)
    window = ((0, 4), (0, 4))
    manager.read_window(1, window, masked=False)
    handle = manager.acquire()
    reads_after_first = len(handle.reads)
    manager.read_window(1, window, masked=False)
    manager.read_window(1, window, masked=False)
    assert len(handle.reads) == reads_after_first


def test_window_cache_disabled_by_default():
    resolver = ScriptedResolver([ResolvedAsset("url-1", version="v1")])
    manager = _make_manager(resolver)
    manager.read_window(1, ((0, 4), (0, 4)), masked=False)
    assert len(manager._window_cache) == 0


def test_window_cache_cleared_on_refresh():
    handles = []

    def opener(url, mode="r", **kwargs):
        handle = FakeHandle(
            url, "v1", read_error=_auth_error() if len(handles) == 0 else None
        )
        handles.append(handle)
        return handle

    resolver = ScriptedResolver(
        [ResolvedAsset("url-1", version="v1"), ResolvedAsset("url-2", version="v1")]
    )
    manager = _make_manager(resolver, opener=opener, window_cache_size=4)
    manager.read_window(1, ((0, 4), (0, 4)), masked=False)
    assert len(manager._window_cache) == 1
    manager.force_refresh()
    assert len(manager._window_cache) == 0


def test_metadata_cache_partitioned_by_version():
    resolver = ScriptedResolver([ResolvedAsset("url-1", version="v1")])
    manager = _make_manager(resolver)
    manager.set_metadata("tags", {"a": 1})
    assert manager.get_metadata("tags") == {"a": 1}
    assert manager.get_metadata("missing") is None


# ---------------------------------------------------------------------------
# Resolver registry
# ---------------------------------------------------------------------------


def test_resolver_registry_roundtrip():
    class MyResolver(RemoteAssetResolver):
        def resolve(self, identity, *, force_refresh=False):
            return ResolvedAsset("u")

    register_resolver("test-my-resolver", MyResolver)
    assert get_registered_resolver("test-my-resolver") is MyResolver
    with pytest.raises(KeyError):
        get_registered_resolver("does-not-exist")


def test_simple_resolver_wraps_errors():
    def bad(identity, force_refresh=False):
        raise RuntimeError("catalog down")

    resolver = SimpleResolver(bad)
    with pytest.raises(AssetResolutionError, match="catalog down"):
        resolver.resolve(AssetIdentity("a"))


def test_simple_resolver_rejects_wrong_return_type():
    resolver = SimpleResolver(lambda identity, force_refresh=False: "not-a-resolved")
    with pytest.raises(AssetResolutionError):
        resolver.resolve(AssetIdentity("a"))
