"""
Remote asset resolution for lazy raster reads.

Lazy Dask tasks may be executed long after the signed URL of a remote asset
has expired.  Instead of storing a (possibly short-lived) URL, callers store a
stable :class:`AssetIdentity` together with an :class:`RemoteAssetResolver`
that knows how to turn the identity into a current URL and open options at
the moment the data is actually read.

The pieces in this module guarantee that:

* the current URL and options are resolved at open time, not graph-build time;
* authentication failures trigger a bounded, coordinated refresh after which
  the asset is re-opened;
* the content version, grid and band schema recorded at the first open are
  re-checked on every (re-)open so that pixels from a new asset version can
  never be mixed into the same computation;
* only the re-resolvable identity is pickled, never a live file handle;
* resources opened during a computation are released on close.

Plain local paths continue to use the regular :class:`~rioxarray._io.URIManager`
path in :func:`rioxarray.open_rasterio` and are not affected.
"""

from __future__ import annotations

import abc
import contextlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from rioxarray.exceptions import RioXarrayError

__all__ = [
    "AssetConsistencyError",
    "AssetIdentity",
    "AssetResolutionError",
    "AssetSchemaMismatchError",
    "AssetURIManager",
    "AssetVersionMismatchError",
    "RemoteAssetResolver",
    "ResolvedAsset",
    "SimpleResolver",
    "get_registered_resolver",
    "is_credential_error",
    "is_remote_asset",
    "register_resolver",
]


class AssetResolutionError(RioXarrayError):
    """Raised when a remote asset cannot be resolved or opened."""


class AssetConsistencyError(RioXarrayError):
    """Base class for asset version/schema drift errors."""


class AssetVersionMismatchError(AssetConsistencyError):
    """Raised when a re-opened asset reports a different content version."""


class AssetSchemaMismatchError(AssetConsistencyError):
    """Raised when a re-opened asset has a different grid or band schema."""


# ---------------------------------------------------------------------------
# Identity / resolution value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetIdentity:
    """Stable, serializable identity of a remote asset.

    The identity does not contain credentials or signed URLs.  It is everything
    a :class:`RemoteAssetResolver` needs to locate the *current* representation
    of the asset.

    Parameters
    ----------
    asset_id:
        Stable identifier of the asset (e.g. a STAC item/asset key or a
        catalog UUID).  It must remain valid across URL refreshes.
    version:
        Optional content version the computation is pinned to.  When given,
        every (re-)open must resolve to exactly this version; otherwise an
        :class:`AssetVersionMismatchError` is raised.  When ``None``, the
        version reported by the first successful open is pinned for the
        lifetime of the manager.
    extra:
        Additional resolver-specific identity information (e.g. collection,
        band names).  Must contain only pickleable, immutable values.
    """

    asset_id: str
    version: Optional[str] = None
    extra: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.asset_id, str) or not self.asset_id:
            raise ValueError("'asset_id' must be a non-empty string")
        # ``tuple(tuple(...))`` keeps the object hashable and pickleable even
        # if a mapping was supplied to the ``extra`` keyword.
        if not isinstance(self.extra, tuple):
            object.__setattr__(self, "extra", tuple(sorted(self.extra.items())))

    @classmethod
    def from_mapping(
        cls,
        asset_id: str,
        version: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> "AssetIdentity":
        """Build an identity from a plain mapping of extra fields."""
        return cls(
            asset_id=asset_id,
            version=version,
            extra=tuple(sorted((extra or {}).items())),
        )

    def extra_mapping(self) -> dict[str, Any]:
        """Return the extra identity fields as a dictionary."""
        return dict(self.extra)

    def pinned(self, version: str) -> "AssetIdentity":
        """Return a copy pinned to the given content version."""
        return replace(self, version=version)


@dataclass(frozen=True)
class AssetSchema:
    """Immutable snapshot of the schema of an opened raster.

    The snapshot captures everything that must stay identical for pixels read
    at different times to be interchangeable within one computation: the
    raster grid (dimensions, geo-transform and CRS) and the band schema
    (count, ordering, dtypes, nodata and scaling).
    """

    count: int
    height: int
    width: int
    transform: tuple[float, float, float, float, float, float]
    crs: Optional[str]
    dtypes: tuple[str, ...]
    nodatavals: tuple[Optional[float], ...]
    indexes: tuple[int, ...]
    scales: tuple[float, ...]
    offsets: tuple[float, ...]
    descriptions: tuple[Optional[str], ...]
    units: tuple[Optional[str], ...]

    @classmethod
    def from_dataset(cls, riods: Any) -> "AssetSchema":
        """Snapshot the schema of an open rasterio dataset."""
        transform = riods.transform
        return cls(
            count=int(riods.count),
            height=int(riods.height),
            width=int(riods.width),
            transform=(
                float(transform.a),
                float(transform.b),
                float(transform.c),
                float(transform.d),
                float(transform.e),
                float(transform.f),
            ),
            crs=riods.crs.to_string() if riods.crs else None,
            dtypes=tuple(riods.dtypes),
            nodatavals=tuple(riods.nodatavals),
            indexes=tuple(int(index) for index in riods.indexes),
            scales=tuple(float(scale) for scale in riods.scales),
            offsets=tuple(float(offset) for offset in riods.offsets),
            descriptions=tuple(riods.descriptions),
            units=tuple(riods.units),
        )

    def diff(self, other: "AssetSchema") -> list[str]:
        """Human readable list of schema differences with ``other``."""
        differences = []
        for field_name in (
            "count",
            "height",
            "width",
            "transform",
            "crs",
            "dtypes",
            "nodatavals",
            "indexes",
            "scales",
            "offsets",
            "descriptions",
            "units",
        ):
            if getattr(self, field_name) != getattr(other, field_name):
                differences.append(
                    f"{field_name}: {getattr(self, field_name)!r} != "
                    f"{getattr(other, field_name)!r}"
                )
        return differences

    def assert_matches(self, other: "AssetSchema", *, asset_id: str) -> None:
        """Raise :class:`AssetSchemaMismatchError` if schemas differ."""
        differences = self.diff(other)
        if differences:
            raise AssetSchemaMismatchError(
                f"Grid/band schema of asset {asset_id!r} changed between "
                "opens. Mixing pixels from different schemas in a single "
                "computation is not allowed:\n  - "
                + "\n  - ".join(differences)
            )


@dataclass(frozen=True)
class ResolvedAsset:
    """Result of resolving an :class:`AssetIdentity`.

    Parameters
    ----------
    url:
        Current URL (signed or otherwise) to pass to the rasterio opener.
    open_options:
        Keyword arguments to pass to the opener together with the URL.
        Credentials/headers typically live here.
    version:
        Content version of the resolved asset, or ``None`` if the resolver
        cannot determine it.
    """

    url: str
    open_options: dict[str, Any] = field(default_factory=dict)
    version: Optional[str] = None


# ---------------------------------------------------------------------------
# Resolver abstraction
# ---------------------------------------------------------------------------


class RemoteAssetResolver(abc.ABC):
    """Resolve stable asset identities to current URLs and open options.

    Implementations talk to a catalog, STAC API, object store or any other
    authority that can mint a fresh signed URL for an asset.  Resolvers must
    be pickleable so that they travel with Dask graphs to worker processes;
    the easiest way is to keep only simple configuration on the instance and
    register the class via :func:`register_resolver`.
    """

    @abc.abstractmethod
    def resolve(
        self,
        identity: AssetIdentity,
        *,
        force_refresh: bool = False,
    ) -> ResolvedAsset:
        """Return the current URL/options for ``identity``.

        Parameters
        ----------
        identity:
            The stable asset identity to resolve.
        force_refresh:
            If ``True``, discard any cached URL/credentials and mint a fresh
            resolution.  Set after an authentication failure.

        Raises
        ------
        AssetResolutionError
            If the asset cannot be resolved.
        """

    # pylint: disable=unused-argument
    def is_auth_error(self, exc: BaseException) -> bool:
        """Whether ``exc`` represents an expired/invalid credential.

        The default recognises the common HTTP 401/403 status codes and
        "expired token"/"signature" style messages emitted by GDAL and object
        stores.  Override for resolver-specific exception types.
        """
        return is_credential_error(exc)


class SimpleResolver(RemoteAssetResolver):
    """Resolver backed by a user supplied callable.

    The callable has the same signature as
    :meth:`RemoteAssetResolver.resolve` (without ``self``) and returns a
    :class:`ResolvedAsset`.  Instances are pickleable as long as the callable
    is a module-level function or a registered resolver name is used.
    """

    def __init__(
        self,
        resolve_callable,
        *,
        auth_error_callable=None,
    ) -> None:
        self._resolve_callable = resolve_callable
        self._auth_error_callable = auth_error_callable

    def resolve(
        self,
        identity: AssetIdentity,
        *,
        force_refresh: bool = False,
    ) -> ResolvedAsset:
        try:
            resolved = self._resolve_callable(
                identity, force_refresh=force_refresh
            )
        except AssetResolutionError:
            raise
        except Exception as exc:  # pragma: no branch - defensive
            raise AssetResolutionError(
                f"Failed to resolve asset {identity.asset_id!r}: {exc}"
            ) from exc
        if not isinstance(resolved, ResolvedAsset):
            raise AssetResolutionError(
                "Resolver callable must return a ResolvedAsset instance, "
                f"got {type(resolved)!r}"
            )
        return resolved

    def is_auth_error(self, exc: BaseException) -> bool:
        if self._auth_error_callable is not None:
            return bool(self._auth_error_callable(exc))
        return super().is_auth_error(exc)


# Registry allowing resolvers to be transported by name (e.g. across Dask
# workers) without relying on pickling lambdas.
_RESOLVER_REGISTRY: dict[str, type[RemoteAssetResolver]] = {}
_REGISTRY_LOCK = threading.Lock()


def register_resolver(name: str, resolver_cls: type[RemoteAssetResolver]) -> None:
    """Register a resolver class under a stable ``name``.

    A registered class can be referenced from pickled asset managers without
    pickling the class itself, which is convenient for Dask workers that only
    know the resolver by name.
    """
    with _REGISTRY_LOCK:
        _RESOLVER_REGISTRY[name] = resolver_cls


def get_registered_resolver(name: str) -> type[RemoteAssetResolver]:
    """Return the resolver class registered under ``name``."""
    with _REGISTRY_LOCK:
        try:
            return _RESOLVER_REGISTRY[name]
        except KeyError:
            raise KeyError(
                f"No remote asset resolver registered under name {name!r}. "
                "Call rioxarray.assets.register_resolver() first."
            ) from None


def is_credential_error(exc: BaseException) -> bool:
    """Heuristic check for expired/invalid credential errors.

    Works on rasterio/GDAL ``RasterioIOError`` messages (``HTTP error 401``,
    ``SignatureDoesNotMatch``, expired token strings, ...) as well as common
    cloud SDK exception class names without importing those SDKs.
    """
    class_name = type(exc).__name__.lower()
    if any(
        token in class_name
        for token in ("unauthor", "forbidden", "credential", "expiredtoken")
    ):
        return True
    message = str(exc).lower()
    # HTTP 401 (unauthorized) / 403 (forbidden) always indicate an
    # authentication/authorization problem for a URL that used to work.
    if "401" in message or "403" in message:
        return True
    if any(
        token in message
        for token in (
            "expired",
            "expire",
            "invalid token",
            "access token",
            "signaturedoesnotmatch",
            "invalid signature",
            "request has expired",
            "token is expired",
            "credentials",
            "access denied",
        )
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Thread local state
# ---------------------------------------------------------------------------


class _AssetThreadState(threading.local):
    def __init__(self) -> None:  # pylint: disable=super-init-not-called
        # Generation of the resolution this thread's handle was opened from.
        self.generation = -1
        self.thread_manager: Optional["ThreadAssetManager"] = None


class ThreadAssetManager:
    """Opens and closes the rasterio handle for a single thread."""

    def __init__(
        self,
        opener: Any,
        resolved: ResolvedAsset,
        *,
        mode: str,
        kwargs: dict[str, Any],
        reader_wrapper: Any = None,
    ) -> None:
        self._opener = opener
        self._resolved = resolved
        self._mode = mode
        self._kwargs = dict(kwargs)
        self._reader_wrapper = reader_wrapper
        self._file_handle: Any = None

    def _open_kwargs(self) -> dict[str, Any]:
        merged = dict(self._resolved.open_options)
        merged.update(self._kwargs)
        return merged

    def _open(self) -> Any:
        riods = self._opener(
            self._resolved.url, mode=self._mode, **self._open_kwargs()
        )
        if self._reader_wrapper is not None:
            riods = self._reader_wrapper(riods)
        return riods

    @property
    def file_handle(self) -> Any:
        if self._file_handle is None:
            self._file_handle = self._open()
        return self._file_handle

    def close(self) -> None:
        if self._file_handle is not None:
            with contextlib.suppress(Exception):
                self._file_handle.close()
            self._file_handle = None

    def __enter__(self) -> "ThreadAssetManager":
        return self

    def __exit__(self, type_, value, traceback) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Main manager
# ---------------------------------------------------------------------------


class _VersionedWindowCache:
    """Small LRU window cache whose entries are namespaced by asset version."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._data: "OrderedDict[tuple, Any]" = OrderedDict()

    def get(self, version: Optional[str], key: tuple) -> Optional[Any]:
        cache_key = (version, key)
        try:
            value = self._data.pop(cache_key)
        except KeyError:
            return None
        self._data[cache_key] = value
        return value

    def put(self, version: Optional[str], key: tuple, value: Any) -> None:
        if self._maxsize <= 0:
            return
        cache_key = (version, key)
        self._data.pop(cache_key, None)
        self._data[cache_key] = value
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


def _normalize_window_key(
    band_key: Any, window: Any, masked: bool, read_kwargs: dict[str, Any]
) -> tuple:
    """Build a hashable cache key from rasterio read arguments."""

    def _hashable(value: Any) -> Any:
        if isinstance(value, list):
            return ("list", tuple(_hashable(item) for item in value))
        if isinstance(value, tuple):
            return ("tuple", tuple(_hashable(item) for item in value))
        if isinstance(value, dict):
            return (
                "dict",
                tuple(sorted((key, _hashable(val)) for key, val in value.items())),
            )
        return value

    return (
        _hashable(band_key),
        _hashable(window),
        bool(masked),
        tuple(sorted(read_kwargs.items())),
    )


class AssetURIManager:
    """File manager for remote assets identified by a stable identity.
    The manager resolves a fresh URL/open-options pair through
    ``resolver`` the first time a thread needs a handle and again whenever a
    credential is rejected.  Resolutions are single-flight: concurrent
    threads share one refresh instead of each minting a new signed URL.

    The content version and :class:`AssetSchema` observed at the first
    successful open are pinned for the lifetime of the manager.  Every
    subsequent (re-)open is checked against them; a mismatch raises an
    :class:`AssetConsistencyError` *before* any new pixels can be returned,
    so a single lazy computation can never silently mix asset versions.

    Parameters
    ----------
    identity:
        Stable, pickleable identity of the remote asset.
    resolver:
        Resolver used to obtain current URLs and open options.
    opener:
        Callable used to open the resolved URL (typically
        :func:`rasterio.open`).
    mode:
        Open mode forwarded to the opener.
    kwargs:
        Extra opener keyword arguments merged on top of the resolver's
        open options.
    max_refreshes:
        Maximum number of credential refreshes (re-resolutions with
        ``force_refresh=True``) the manager will attempt in total.
    window_cache_size:
        Per-version LRU size for cached read windows.  ``0`` disables the
        window cache.  Cached windows are always namespaced by content
        version, so a refresh can never return data from another version.
    """

    def __init__(
        self,
        identity: AssetIdentity,
        resolver: RemoteAssetResolver,
        opener: Any,
        *,
        mode: str = "r",
        kwargs: Optional[dict[str, Any]] = None,
        reader_wrapper: Any = None,
        max_refreshes: int = 2,
        window_cache_size: int = 0,
    ) -> None:
        if max_refreshes < 0:
            raise ValueError("'max_refreshes' must be >= 0")
        if window_cache_size < 0:
            raise ValueError("'window_cache_size' must be >= 0")
        self._identity = identity
        self._resolver = resolver
        self._opener = opener
        self._mode = mode
        self._kwargs = {} if kwargs is None else dict(kwargs)
        self._reader_wrapper = reader_wrapper
        self._max_refreshes = max_refreshes
        self._local = _AssetThreadState()

        # Coordination across threads.  One lock guards every resolution
        # (initial or forced refresh) so that concurrent threads always
        # observe a consistent (generation, resolved URL, version) tuple.
        self._resolve_lock = threading.Lock()
        self._schema_lock = threading.Lock()
        self._all_handles_lock = threading.Lock()
        # Every ThreadAssetManager opened through this manager is tracked
        # here so close_all() can reclaim handles from threads that have
        # since exited.  Entries are removed on close(), so the set does
        # not grow over the manager's lifetime.
        self._all_thread_managers: set[ThreadAssetManager] = set()

        # Resolution state.  ``_generation`` bumps on every successful
        # refresh; threads re-open when their handle's generation lags.
        self._generation = 0
        self._refresh_count = 0
        self._resolved: Optional[ResolvedAsset] = None
        self._pinned_version: Optional[str] = identity.version
        self._schema: Optional[AssetSchema] = None

        # Version aware caches.
        self._metadata_cache: dict[Optional[str], dict[str, Any]] = {}
        self._window_cache = _VersionedWindowCache(maxsize=window_cache_size)

    # -- pickle support: only the re-resolvable identity travels -----------

    def __getstate__(self) -> dict[str, Any]:
        return {
            "identity": self._identity,
            "resolver": self._resolver,
            "opener": self._opener,
            "mode": self._mode,
            "kwargs": self._kwargs,
            "reader_wrapper": self._reader_wrapper,
            "max_refreshes": self._max_refreshes,
            "window_cache_size": self._window_cache._maxsize,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(
            state["identity"],
            state["resolver"],
            state["opener"],
            mode=state["mode"],
            kwargs=state["kwargs"],
            reader_wrapper=state.get("reader_wrapper"),
            max_refreshes=state["max_refreshes"],
            window_cache_size=state["window_cache_size"],
        )

    # -- properties --------------------------------------------------------

    @property
    def identity(self) -> AssetIdentity:
        """Stable identity of the managed asset."""
        return self._identity

    @property
    def pinned_version(self) -> Optional[str]:
        """Content version the manager is pinned to (``None`` until open)."""
        return self._pinned_version

    @property
    def schema(self) -> Optional[AssetSchema]:
        """Pinned grid/band schema (``None`` until the first open)."""
        return self._schema

    # -- resolution --------------------------------------------------------

    def _resolve_initial(self) -> ResolvedAsset:
        """Resolve the asset for the first time, single-flight across threads."""
        with self._resolve_lock:
            if self._resolved is None:
                resolved = self._resolver.resolve(
                    self._identity, force_refresh=False
                )
                self._verify_resolved_version(resolved)
                self._resolved = resolved
            return self._resolved

    def force_refresh(self) -> int:
        """Force a credential refresh and return the new generation.

        Always invokes the resolver with ``force_refresh=True`` (subject to
        the refresh budget).  Prefer :meth:`refresh_if_stale` from read
        retry paths so that concurrent failures share a single refresh.

        Raises
        ------
        AssetResolutionError
            If the refresh budget (``max_refreshes``) is exhausted.
        AssetVersionMismatchError
            If the refreshed resolution no longer matches the pinned version.
        """
        with self._resolve_lock:
            return self._refresh_locked()

    def refresh_if_stale(self, failed_generation: int) -> None:
        """Refresh unless a newer generation already exists.

        Single-flight coordination for read retry paths: when several
        threads see their signed URL expire at the same generation, the
        first one performs the resolver refresh while the others wait on
        the lock and then reuse the new generation without minting another
        URL.  A thread whose handle already lags ``failed_generation``
        behind the current generation simply re-opens on acquire.
        """
        with self._resolve_lock:
            if self._generation > failed_generation:
                # Another thread already refreshed: reuse its generation.
                return
            self._refresh_locked()

    def _refresh_locked(self) -> int:
        """Perform one forced refresh.  Caller must hold ``_resolve_lock``."""
        if self._refresh_count >= self._max_refreshes:
            raise AssetResolutionError(
                "Authentication for asset "
                f"{self._identity.asset_id!r} failed and the refresh "
                f"limit of {self._max_refreshes} has been reached."
            )
        self._refresh_count += 1
        resolved = self._resolver.resolve(
            self._identity, force_refresh=True
        )
        self._verify_resolved_version(resolved)
        self._resolved = resolved
        self._generation += 1
        # Cached windows belong to the previously opened generation; drop
        # them so no stale pixel can be served after a refresh.
        self._window_cache.clear()
        return self._generation

    def _verify_resolved_version(self, resolved: ResolvedAsset) -> None:
        if resolved.version is None:
            return
        if self._pinned_version is None:
            # First successful open: pin whatever version was resolved.
            self._pinned_version = resolved.version
            self._identity = self._identity.pinned(resolved.version)
        elif resolved.version != self._pinned_version:
            raise AssetVersionMismatchError(
                f"Asset {self._identity.asset_id!r} resolved to version "
                f"{resolved.version!r} but the current computation is pinned "
                f"to version {self._pinned_version!r}. Re-open the asset to "
                "start a new computation; mixing versions is not allowed."
            )

    def _verify_schema(self, riods: Any) -> AssetSchema:
        observed = AssetSchema.from_dataset(riods)
        with self._schema_lock:
            if self._schema is None:
                self._schema = observed
            else:
                self._schema.assert_matches(
                    observed, asset_id=self._identity.asset_id
                )
        return observed

    # -- handle lifecycle --------------------------------------------------

    def _build_thread_manager(
        self, resolved: ResolvedAsset
    ) -> ThreadAssetManager:
        thread_manager = ThreadAssetManager(
            self._opener,
            resolved,
            mode=self._mode,
            kwargs=self._kwargs,
            reader_wrapper=self._reader_wrapper,
        )
        with self._all_handles_lock:
            self._all_thread_managers.add(thread_manager)
        return thread_manager

    def acquire(self, needs_lock: bool = True) -> Any:
        """Return a rasterio handle for the current thread.

        Resolves and opens the asset on first use, transparently re-opening
        against the current generation after a coordinated refresh when the
        held handle is stale.
        """
        # pylint: disable=unused-argument
        thread_manager = self._local.thread_manager
        if (
            thread_manager is not None
            and self._local.generation == self._generation
        ):
            return thread_manager.file_handle

        # Single-flight initial resolution; later threads reuse the result.
        resolved = self._resolve_initial()

        # Close the stale handle of *this* thread before re-opening.  Other
        # threads keep their handles until they notice the new generation.
        if thread_manager is not None:
            thread_manager.close()

        thread_manager = self._build_thread_manager(resolved)
        riods = thread_manager.file_handle
        # A refresh may have landed while this thread was waiting: verify
        # against whatever generation the new handle was actually opened on.
        self._verify_schema(riods)
        self._local.thread_manager = thread_manager
        self._local.generation = self._generation
        return riods

    @contextlib.contextmanager
    def acquire_context(self, needs_lock: bool = True):
        """Context manager yielding a handle, closing it on any error."""
        try:
            yield self.acquire(needs_lock=needs_lock)
        except Exception:
            self.close(needs_lock=needs_lock)
            raise

    def close(self, needs_lock: bool = True) -> None:
        """Close the calling thread's handle (if any)."""
        # pylint: disable=unused-argument
        thread_manager = self._local.thread_manager
        if thread_manager is not None:
            thread_manager.close()
            with self._all_handles_lock:
                self._all_thread_managers.discard(thread_manager)
            self._local.thread_manager = None
            self._local.generation = -1

    def close_all(self) -> None:
        """Close every handle opened by this manager in any thread."""
        with self._all_handles_lock:
            managers = list(self._all_thread_managers)
            self._all_thread_managers.clear()
        for thread_manager in managers:
            thread_manager.close()
        self._local.thread_manager = None
        self._local.generation = -1
        self._window_cache.clear()

    def __del__(self) -> None:
        # Reclaim resources opened during the current object's lifetime.
        try:
            self.close_all()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass

    # -- reads with transparent credential recovery ------------------------

    def _read_with_recovery(self, read_callable):
        """Run ``read_callable(riods)``, refreshing credentials on auth errors.

        Refreshes are single-flight: when several threads see their signed
        URL expire simultaneously, only one resolver refresh happens and all
        threads re-open against the resulting generation.  The total number
        of refreshes is bounded by ``max_refreshes``.
        """
        while True:
            try:
                riods = self.acquire()
                return read_callable(riods)
            except AssetResolutionError:
                # Failed initial resolution: nothing more to retry.
                raise
            except Exception as exc:
                if not self._resolver.is_auth_error(exc):
                    raise
                # Remember which generation this handle belonged to, then
                # release it before opening a fresh one.
                failed_generation = self._local.generation
                self.close()
                # Single-flight.  Concurrent failures on the same
                # generation share one resolver refresh; raises
                # AssetResolutionError once the budget is exhausted.
                self.refresh_if_stale(failed_generation)

    def read_window(
        self,
        band_key: Any,
        window: Any,
        *,
        masked: bool,
        read_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Read a window through the manager with auth recovery and caching.

        Windows are cached per pinned content version so that a refresh can
        never serve pixels belonging to another version.
        """
        read_kwargs = read_kwargs or {}
        cache_key = _normalize_window_key(
            band_key, window, masked, read_kwargs
        )

        cached = self._window_cache.get(self._pinned_version, cache_key)
        if cached is not None:
            return cached

        def _do_read(riods):
            return riods.read(
                band_key, window=window, masked=masked, **read_kwargs
            )

        result = self._read_with_recovery(_do_read)
        self._window_cache.put(self._pinned_version, cache_key, result)
        return result

    # -- version aware metadata cache --------------------------------------

    def get_metadata(self, key: str) -> Any:
        """Return cached metadata ``key`` for the pinned asset version."""
        return self._metadata_cache.get(self._pinned_version, {}).get(key)

    def set_metadata(self, key: str, value: Any) -> None:
        """Store metadata ``key`` under the pinned asset version."""
        self._metadata_cache.setdefault(self._pinned_version, {})[key] = value

    # -- xarray FileManager compatibility shims ----------------------------

    @property
    def kwargs(self) -> dict[str, Any]:
        """Opener keyword arguments (FileManager-compatible accessor)."""
        return dict(self._kwargs)


def is_remote_asset(filename: Any) -> bool:
    """Return whether ``filename`` is an :class:`AssetIdentity`."""
    return isinstance(filename, AssetIdentity)
