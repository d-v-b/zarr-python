from collections.abc import MutableMapping
from typing import Any, List, Mapping, Optional, Sequence, Union

from zarr.meta import Metadata2
from zarr.util import normalize_storage_path
from zarr.context import Context


# v2 store keys
array_meta_key = ".zarray"
group_meta_key = ".zgroup"
attrs_key = ".zattrs"

DEFAULT_ZARR_VERSION = 2


class BaseStore(MutableMapping):
    """Abstract base class for store implementations.

    This is a thin wrapper over MutableMapping that provides methods to check
    whether a store is readable, writeable, eraseable and or listable.

    Stores cannot be mutable mapping as they do have a couple of other
    requirements that would break Liskov substitution principle (stores only
    allow strings as keys, mutable mapping are more generic).

    Having no-op base method also helps simplifying store usage and do not need
    to check the presence of attributes and methods, like `close()`.

    Stores can be used as context manager to make sure they close on exit.

    .. added: 2.11.0

    """

    _readable = True
    _writeable = True
    _erasable = True
    _listable = True
    _store_version = 2
    _metadata_class = Metadata2

    def is_readable(self):
        return self._readable

    def is_writeable(self):
        return self._writeable

    def is_listable(self):
        return self._listable

    def is_erasable(self):
        return self._erasable

    def __enter__(self):
        if not hasattr(self, "_open_count"):
            self._open_count = 0
        self._open_count += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._open_count -= 1
        if self._open_count == 0:
            self.close()

    def close(self) -> None:
        """Do nothing by default"""
        pass

    def rename(self, src_path: str, dst_path: str) -> None:
        if not self.is_erasable():
            raise NotImplementedError(
                f'{type(self)} is not erasable, cannot call "rename"'
            )  # pragma: no cover
        _rename_from_keys(self, src_path, dst_path)

    @staticmethod
    def _ensure_store(store: Any):
        """
        We want to make sure internally that zarr stores are always a class
        with a specific interface derived from ``BaseStore``, which is slightly
        different than ``MutableMapping``.

        We'll do this conversion in a few places automatically
        """
        from zarr.storage import KVStore  # avoid circular import

        if isinstance(store, BaseStore):
            if not store._store_version == 2:
                raise ValueError(
                    f"cannot initialize a v2 store with a v{store._store_version} store"
                )
            return store
        elif isinstance(store, MutableMapping):
            return KVStore(store)
        else:
            for attr in [
                "keys",
                "values",
                "get",
                "__setitem__",
                "__getitem__",
                "__delitem__",
                "__contains__",
            ]:
                if not hasattr(store, attr):
                    break
            else:
                return KVStore(store)

        raise ValueError(
            "Starting with Zarr 2.11.0, stores must be subclasses of "
            "BaseStore, if your store exposes the MutableMapping interface "
            f"wrap it in Zarr.storage.KVStore. Got {store}"
        )

    def getitems(
        self, keys: Sequence[str], *, contexts: Mapping[str, Context]
    ) -> Mapping[str, Any]:
        """Retrieve data from multiple keys.

        Parameters
        ----------
        keys : Iterable[str]
            The keys to retrieve
        contexts: Mapping[str, Context]
            A mapping of keys to their context. Each context is a mapping of store
            specific information. E.g. a context could be a dict telling the store
            the preferred output array type: `{"meta_array": cupy.empty(())}`

        Returns
        -------
        Mapping
            A collection mapping the input keys to their results.

        Notes
        -----
        This default implementation uses __getitem__() to read each key sequentially and
        ignores contexts. Overwrite this method to implement concurrent reads of multiple
        keys and/or to utilize the contexts.
        """
        return {k: self[k] for k in keys if k in self}


class Store(BaseStore):
    """Abstract store class used by implementations following the Zarr v2 spec.

    Adds public `listdir`, `rename`, and `rmdir` methods on top of BaseStore.

    .. added: 2.11.0

    """

    def listdir(self, path: str = "") -> List[str]:
        path = normalize_storage_path(path)
        return _listdir_from_keys(self, path)

    def rmdir(self, path: str = "") -> None:
        if not self.is_erasable():
            raise NotImplementedError(
                f'{type(self)} is not erasable, cannot call "rmdir"'
            )  # pragma: no cover
        path = normalize_storage_path(path)
        _rmdir_from_keys(self, path)


# allow MutableMapping for backwards compatibility
StoreLike = Union[BaseStore, MutableMapping]


def _path_to_prefix(path: Optional[str]) -> str:
    # assume path already normalized
    if path:
        prefix = path + "/"
    else:
        prefix = ""
    return prefix


def _rename_from_keys(store: BaseStore, src_path: str, dst_path: str) -> None:
    # assume path already normalized
    src_prefix = _path_to_prefix(src_path)
    dst_prefix = _path_to_prefix(dst_path)
    version = getattr(store, "_store_version", 2)
    if version == 2:
        for key in list(store.keys()):
            if key.startswith(src_prefix):
                new_key = dst_prefix + key.lstrip(src_prefix)
                store[new_key] = store.pop(key)
    else:
        raise NotImplementedError("This function only supports Zarr version 2.")


def _rmdir_from_keys(store: StoreLike, path: Optional[str] = None) -> None:
    # assume path already normalized
    prefix = _path_to_prefix(path)
    for key in list(store.keys()):
        if key.startswith(prefix):
            del store[key]


def _listdir_from_keys(store: BaseStore, path: Optional[str] = None) -> List[str]:
    # assume path already normalized
    prefix = _path_to_prefix(path)
    children = set()
    for key in list(store.keys()):
        if key.startswith(prefix) and len(key) > len(prefix):
            suffix = key[len(prefix) :]
            child = suffix.split("/")[0]
            children.add(child)
    return sorted(children)


def _prefix_to_array_key(store: StoreLike, prefix: str) -> str:
    if getattr(store, "_store_version", 2) == 3:
        raise NotImplementedError("This function only supports Zarr version 2.")
    else:
        key = prefix + array_meta_key
    return key


def _prefix_to_group_key(store: StoreLike, prefix: str) -> str:
    if getattr(store, "_store_version", 2) == 3:
        raise NotImplementedError("This function only supports Zarr version 2.")
    else:
        key = prefix + group_meta_key
    return key


def _prefix_to_attrs_key(store: StoreLike, prefix: str) -> str:
    if getattr(store, "_store_version", 2) == 3:
        raise NotImplementedError("This function only supports Zarr version 2.")
    else:
        key = prefix + attrs_key
    return key
