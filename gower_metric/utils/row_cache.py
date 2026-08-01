from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from gower_metric.utils.to_array import to_array

DEFAULT_MAXSIZE = 4096

Transform = Callable[[np.ndarray], pd.DataFrame | np.ndarray]


def _record_key(record: Any) -> tuple:
    """Build a hashable key from a raw record without converting it.

    Args:
        record (Any): Raw record: array, Series, or sequence of feature values.

    Returns:
        tuple: Key identifying the record by value.

    Raises:
        TypeError: If the record is not iterable. Callers treat that as
            "uncacheable" rather than as a failure.

    """
    if isinstance(record, np.ndarray):
        flat = record if record.ndim == 1 else record.ravel()
        return tuple(flat.tolist())
    if isinstance(record, pd.Series):
        return tuple(record.tolist())
    return tuple(record)


class RowEncodeCache:
    """Bounded LRU turning raw records into native rows."""

    def __init__(self, maxsize: int = DEFAULT_MAXSIZE) -> None:
        """Initialize an empty cache.

        Args:
            maxsize (int): Maximum number of encoded records retained.

        """
        self.maxsize = maxsize
        self._entries: OrderedDict[tuple, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        """Return the number of memorized records.

        Returns:
            int: Current entry count.

        """
        return len(self._entries)

    def clear(self) -> None:
        """Drop every memorized record."""
        self._entries.clear()

    def __getstate__(self) -> dict[str, Any]:
        """Return the picklable state without the entries.

        Returns:
            dict[str, Any]: Configuration needed to restore an empty cache.

        """
        return {"maxsize": self.maxsize}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore an empty cache with the stored capacity.

        Args:
            state (dict[str, Any]): State produced by ``__getstate__``.

        """
        self.maxsize = state["maxsize"]
        self._entries = OrderedDict()

    def encode_pair(
        self,
        a: Any,
        b: Any,
        transform: Transform,
        data_type: type[np.floating],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return both records encoded.

        Args:
            a (Any): First raw record.
            b (Any): Second raw record.
            transform (Transform): Encoder mapping a ``(n, n_features)`` block of
                raw values to its numeric representation.
            data_type (type[np.floating]): Native compute precision of the result.

        Returns:
            tuple[np.ndarray, np.ndarray]: Contiguous encoded records.

        """
        try:
            key_a = _record_key(a)
            key_b = _record_key(b)
            hit_a, hit_b = self._entries.get(key_a), self._entries.get(key_b)
        except TypeError:
            return self._encode(a, b, transform, data_type)

        if hit_a is not None and hit_b is not None:
            self._entries.move_to_end(key_a)
            self._entries.move_to_end(key_b)
            return hit_a, hit_b

        encoded_a, encoded_b = self._encode(a, b, transform, data_type)
        self._store(key_a, encoded_a)
        self._store(key_b, encoded_b)
        return encoded_a, encoded_b

    @staticmethod
    def _encode(
        a: Any,
        b: Any,
        transform: Transform,
        data_type: type[np.floating],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode both records in a single ``transform`` call.

        Args:
            a (Any): First raw record.
            b (Any): Second raw record.
            transform (Transform): Encoder to apply.
            data_type (type[np.floating]): Native compute precision of the result.

        Returns:
            tuple[np.ndarray, np.ndarray]: Contiguous encoded records.

        """
        rows = np.vstack((to_array(a).reshape(1, -1), to_array(b).reshape(1, -1)))
        pair = np.ascontiguousarray(transform(rows), dtype=data_type)
        return pair[0], pair[1]

    def _store(self, key: tuple, row: np.ndarray) -> None:
        """Insert one encoded record as most-recently-used and evict if needed.

        Args:
            key (tuple): Hashable key built from the raw record.
            row (np.ndarray): The encoded record.

        """
        self._entries[key] = row
        self._entries.move_to_end(key)
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)
