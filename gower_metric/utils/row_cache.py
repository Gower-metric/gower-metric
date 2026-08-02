# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

from collections import OrderedDict
from typing import cast

import numpy as np
import pandas as pd

from gower_metric._typing import (
    FloatArray,
    FloatDType,
    Record,
    RecordKey,
    Transform,
)
from gower_metric.utils.to_array import to_array

DEFAULT_MAXSIZE = 4096


def _record_key(record: Record) -> RecordKey:
    """Build a hashable key from a raw record without converting it.

    Args:
        record (Record): Raw record: array, Series, or sequence of feature values.

    Returns:
        RecordKey: Key identifying the record by value.

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
        self._entries: OrderedDict[RecordKey, FloatArray] = OrderedDict()

    def __len__(self) -> int:
        """Return the number of memorized records.

        Returns:
            int: Current entry count.

        """
        return len(self._entries)

    def clear(self) -> None:
        """Drop every memorized record."""
        self._entries.clear()

    def __getstate__(self) -> dict[str, object]:
        """Return the picklable state without the entries.

        Returns:
            dict[str, object]: Configuration needed to restore an empty cache.

        """
        return {"maxsize": self.maxsize}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore an empty cache with the stored capacity.

        Args:
            state (dict[str, object]): State produced by ``__getstate__``.

        """
        self.maxsize = int(cast("int", state["maxsize"]))
        self._entries = OrderedDict()

    def encode_pair(
        self,
        a: Record,
        b: Record,
        transform: Transform,
        data_type: FloatDType,
    ) -> tuple[FloatArray, FloatArray]:
        """Return both records encoded.

        Args:
            a (Record): First raw record.
            b (Record): Second raw record.
            transform (Transform): Encoder mapping a ``(n, n_features)`` block of
                raw values to its numeric representation.
            data_type (FloatDType): Native compute precision of the result.

        Returns:
            tuple[FloatArray, FloatArray]: Contiguous encoded records.

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
        a: Record,
        b: Record,
        transform: Transform,
        data_type: FloatDType,
    ) -> tuple[FloatArray, FloatArray]:
        """Encode both records in a single ``transform`` call.

        Args:
            a (Record): First raw record.
            b (Record): Second raw record.
            transform (Transform): Encoder to apply.
            data_type (FloatDType): Native compute precision of the result.

        Returns:
            tuple[FloatArray, FloatArray]: Contiguous encoded records.

        """
        rows = np.vstack((to_array(a).reshape(1, -1), to_array(b).reshape(1, -1)))
        pair = np.ascontiguousarray(transform(rows), dtype=data_type)
        return pair[0], pair[1]

    def _store(self, key: RecordKey, row: FloatArray) -> None:
        """Insert one encoded record as most-recently-used and evict if needed.

        Args:
            key (RecordKey): Hashable key built from the raw record.
            row (FloatArray): The encoded record.

        """
        self._entries[key] = row
        self._entries.move_to_end(key)
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)
