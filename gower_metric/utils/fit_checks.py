# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from gower_metric.utils.transforms.encoding import fitted_codes

if TYPE_CHECKING:
    from gower_metric.core.metric import Gower


def check_precision(model: "Gower", col_names: Sequence[object] | None) -> None:
    """Refuse fitted codes, ranges or weights that ``model.data_type`` cannot hold."""
    for j, codes in fitted_codes(model).items():
        stored: dict[float, object] = {}
        for value, code in codes.items():
            clash = stored.setdefault(float(model.data_type(code)), value)
            if clash is not value:
                name = col_names[j] if col_names is not None else j
                msg = (
                    f"Column {name!r} ({model.feature_types[j]}) holds {clash} "
                    f"and {value}, whose codes {np.dtype(model.data_type).name} "
                    "stores as the same number. Use a wider data_type."
                )
                raise ValueError(msg)

    limit = float(np.finfo(model.data_type).max)
    precision = np.dtype(model.data_type).name
    for j, lo, hi in (
        *zip(
            model.numeric_indices,
            model.numeric_mins,
            model.numeric_maxs,
            strict=True,
        ),
        *zip(
            model.ratio_scale_indices,
            model.ratio_mins,
            model.ratio_maxs,
            strict=True,
        ),
    ):
        if max(abs(float(lo)), abs(float(hi)), float(hi) - float(lo)) > limit:
            name = col_names[j] if col_names is not None else j
            msg = (
                f"Column {name!r} spans {lo:g} to {hi:g}, which {precision} "
                f"cannot hold: its largest number is {limit:g}. Use a wider "
                "data_type."
            )
            raise ValueError(msg)
    if sum(model.weights.tolist()) > limit:
        msg = (
            f"feature_weights add up to more than {precision} can hold "
            f"({limit:g}). Only their ratios matter, so scale them down."
        )
        raise ValueError(msg)


def constant_features(model: "Gower") -> list[int]:
    """Return the indices of features that do not vary in the fitted data."""
    out: list[int] = []
    for indices, mins, maxs in (
        (model.numeric_indices, model.numeric_mins, model.numeric_maxs),
        (model.ratio_scale_indices, model.ratio_mins, model.ratio_maxs),
    ):
        out.extend(j for pos, j in enumerate(indices) if not maxs[pos] > mins[pos])
    for j in model.categorical_ordinal_indices:
        counts = np.asarray(model.cat_ord_metadata[j]["counts"])
        if np.count_nonzero(counts > 0) < 2:  # noqa: PLR2004
            out.append(int(j))
    return sorted(out)


def report_constant_features(
    model: "Gower",
    col_names: Sequence[object] | None,
) -> None:
    """Warn about constant features; refuse a fit that leaves nothing to compare."""
    stored = model.weights.astype(model.data_type)
    one = [*model.binary_symmetric_indices, *model.binary_asymmetric_indices]
    two = [*model.numeric_indices, *model.ratio_scale_indices]
    for step, cols in (
        ("one", one + model.categorical_nominal_indices),
        ("two", two + model.categorical_ordinal_indices),
    ):
        kept = [j for j in cols if j not in model.excluded_features]
        if model.conditional_distances and not stored[kept].any():
            msg = (
                f"step {step} of conditional_distances has nothing to compare: "
                "each of its features is constant or has weight 0 in "
                f"{np.dtype(model.data_type).name}"
            )
            raise ValueError(msg)
    if not model.excluded_features:
        return
    names = [
        repr(col_names[j]) if col_names is not None else str(j)
        for j in model.excluded_features
    ]
    if len(model.excluded_features) == len(model.feature_types):
        msg = (
            "Every feature is constant in the fitted data, so there is "
            "nothing to compare: " + ", ".join(names)
        )
        raise ValueError(msg)
    msg = (
        "Excluded from the coefficient because they are constant in the "
        "fitted data: " + ", ".join(names) + ". Podani (1999, p. 335): "
        "'variables with constant values are not allowed', in agreement "
        "with Gower's formula, which divides by the range. Each is "
        "dropped from both the numerator and the denominator, so the "
        "result is what you get without the column."
    )
    warnings.warn(msg, UserWarning, stacklevel=3)
