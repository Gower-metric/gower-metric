# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

_POSITIONAL_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def retired_args(
    reason: str,
    *names: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Accept and discard parameters the decorated function no longer declares.

    Args:
        reason (str): Predicate appended after the parameter names, e.g.
            ``"no longer has any effect"``.
        *names (str): Retired parameter names to absorb.

    Returns:
        Callable: A decorator applying the behaviour.

    Raises:
        ValueError: If a retired name is still declared by the decorated
            function, or if no names are given.

    Example:
        >>> @retired_args("no longer has any effect", "n_jobs")
        ... def compute(data, *, verbose=False):
        ...     return data
        >>> compute([1], n_jobs=4)

    """
    if not names:
        msg = "retired_args requires at least one parameter name"
        raise ValueError(msg)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(func)

        still_declared = sorted(set(names) & set(signature.parameters))
        if still_declared:
            msg = (
                f"{func.__qualname__} still declares {', '.join(still_declared)}; "
                "remove the parameter(s) from the signature so the decorator can "
                "absorb them"
            )
            raise ValueError(msg)

        accepts_var_positional = any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        max_positional = sum(
            parameter.kind in _POSITIONAL_KINDS
            for parameter in signature.parameters.values()
        )
        retired = ", ".join(names)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not accepts_var_positional and len(args) > max_positional:
                msg = (
                    f"{func.__name__}() takes {max_positional} positional "
                    f"arguments but {len(args)} were given. {retired} were "
                    "removed from the signature; pass the remaining options by "
                    "keyword."
                )
                raise TypeError(msg)

            supplied = sorted(name for name in names if name in kwargs)
            if supplied:
                for name in supplied:
                    del kwargs[name]
                warnings.warn(
                    f"{', '.join(supplied)} {reason}",
                    DeprecationWarning,
                    stacklevel=2,
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator
