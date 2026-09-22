# SPDX-License-Identifier: MPL-2.0
#  Created on 06-Jun-2022
#  @author: tbowers
"""
Public package interface and warning formatting helpers.
"""

from __future__ import annotations

# Built-In Libraries
import importlib
import pathlib
import typing

# Names are materialized by __getattr__; Pylint cannot infer lazy exports.
# pylint: disable=undefined-all-variable
__all__ = [
    "LMI",
    "NASA42",
    "imcombine",
    "savetime",
    "trim_oscan",
    "wrap_trim_oscan",
    "solve_field",
    "validate_solution",
    "mmms",
    "set_std_tickparams",
    "short_warning",
]
# pylint: enable=undefined-all-variable

_LAZY_EXPORTS = {
    **{name: "pyldt.reduction" for name in ("LMI", "NASA42")},
    "imcombine": "pyldt.combine",
    **{
        name: "pyldt.calibration"
        for name in ("savetime", "trim_oscan", "wrap_trim_oscan")
    },
    **{name: "pyldt.astrometry" for name in ("solve_field", "validate_solution")},
    **{name: "pyldt.utils" for name in ("mmms", "set_std_tickparams")},
}


def short_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: typing.TextIO | None = None,
    line: str | None = None,
) -> str:
    """
    Format a warning as a compact, single-line message.

    Parameters
    ----------
    message : Warning or str
        Warning instance or warning text.
    category : type of Warning
        Warning category.
    filename : str
        Source filename associated with the warning.
    lineno : int
        Source line number associated with the warning.
    file : file-like object, optional
        Output stream supplied by :mod:`warnings`. It is not used.
    line : str, optional
        Source line supplied by :mod:`warnings`. It is not used.

    Returns
    -------
    str
        Compact warning text.
    """
    del file, line
    return f" {category.__name__}: {message} ({pathlib.Path(filename).name}:{lineno})\n"


def __getattr__(name: str) -> typing.Any:
    """Load public objects on first access instead of importing heavy modules."""
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as err:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from err
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily exported names in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
