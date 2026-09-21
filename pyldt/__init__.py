# SPDX-License-Identifier: MPL-2.0
#  Created on 06-Jun-2022
#  @author: tbowers
"""
Public package interface and warning formatting helpers.
"""

from __future__ import annotations

# Imports for signal and log handling
import pathlib
import typing
import warnings

# Local Imports
from pyldt.astrometry import *  # noqa
from pyldt.reduction import *  # noqa
from pyldt.utils import *  # noqa


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


warnings.formatwarning = short_warning
