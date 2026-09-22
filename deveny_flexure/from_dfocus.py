# SPDX-License-Identifier: MPL-2.0
"""Compatibility exports for the shared DeVeny extraction routines.

The maintained implementations live in :mod:`pyldt.flexure`. This module is
kept so older analysis scripts importing ``from_dfocus`` continue to work.
"""

from pyldt.flexure import extract_spectrum, find_lines, gaussfit_func, specavg

__all__ = ["extract_spectrum", "find_lines", "gaussfit_func", "specavg"]
