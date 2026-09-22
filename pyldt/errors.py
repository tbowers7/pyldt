# SPDX-License-Identifier: MPL-2.0
"""PyLDT exception hierarchy."""


class PyldtError(Exception):
    """Base class for PyLDT exceptions."""


class InputError(PyldtError):
    """Raised when a caller supplies invalid or inconsistent input."""
