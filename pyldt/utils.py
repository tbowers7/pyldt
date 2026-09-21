# SPDX-License-Identifier: MPL-2.0
#  Created on 26-Oct-2020
#  @author: tbowers
"""
PyLDT contains image calibration routines for LDT facility instruments

Lowell Discovery Telescope (Lowell Observatory: Flagstaff, AZ)
http://www.lowell.edu

Utility files for use with PyLDT

There are some utility functions here...
"""

from __future__ import annotations

# Built-In Libraries
import typing

# Third-Party Libraries
import numpy as np

# Internal Imports


# Define API
__all__ = ["mmms", "set_std_tickparams"]


def mmms(image: np.ndarray) -> tuple[np.floating[typing.Any], ...]:
    """
    Calculate basic image statistics.

    Parameters
    ----------
    image : numpy.ndarray
        Image values to summarize.

    Returns
    -------
    tuple of numpy.floating
        Minimum, maximum, mean, and standard deviation, in that order.
    """
    return np.min(image), np.max(image), np.mean(image), np.std(image)


def set_std_tickparams(axis: typing.Any, tsz: float) -> None:
    """
    Apply the standard PyLDT tick formatting to a plot axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis whose tick parameters are changed.
    tsz : float
        Tick-label font size.
    """
    axis.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=tsz,
    )
