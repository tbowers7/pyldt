# -*- coding: utf-8 -*-
#
#  This file is part of PyLDT.
#
#   This Source Code Form is subject to the terms of the Mozilla Public
#   License, v. 2.0. If a copy of the MPL was not distributed with this
#   file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 26-Oct-2020
#
#  @author: tbowers

"""PyLDT contains image calibration routines for LDT facility instruments

Lowell Discovery Telescope (Lowell Observatory: Flagstaff, AZ)
http://www.lowell.edu

Utility files for use with PyLDT

There are some utility functions here...
"""

# Built-In Libraries

# Third-Party Libraries
import numpy as np

# Internal Imports


# Define API
__all__ = ["mmms", "set_std_tickparams"]


def mmms(image):
    """mmms performs basic statistics on an image (min, max, mean, stddev)

    :param image:
    :return:
    """
    return np.min(image), np.max(image), np.mean(image), np.std(image)


def set_std_tickparams(axis, tsz):
    """set_std_tickparams _summary_
    _extended_summary_
    Parameters
    ----------
    axis : `matplotlib.pyplot.axis`
        PyPlot axis who whom the tick parameters must be set
    tsz : `int` or `float`
        TypeSiZe
    """
    axis.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=tsz,
    )
