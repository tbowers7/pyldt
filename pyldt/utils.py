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


def mmms(image):
    """mmms performs basic statistics on an image (min, max, mean, stddev)

    :param image:
    :return:
    """
    return np.min(image), np.max(image), np.mean(image), np.std(image)
