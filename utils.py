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

"""Utility files for use with PyLDT

There are some utility functions here...
"""

# Built-In Libraries
from __future__ import division, print_function, absolute_import

# Third-Party Libraries
import numpy as np
from patsy import dmatrix
import statsmodels.api as sm

# Boilerplate variables
__author__ = 'Timothy P. Ellsworth Bowers'
__copyright__ = 'Copyright 2020'
__credits__ = ['Lowell Observatory']
__license__ = 'MPL-2.0'
__version__ = '0.2.0'
__email__ = 'tbowers@lowell.edu'
__status__ = 'Development Status :: 2 - Pre-Alpha'


def mmms(image):
    """mmms performs basic statistics on an image (min, max, mean, stddev)

    :param image:
    :return:
    """
    return np.min(image), np.max(image), np.mean(image), np.std(image)


def spline3(xs, ys, order, debug=True):
    """spline computes a cubic spline

    :param xs:
    :param ys:
    :param order:
    :param debug:
    :return:
    """
    # Define the knots for the cubic spline: evenly sized segments
    knots = np.asarray(range(order))[1:] / order * (xs[-1] + 1)

    # Create the string of these values to pass to dmatrix (from patsy)
    knot_str = '('
    for kn in knots:
        if kn != knots[-1]:
            knot_str += '{},'.format(kn)
        else:
            knot_str += '{})'.format(kn)

    if debug:
        print(knot_str)

    # Fit a natural spline with specified knots
    x_natural = dmatrix('cr(x, knots=' + knot_str + ')', {'x': xs})
    fit_natural = sm.GLM(ys, x_natural).fit()

    # Create spline at xs
    fit = fit_natural.predict(dmatrix('cr(xs, knots=' + knot_str + ')',
                                      {'xs': xs}))

    if debug:
        print(type(x_natural), type(fit_natural), type(fit))

    return fit
