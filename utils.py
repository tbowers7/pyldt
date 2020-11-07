# -*- coding: utf-8 -*-
#
#  This file is part of PyLDT.
#
#  PyLDT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  any later version.
#
#  PyLDT is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with PyLDT.  If not, see <https://www.gnu.org/licenses/>.
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
