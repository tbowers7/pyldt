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
__copyright__ = 'Copyright 2021'
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


def spline3(xs, ys, order=2, debug=True):
    """spline3 computes a cubic spline of specified order to fit input data

    :param xs: Array of input abscissa values
    :param ys: Array of input ordinate values
    :param order: Order of the cublic spline to fit [Default: 2]
    :param debug: Print debugging statements [Default: True]
    :return: Returns the fitted ordinate values to the input data
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
        print(f"The knot string is: >>{knot_str}<<")

    # Fit a natural spline with specified knots
    x_bs = dmatrix(f'bs(x, knots={knot_str})', {'x': xs})
    fit_bs = sm.GLM(ys, x_bs).fit()
 
    x_cr = dmatrix(f'cr(x, knots={knot_str})', {'x': xs})
    fit_cr = sm.GLM(ys, x_cr).fit()

    # Create spline at xs
    fit1 = fit_bs.predict(dmatrix(f'bs(xs, knots={knot_str})',{'xs': xs}))
    fit2 = fit_cr.predict(dmatrix(f'cr(xs, knots={knot_str})',{'xs': xs}))

    # if debug:
    #     print(f"Type of x_natural: {type(x_natural)}")
    #     print(f"Type of fit_natural: {type(fit_natural)}")
    #     print(f"Type of fit: {type(fit)}")

    return fit1, fit2, knots
