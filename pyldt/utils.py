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
from patsy import dmatrix, PatsyError
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


def spline3(xs, ys, order=4, debug=False, ftype='bs', bc='nak'):
    """spline3 computes a cubic spline of specified order to fit input data

    :param xs: Array of input abscissa values
    :param ys: Array of input ordinate values
    :param order: Order of the cublic spline to fit [Default: 4]
    :param debug: Print debugging statements [Default: False]
    :param ftype: Type of spline [Default: 'bs': 'B-Spline']
    :param bc: Boundary Condition [Default: 'nak' = Not-a-Knot]
    :return: Returns the fitted ordinate values, knot position, and errmsg
    """

    errmsg = ''

    # Define the knots for the cubic spline: evenly sized segments
    knots = np.asarray(range(order))[1:] / order * (xs[-1] + 1)

     # For the "not-a-knot" boundary condition, remove first and last knots:
    if bc == 'nak':
        knots = knots[1:-1]
    
    # Select the type of cublic spline fit:
    if ftype == 'bs':
        # B-Spline
 
        # Create the string of knot values to pass to dmatrix (from patsy)
        knot_str = '('
        for kn in knots:
            knot_str += f'{kn},'
        knot_str = knot_str[:-1]+')'
    
        # Print statement
        if debug:
            print(f"The knot string is: >>{knot_str}<<")

        # Fit the B-Spline with specified knots
        x_bs = dmatrix(f'bs(x, knots={knot_str}, degree=3)', {'x': xs})
        
        # Fit Generalized Linear Model on the transformed dataset
        fit_bs = sm.GLM(ys, x_bs).fit()

        # Create the spline along the spectrum
        fit = fit_bs.predict(x_bs)

    elif ftype == 'cr':
        # Natural cublic spline (cubic regression)

        # Fit a natural spline with degrees of freedom = K = order-1
        try:
            x_cr = dmatrix(f'cr(x, df={order-1})', {'x': xs})

            # Fit Generalized Linear Model on transformed dataset
            fit_cr = sm.GLM(ys, x_cr).fit()  

            # Create the spline along the spectrum
            fit = fit_cr.predict(dmatrix(f'cr(x, df={order-1})',{'x': xs}))

        except PatsyError:
            print("Order too low.  Cannot compute.")
            errmsg = "Selected order is too small for fit."
            fit = None
        
    else:
        print(f"Did something wrong, dude.  " + \
            f"Don't recognize the fit type '{ftype}'")
        errmsg = "Wrong fit type passed to spline3"
        fit = None

    return fit, knots, errmsg
