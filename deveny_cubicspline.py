

## Numpy
import numpy as np

from patsy import dmatrix
import statsmodels.api as sm




def spline3(xs, ys, order):


    # Define the knots for the cubic spline: evenly sized segments
    knots = np.asarray(range(order))[1:] / order * (xs[-1]+1)

    # Create the string of these values to pass to dmatrix (from patsy)
    knot_str = '('
    for kn in knots:
        if kn != knots[-1]:
            knot_str += '{},'.format(kn)
        else:
            knot_str += '{})'.format(kn)

    print(knot_str)

    # Fit a natural spline with specified knots
    x_natural = dmatrix('cr(x, knots='+knot_str+')', {'x': xs})
    fit_natural = sm.GLM(ys, x_natural).fit()

    # Create spline at xs
    fit = fit_natural.predict(dmatrix('cr(xs, knots='+knot_str+')',
                                      {'xs': xs}))

    print(type(x_natural), type(fit_natural), type(fit))
    
    return fit
