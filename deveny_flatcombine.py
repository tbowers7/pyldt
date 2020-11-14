### Import the necessary packages

## Built-in Libraries
from pathlib import Path
import shutil
import warnings
import os

## Numpy
import numpy as np

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from patsy import dmatrix
import statsmodels.api as sm

## Astropy
from astropy.nddata import CCDData
from astropy import units as u
from astropy.modeling import models
from astropy.stats import mad_std
from astropy.utils.exceptions import AstropyWarning


## CCDPROC
import ccdproc as ccdp

## Local libraries



###========================================================================
### Define named things
BIN11 = '1 1'
ZEROFN = 'Zero.fits'
PREFIX = '20201003'


## Suppress the warning:
##    WARNING: FITSFixedWarning: RADECSYS= 'FK5 ' / Astrometric System
##    the RADECSYS keyword is deprecated, use RADESYSa. [astropy.wcs.wcs]
warnings.simplefilter('ignore', AstropyWarning)
warnings.simplefilter('ignore', UserWarning)


### After all that jazz...
flat_cl = ccdp.ImageFileCollection('.', glob_include='Flat*.fits')

for ccd, file_name in flat_cl.ccds(return_fname=True):
    
    
    ## We're going to collapse the flatfield image and fit a cubic spline to it
    nrows = ccd.shape[0]
    spec = np.sum(ccd, axis=0) / nrows
    
    pixnum = range(ccd.shape[1])  # Running pixel number
    order = 4
    knots = (range(order)/order)[1:]*ccd.shape[0]
    print(ccd.shape[0])
    print(knots)
    
    # Fit a natural spline with knots at ages 30, 50 and 70
    x_natural = dmatrix('cr(x, knots=(30, 50, 70))', {'x': pixnum})
    fit_natural = sm.GLM(spec, x_natural).fit()
    
    # Create spline lines for 50 evenly spaced values of age
    flat_fit = fit_natural.predict(dmatrix('cr(pixnum, knots=(30, 50, 70))', {'pixnum': pixnum}))

    
    
    
    ## Figure!
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(pixnum,spec)
    #ax.plot(pixnum,flat_fit,'r-')
    ax.set_ylim(ymin=0)
    plt.title(file_name)
    plt.show()
    
