# -*- coding: utf-8 -*-
#
#  This file is part of PyLDT.
#
#   This Source Code Form is subject to the terms of the Mozilla Public
#   License, v. 2.0. If a copy of the MPL was not distributed with this
#   file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 26-Jan-2021
#
#  @author: tbowers

"""PyLDT contains image calibration routines for LDT facility instruments

Lowell Discovery Telescope (Lowell Observatory: Flagstaff, AZ)
http://www.lowell.edu

This file contains routines for extracting spectra from 2D spectrographic
images.  They are intended to operate in a manner similar to their IRAF
namesakes.
"""

# Built-In Libraries
from __future__ import division, print_function, absolute_import
from datetime import datetime
import glob
import os
from pathlib import Path
import shutil
import sys
import warnings

# Numpy & Similar
import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrix
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
import statsmodels.api as sm

# Astropy and CCDPROC
from astropy.io import fits
from astropy.modeling import models
from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy.utils.exceptions import AstropyWarning
import ccdproc as ccdp
from ccdproc.utils.slices import slice_from_string

from rascal.calibrator import Calibrator
from rascal.util import load_calibration_lines


# Intrapackage
from .utils import *

# Boilerplate variables
__author__ = 'Timothy P. Ellsworth Bowers'
__copyright__ = 'Copyright 2020'
__credits__ = ['Lowell Observatory']
__license__ = 'MPL-2.0'
__version__ = '0.2.0'
__email__ = 'tbowers@lowell.edu'
__status__ = 'Development Status :: 2 - Pre-Alpha'


def twodspec_response():
    """Do something like IRAF's twodspec.longlist.response()
    """
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
        
def twodspec_apextract():
    ## Suppress the warning:
    ##    WARNING: FITSFixedWarning: RADECSYS= 'FK5 ' / Astrometric System
    ##    the RADECSYS keyword is deprecated, use RADESYSa. [astropy.wcs.wcs]
    warnings.simplefilter('ignore', AstropyWarning)
    warnings.simplefilter('ignore', UserWarning)


    if len(sys.argv) < 2:
        print("Script requires a filename to extract.")
        sys.exit()

    file_name = sys.argv[1]
        
    if not os.path.isfile(file_name):
        print("File either does not exist or cannot be read.")
        sys.exit()

    ccd = CCDData.read(file_name)

    spec = ccdp.block_average(ccd[200:350], [150, 1])
    print(spec.ndim)


    print(spec.shape, ccd.shape)

    pixnum = np.asarray(range(ccd.shape[1])).flatten()  # Running pixel number

    print(type(spec),type(pixnum))
    print(spec.shape, pixnum.shape)


    ## Figure!
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(pixnum,np.transpose(spec))
    #ax.plot(pixnum,flat_fit,'r-')
    ax.set_ylim(ymin=0)
    plt.title(file_name)
    plt.show()


    oned_fn = '{0}_1d{1}'.format(file_name[:-5],file_name[-5:])

    spec.write(oned_fn, overwrite=True)
    os.remove(file_name)


def twodspec_apdefine():
    pass


def twodspec_identify():
    # Open the example file
    spectrum2D = fits.open("20201003.0038b_comb_1d.fits")[0].data
    npix = spectrum2D.shape[1]

    # Get the median along the spectral direction
    spectrum = np.median(spectrum2D, axis=0)

    print(spectrum.shape)

    # Load the Lines from library
    atlas = load_calibration_lines(elements=["Ne","Ar","Hg","Cd"])

    #print(atlas)

    # Get the spectral lines
    peaks, _ = find_peaks(spectrum)

    print(peaks)
    print(peaks[1:] - peaks[:-1])

    print(type(peaks),type(atlas))


    pixnum = np.arange(npix)  # Running pixel number
    ## Figure!
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(pixnum,np.transpose(spectrum))
    #ax.plot(pixnum,flat_fit,'r-')
    ax.set_ylim(ymin=0)
    #plt.title(file_name)
    plt.show()






    # Set up the Calibrator object
    c = Calibrator(peaks, npix)

    # Solve for the wavelength calibration
    best_p = c.fit()

    # Produce the diagnostic plot
    c.plot_fit(spectrum, best_p)




def twodspec_reidentify():
    pass