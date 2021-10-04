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

This module provides a wrapper for solving the plate scale of LMI images using
Astrometry.Net 
"""

# Built-In Libraries
from datetime import datetime
import glob
import os
from pathlib import Path
import shutil
import warnings

# Numpy
import numpy as np

# Astropy and CCDPROC
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
from astroquery.astrometry_net import AstrometryNet
from astroquery.exceptions import TimeoutError
import ccdproc as ccdp
from ccdproc.utils.slices import slice_from_string

# Intrapackage
#rom .utils import *

# Boilerplate variables
__author__ = 'Timothy P. Ellsworth Bowers'
__copyright__ = 'Copyright 2021'
__credits__ = ['Lowell Observatory']
__license__ = 'MPL-2.0'
__version__ = '0.2.0'
__email__ = 'tbowers@lowell.edu'
__status__ = 'Development Status :: 3 - Alpha'

def astrometry_net(img_fn):

    ast = AstrometryNet()
    ast.api_key = "advyolesvrcrexse"

    # Loop variables
    try_again = True
    submission_id = None

    while try_again:
        try:
            if not submission_id:
                wcs_header = ast.solve_from_image(img_fn,
                                            submission_id=submission_id)
            else:
                wcs_header = ast.monitor_submission(submission_id,
                                                    solve_timeout=120)
        except TimeoutError as e:
            submission_id = e.args[1]
        else:
            # Got a result: Terminate
            try_again = False
    
    with fits.open(img_fn) as hdul:

        w = WCS(wcs_header)
        print(f"\n{w}")

        hdul[0].header.update(w.to_header())
        print(f"\n{hdul[0].header}")
        hdul.flush()
