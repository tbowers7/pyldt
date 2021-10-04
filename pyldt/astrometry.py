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

# 3rd Party Libraries
from astropy.nddata import CCDData
from astropy.wcs import WCS
from astroquery.astrometry_net import AstrometryNet
from astroquery.exceptions import TimeoutError

# Intrapackage


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
        except ConnectionError:
            pass
        else:
            # Got a result: Terminate
            try_again = False
    
    ccd = CCDData.read(img_fn)

    print(f"\n{wcs_header}")

    w = WCS(wcs_header)
    print(f"\n{w}")

    ccd.header.update(w.to_header())
    print(f"\n{ccd.header}")
    ccd.write(img_fn, overwrite=True)
