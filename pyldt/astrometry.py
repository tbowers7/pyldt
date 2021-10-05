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
from astroquery.exceptions import TimeoutError as aqe_TimeoutError

# Intrapackage
from .reduction import _savetime, PKG_NAME


def astrometry_net(img_fn, debug=False):
    """astrometry_net Get a plate solution from Astrometry.Net

    Plate solutions not only provide accurate astrometry of objects in an
    image, they can also help to identify distortions or rotations in the
    image not already described in the FITS header.

    Parameters
    ----------
    img_fn : `str`
        Filename of the image on which to do a plate solution
    debug : `bool`, optional
        Print debugging statements? [Default: False]
    """
    # Instantiate the Astrometry.Net communicator
    ast = AstrometryNet()
    ast.api_key = "advyolesvrcrexse"

    # Loop variables
    try_again = True
    submission_id = None

    # Loop until a solution is returned
    while try_again:
        try:
            if not submission_id:
                # Find objects in the image and send the list to Astrometry.Net
                wcs_header = ast.solve_from_image(img_fn,
                                            submission_id=submission_id)
            else:
                # Subsequent times through the loop, check on the submission
                wcs_header = ast.monitor_submission(submission_id,
                                                    solve_timeout=120)
        except aqe_TimeoutError as error:
            submission_id = error.args[1]
        except ConnectionError:
            pass
        else:
            # Got a result: Terminate
            try_again = False

    # Instantiate a WCS object from the wcs header returned by Astronmetry.Net
    solved_wcs = WCS(wcs_header)

    # Read in the FITS file to a CCDData object
    ccd = CCDData.read(img_fn)

    if debug:
        # If desired, print a bunch of diagnostics
        print(f"\n{ccd.wcs}")
        print(f"\n{wcs_header}")
        print(f"\n{solved_wcs}")

    # Place the WCS object into the .wcs attribute of the CCDData object
    ccd.wcs = solved_wcs

    # For good measure, also attempt to update the header with the WCS object
    ccd.header.update(solved_wcs.to_header())

    # Add some history information
    ccd.header['HISTORY'] = PKG_NAME
    ccd.header['HISTORY'] = 'Plate solution performed via astroquery.astrometry_net'
    ccd.header['HISTORY'] = 'Solved WCS added: ' + _savetime()

    if debug:
        # Print out the final header before writing to disk
        print(f"\n{ccd.header}")

    # Write the CCDData object to disk with the updated WCS information
    ccd.write(img_fn, overwrite=True)
