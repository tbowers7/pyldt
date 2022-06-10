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
import astropy.io.fits
import astropy.nddata
import astropy.wcs
import astroquery.astrometry_net
import astroquery.exceptions
import numpy as np
import requests.exceptions

# Internal Imports
from pyldt import reduction


# Define API
__all__ = ["solve_field", "validate_solution"]


def solve_field(img_fn, detect_threshold=10, debug=False):
    """solve_field Get a plate solution from Astrometry.Net

    Plate solutions not only provide accurate astrometry of objects in an
    image, they can also help to identify distortions or rotations in the
    image not already described in the FITS header.

    Parameters
    ----------
    img_fn : `str` or `pathlib.Path`
        Filename of the image on which to do a plate solution
    detect_threshold, `float`, optional
        Detection limit, as # of sigma above background
    debug : `bool`, optional
        Print debugging statements? [Default: False]

    Returns
    -------
    `astropy.wcs.WCS`
        The resultant WCS from the solving process
    is_solved : `bool`
        Whether the returned WCS is the Astrometry.Net solution or not
    """
    # Instantiate the Astrometry.Net communicator
    ast = astroquery.astrometry_net.AstrometryNet()

    # Loop variables
    try_again = True
    submission_id = None

    # Loop until a solution is returned
    while try_again:
        try:
            if not submission_id:
                # Find objects in the image and send the list to Astrometry.Net
                wcs_header = ast.solve_from_image(
                    img_fn,
                    submission_id=submission_id,
                    detect_threshold=detect_threshold,
                )
            else:
                # Subsequent times through the loop, check on the submission
                wcs_header = ast.monitor_submission(submission_id, solve_timeout=120)
        except astroquery.exceptions.TimeoutError as error:
            submission_id = error.args[1]
        except (ConnectionError, requests.exceptions.JSONDecodeError):
            pass
        else:
            # Got a result: Terminate
            try_again = False
    print("done.")

    # Instantiate a WCS object from the wcs header returned by Astronmetry.Net
    solved_wcs = astropy.wcs.WCS(wcs_header)

    # Similarly, instantiate a WCS object from the original file
    with astropy.io.fits.open(img_fn) as hdulist:
        existing_wcs = astropy.wcs.WCS(hdulist[0].header)

    # Read in the FITS file to a CCDData object
    ccd = astropy.nddata.CCDData.read(img_fn)

    # Validate the solved WCS against the lois-written WCS
    #  If the solution is way off, just keep the lois WCS
    use_wcs, is_solved = validate_solution(solved_wcs, existing_wcs)

    if debug:
        # If desired, print a bunch of diagnostics
        print(f"\nccd.wcs:\n{ccd.wcs}")
        print(f"\nwcs_header:\n{wcs_header}")
        print(f"\nsolved_wcs:\n{use_wcs}")

    # Place the WCS object into the .wcs attribute of the CCDData object
    ccd.wcs = use_wcs

    # For good measure, also attempt to update the header with the WCS object
    ccd.header.update(use_wcs.to_header())

    # Add some history information
    ccd.header["HISTORY"] = reduction.PKG_NAME
    ccd.header["HISTORY"] = "Plate solution performed via astroquery.astrometry_net"
    ccd.header["HISTORY"] = "Solved WCS added: " + reduction.savetime()

    if debug:
        # Print out the final header before writing to disk
        print(f"\n{ccd.header}")

    # Write the CCDData object to disk with the updated WCS information
    ccd.write(img_fn, overwrite=True)

    return use_wcs, is_solved


def validate_solution(solved, lois, rtol=1e-05, atol=3e-07, debug=False):
    """validate_solution Validate the Astrometry.Net plate solution

    If the Astrometry.Net solution is way off, keep the original WCS.
    Otherwise, use the new solution.

    Parameters
    ----------
    solved : `astropy.wcs.WCS`
        The Astrometry.Net-solved WCS
    lois : `astropy.wcs.WCS`
        The original WCS from the image header
    rtol : `float`, optional
        Relative tolerance, passed to np.allclose()  [Default: 1e-05]
    atol : `float`, optional
        Absolute tolerance, passed to np.allclose()  [Default: 3e-07]
    debug : `bool`, optional
        Print debugging statements?  [Default: False]

    Returns
    -------
    wcs : `astropy.wcs.WCS`
        The WCS to use with this frame
    is_close : `bool`
        Whether the solved WCS is close to the lois default
    """
    # Ask numpy!
    is_close = np.allclose(
        solved.pixel_scale_matrix, lois.pixel_scale_matrix, rtol=rtol, atol=atol
    )

    print(f"\nThe Astrometry.Net solution ≈ the lois default:   {is_close}")
    if debug:
        print(f"Solved:\n{solved.pixel_scale_matrix * 3600}")
        print(f"Lois:\n{lois.pixel_scale_matrix * 3600}")

    return (solved, is_close) if is_close else (lois, is_close)
