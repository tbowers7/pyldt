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
import time

# 3rd Party Libraries
import astropy.coordinates
import astropy.io.fits
import astropy.nddata
import astropy.units as u
import astropy.wcs
import astroquery.astrometry_net
import astroquery.exceptions
import numpy as np
import requests.exceptions

# Internal Imports
from pyldt import reduction


# Define API
__all__ = ["solve_field", "validate_solution"]


def solve_field(
    img_fn,
    *,
    detect_threshold=10,
    fwhm=3,
    plate_scale=None,
    plate_error=10,
    force_image_upload=False,
    validate=True,
    add_scale=False,
    add_center_coords=False,
    debug=False,
):
    """Get a plate solution from Astrometry.Net

    Plate solutions not only provide accurate astrometry of objects in an
    image, they can also help to identify distortions or rotations in the
    image not already described in the FITS header.

    If an estimated plate scale is given, that is passed to Astrometry.Net
    with ±``plate_error``% bounds to speed up the solution.

    Parameters
    ----------
    img_fn : :obj:`str` or :obj:`pathlib.Path`
        Filename of the image on which to do a plate solution
    detect_threshold: float, optional
        Detection limit, as # of sigma above background
    fwhm : float, optional
        FWHM of detected objects, in pixels
    plate_scale : :obj:`astropy.units.Quantity` or :obj:`float`, optional
        The estimated plate scale of the image, to be passed to Astrometry.Net
        for more quickly narrowing the solution parameters.  If ``plate_scale``
        is a :obj:`astropy.units.Quantity`, awesome.  If not, then the value
        will be assumed to be in arcsec/pix.  (Default: None)
    plate_error : float, optional
        Percentage error allowed on the plate scale for solution (Default: 10%)
    force_image_upload : bool, optional
        Pass-through option to ``astroquery`` on whether or not to force an
        image upload rather than find sources locally.  (Default: False)
    validate : bool, optional
        Validate the solved WCS against the included WCS (likely from lois)?
        (Default: True)
    add_scale : bool, optional
        Add the SCALE keyword to the FITS header from the Astrometry.Net
        solution?  (Default: False)
    add_center_coords : bool, optional
        Add RA/DEC keywords to the FITS header corresponding to the center of
        the image using the solution from the Astrometry.Net?  (Default: False)
    debug : bool, optional
        Print debugging statements? (Default: False)

    Returns
    -------
    :obj:`astropy.wcs.WCS`
        The resultant WCS from the solving process
    is_solved : bool
        Whether the returned WCS is the Astrometry.Net solution or not
    """
    # Instantiate the Astrometry.Net communicator
    ast = astroquery.astrometry_net.AstrometryNet()

    scale_lower = None
    scale_upper = None
    scale_units = None

    # If estimated plate scale is passed, generate submission bounds
    if plate_scale is not None:
        # Check if `plate_scale` is a Quantity:
        if isinstance(plate_scale, u.Quantity):
            try:
                plate_scale <<= u.arcsec / u.pix
                scale_lower = plate_scale * (1 - plate_error / 100)
                scale_upper = plate_scale * (1 + plate_error / 100)
                scale_units = "arcsecperpix"
            except u.core.UnitConversionError:
                # If the input cannot be converted to "/pix, set to None
                plate_scale = None
        elif isinstance(plate_scale, (float, int)):
            # If it's a float, assume arcsec/pix
            plate_scale = u.Quantity(plate_scale, u.arcsec / u.pix)
            scale_lower = plate_scale * (1 - plate_error / 100)
            scale_upper = plate_scale * (1 + plate_error / 100)
            scale_units = "arcsecperpix"
        else:
            plate_scale = None

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
                    scale_units=scale_units,
                    scale_est=plate_scale.value,
                    scale_lower=scale_lower.value,
                    scale_upper=scale_upper.value,
                    publicly_visible="n",
                    allow_commercial_use="n",
                    fwhm=fwhm,
                    force_image_upload=force_image_upload,
                )
            else:
                # Subsequent times through the loop, check on the submission
                wcs_header = ast.monitor_submission(submission_id, solve_timeout=120)
        except astroquery.exceptions.TimeoutError as error:
            submission_id = error.args[1]
        except (ConnectionError, requests.exceptions.JSONDecodeError):
            pass
        except requests.exceptions.ReadTimeout:
            # Wait 30 seconds and try again
            time.sleep(30)
        else:
            # Got a result: Terminate
            try_again = False
    print("done.")

    # Instantiate a WCS object from the wcs header returned by Astronmetry.Net
    solved_wcs = astropy.wcs.WCS(wcs_header)

    # Similarly, instantiate a WCS object from the original file
    with astropy.io.fits.open(img_fn) as hdulist:
        existing_wcs = astropy.wcs.WCS(hdulist[0].header)

    # Read in the FITS file to a CCDData object, applying BUNIT as necessary
    bunit = hdulist[0].header.get("bunit", None)
    ccd = astropy.nddata.CCDData.read(img_fn, unit="adu" if bunit is None else None)

    # Validate the solved WCS against the lois-written WCS
    #  If the solution is way off, just keep the lois WCS
    if validate and ccd.wcs is not None:
        use_wcs, is_solved = validate_solution(solved_wcs, existing_wcs, debug=debug)
    else:
        use_wcs, is_solved = solved_wcs, True

    if debug:
        # If desired, print a bunch of diagnostics
        print(f"\nccd.wcs:\n{ccd.wcs}")
        print(f"\nwcs_header:\n{wcs_header}")
        print(f"\nsolved_wcs:\n{use_wcs}")

    # Place the WCS object into the .wcs attribute of the CCDData object
    ccd.wcs = use_wcs

    # For good measure, also attempt to update the header with the WCS object
    ccd.header.update(use_wcs.to_header(relax=True))

    # If `add_scale`, add it:
    if add_scale:
        try:
            scale_str = next(
                (x for x in wcs_header["COMMENT"] if x.startswith("scale:")), None
            )
            try:
                solved_scale = float(scale_str.split()[1])
            except TypeError:
                solved_scale = -1.0
            ccd.header["SCALE"] = np.round(solved_scale, 3)
        except KeyError:
            # Bad solution, "COMMENT"s not included in returned header
            pass

    # If `add_center_coords`, add them:
    if add_center_coords:
        center = use_wcs.pixel_to_world(
            ccd.header["NAXIS1"] // 2, ccd.header["NAXIS2"] // 2
        )
        if isinstance(center, astropy.coordinates.SkyCoord):
            ra, dec = center.to_string(style="hmsdms", precision=1).split()
            ccd.header["RA"] = ra.replace("h", ":").replace("m", ":").replace("s", "")
            ccd.header["DEC"] = dec.replace("d", ":").replace("m", ":").replace("s", "")

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


def validate_solution(
    solved: astropy.wcs.WCS,
    lois: astropy.wcs.WCS,
    rtol: float = 1e-05,
    atol: float = 3e-07,
    debug: bool = False,
) -> tuple[astropy.wcs.WCS, bool]:
    """Validate the Astrometry.Net plate solution

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
