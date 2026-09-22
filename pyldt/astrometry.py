# SPDX-License-Identifier: MPL-2.0
#  Created on 26-Oct-2020
#  @author: tbowers
"""
PyLDT contains image calibration routines for LDT facility instruments

Lowell Discovery Telescope (Lowell Observatory: Flagstaff, AZ)
http://www.lowell.edu

This module provides a wrapper for solving the plate scale of LMI images using
Astrometry.Net
"""

from __future__ import annotations

# Astropy exposes units and HDU members dynamically.
# pylint: disable=no-member

# Built-In Libraries
import pathlib
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
from pyldt.calibration import PKG_NAME, savetime, write_ccd_atomic

# Define API
__all__ = ["solve_field", "validate_solution"]


def solve_field(
    img_fn: str | pathlib.Path,
    *,
    detect_threshold: float = 10,
    fwhm: float = 3,
    plate_scale: u.Quantity | float | None = None,
    plate_error: float = 10,
    force_image_upload: bool = False,
    validate: bool = True,
    add_scale: bool = False,
    add_center_coords: bool = False,
    max_attempts: int = 5,
    retry_delay: float = 30,
    solve_timeout: float = 120,
    debug: bool = False,
) -> tuple[astropy.wcs.WCS, bool]:
    """
    Get a plate solution from Astrometry.Net

    Plate solutions not only provide accurate astrometry of objects in an
    image, they can also help to identify distortions or rotations in the
    image not already described in the FITS header.

    If an estimated plate scale is given, that is passed to Astrometry.Net
    with ±``plate_error``% bounds to speed up the solution.

    Parameters
    ----------
    img_fn : :obj:`str` or :obj:`pathlib.Path`
        Filename of the image on which to do a plate solution
    detect_threshold : float, optional
        Detection limit as a number of standard deviations above background.
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
    max_attempts : int, optional
        Maximum number of submission or monitoring attempts before propagating
        the last network error. (Default: 5)
    retry_delay : float, optional
        Seconds to wait between failed network attempts. (Default: 30)
    solve_timeout : float, optional
        Timeout passed when monitoring an existing submission. (Default: 120)
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
                plate_scale = plate_scale.to(u.arcsec / u.pix)
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

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one.")
    if retry_delay < 0:
        raise ValueError("retry_delay cannot be negative.")
    if plate_error < 0:
        raise ValueError("plate_error cannot be negative.")

    # Loop variables
    submission_id = None
    wcs_header = None

    # Retry transient service failures, but never wait forever.
    for attempt in range(max_attempts):
        try:
            if not submission_id:
                # Find objects in the image and send the list to Astrometry.Net
                wcs_header = ast.solve_from_image(
                    img_fn,
                    submission_id=submission_id,
                    detect_threshold=detect_threshold,
                    scale_units=scale_units,
                    scale_est=getattr(plate_scale, "value", None),
                    scale_lower=getattr(scale_lower, "value", None),
                    scale_upper=getattr(scale_upper, "value", None),
                    publicly_visible="n",
                    allow_commercial_use="n",
                    fwhm=fwhm,
                    force_image_upload=force_image_upload,
                )
            else:
                # Subsequent times through the loop, check on the submission
                wcs_header = ast.monitor_submission(
                    submission_id, solve_timeout=solve_timeout
                )
        except astroquery.exceptions.TimeoutError as error:
            if len(error.args) > 1:
                submission_id = error.args[1]
            if attempt == max_attempts - 1:
                raise
        except (
            ConnectionError,
            requests.exceptions.ConnectionError,
            requests.exceptions.JSONDecodeError,
            requests.exceptions.ReadTimeout,
        ):
            if attempt == max_attempts - 1:
                raise
        else:
            break
        if retry_delay:
            time.sleep(retry_delay)
    if wcs_header is None:
        raise RuntimeError("Astrometry.Net returned no WCS solution.")
    print("done.")

    # Instantiate a WCS object from the wcs header returned by Astronmetry.Net
    solved_wcs = astropy.wcs.WCS(wcs_header)

    # Similarly, instantiate a WCS object from the original file
    with astropy.io.fits.open(img_fn) as hdulist:
        original_header = hdulist[0].header.copy()
        existing_wcs = astropy.wcs.WCS(original_header)

    # Read in the FITS file to a CCDData object, applying BUNIT as necessary
    bunit = original_header.get("bunit", None)
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
            comments = wcs_header["COMMENT"]
            if isinstance(comments, str):
                comments = [comments]
            scale_str = next(
                (comment for comment in comments if comment.startswith("scale:")),
                None,
            )
            if scale_str is not None:
                solved_scale = float(scale_str.split()[1])
                ccd.header["SCALE"] = np.round(solved_scale, 3)
        except (KeyError, IndexError, ValueError):
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
    ccd.header["HISTORY"] = PKG_NAME
    ccd.header["HISTORY"] = "Plate solution performed via astroquery.astrometry_net"
    ccd.header["HISTORY"] = "Solved WCS added: " + savetime()

    if debug:
        # Print out the final header before writing to disk
        print(f"\n{ccd.header}")

    # Write the CCDData object to disk with the updated WCS information
    write_ccd_atomic(ccd, img_fn)

    return use_wcs, is_solved


def validate_solution(
    solved: astropy.wcs.WCS,
    lois: astropy.wcs.WCS,
    rtol: float = 1e-05,
    atol: float = 3e-07,
    max_center_separation: u.Quantity | float = 1 * u.deg,
    debug: bool = False,
) -> tuple[astropy.wcs.WCS, bool]:
    """
    Validate the Astrometry.Net plate solution

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
    max_center_separation : `astropy.units.Quantity` or `float`, optional
        Maximum separation between the two WCS reference sky positions. A
        unitless value is interpreted as degrees. (Default: 1 degree)
    debug : `bool`, optional
        Print debugging statements?  [Default: False]

    Returns
    -------
    wcs : `astropy.wcs.WCS`
        The WCS to use with this frame
    is_close : `bool`
        Whether the solved WCS is close to the lois default
    """
    scale_is_close = np.allclose(
        solved.pixel_scale_matrix, lois.pixel_scale_matrix, rtol=rtol, atol=atol
    )
    separation_limit = u.Quantity(max_center_separation, u.deg)
    if separation_limit <= 0 * u.deg:
        raise ValueError("max_center_separation must be positive.")
    try:
        solved_center = astropy.coordinates.SkyCoord(
            *solved.celestial.wcs.crval, unit=u.deg
        )
        lois_center = astropy.coordinates.SkyCoord(
            *lois.celestial.wcs.crval, unit=u.deg
        )
        center_is_close = solved_center.separation(lois_center) <= separation_limit
    except (ValueError, IndexError):
        center_is_close = False
    is_close = bool(scale_is_close and center_is_close)

    print(f"\nThe Astrometry.Net solution ≈ the lois default:   {is_close}")
    if debug:
        print(f"Solved:\n{solved.pixel_scale_matrix * 3600}")
        print(f"Lois:\n{lois.pixel_scale_matrix * 3600}")

    return (solved, is_close) if is_close else (lois, is_close)
