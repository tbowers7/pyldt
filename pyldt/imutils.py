# SPDX-License-Identifier: MPL-2.0
#  Created on 26-Oct-2020
#  @author: tbowers
"""
PyLDT contains image calibration routines for LDT facility instruments

Lowell Discovery Telescope (Lowell Observatory: Flagstaff, AZ)
http://www.lowell.edu

This module contains various image utility routines.
"""

from __future__ import annotations

# Built-In Libraries
import pathlib
import typing

# 3rd Party Libraries
import astropy.nddata
import astropy.io.fits
import numpy as np

# Internal-ish Imports
from obstools import deveny_grangle


def make_flat_as_star(
    flatfn: str | pathlib.Path | astropy.nddata.CCDData,
    biasfn: str | pathlib.Path,
    outseq: int,
    copyfn: str | pathlib.Path | astropy.nddata.CCDData | None = None,
    verbose: bool = True,
    objname: str | None = None,
) -> None:
    """
    Make a DeVeny flat-field frame resemble a stellar spectrum.

    This function creates an output image that mimics a stellar spectrum
    except the data is from a flatfield.  The purpose of this is for use
    with PypeIt to generate an "object" spectrum of the flatfield.

    Parameters
    ----------
    flatfn : path-like or astropy.nddata.CCDData
        Flat-field image to convert.
    biasfn : path-like
        Bias frame from the same night.
    outseq : int
        File sequence number for the output image.
    copyfn : path-like or astropy.nddata.CCDData, optional
        Frame whose header is copied. The flat frame is used when omitted.
    verbose : bool, optional
        Print processing details.
    objname : str, optional
        Object name inserted into the output header.
    """

    if copyfn is None:
        copyfn = flatfn

    bias = astropy.nddata.CCDData.read(biasfn)
    flat = (
        flatfn
        if isinstance(flatfn, astropy.nddata.CCDData)
        else astropy.nddata.CCDData.read(flatfn)
    )
    copy = (
        copyfn
        if isinstance(copyfn, astropy.nddata.CCDData)
        else astropy.nddata.CCDData.read(copyfn)
    )

    if objname is None:
        objname = "FlatFieldAsStar"

    if verbose:
        print(f"COPY: {copy.shape}")
        print(f"BIAS: {bias.shape}")
        print(f"FLAT: {flat.shape}")
        print(f"Objname: {objname}")

    # Update the header of the copy CCDData object
    copy.header["obstype"] = "OBJECT"
    copy.header["imagetyp"] = "OBJECT"
    copy.header["filename"] = (
        f"{flat.header['filename'].split('.')[0]}.{outseq:04d}.fits"
    )
    copy.header["objname"] = objname
    copy.header["object"] = objname
    copy.header["scitarg"] = objname
    copy.header["exptime"] = flat.header["exptime"]
    copy.header["date-obs"] = flat.header["date-obs"]
    copy.header["utcstart"] = flat.header["utcstart"]
    copy.header["ut"] = flat.header["ut"]
    copy.header["utcend"] = flat.header["utcend"]
    copy.header["lst-obs"] = flat.header["lst-obs"]
    copy.header["st"] = flat.header["st"]

    if verbose:
        print(
            f"Stats on input flat... median: {np.median(flat.data)}, max: {np.max(flat.data)}"
        )

    # Start with the bais for the data
    copy.data = bias.data

    # Define the strip for use here:
    ymin, ymax = (305, 320)

    # Look at the middle of this strip in the bias for a mean floor
    base = np.median(bias[ymin:ymax, 500:1500])

    # Cut out a strip from the flat -- remove base, and divide by 10
    strip = (flat[ymin:ymax, :] - base) / 10.0
    if verbose:
        print(f"Shape of the strip: {strip.shape}")
        print(f"Base level for the bias: {base}")
        print(f"Median level of the strip: {np.median(strip)}")

    # Make a Gaussian in y to apply to the strip to make it look stellar
    y_arr = np.arange(15)
    g_arr = np.exp(-((y_arr - 7) ** 2) / 6)
    if verbose:
        pass  # print(f"Gaussian g: {g}")

    # Apply the gaussian to the strip
    strip2 = strip * g_arr.reshape(len(g_arr), 1)

    # Put the gaussian-ed strip into the copy
    copy.data[ymin:ymax, :] = strip2 + base

    # Write the thing to the outfile
    copy.write(f"{copy.header['filename'].split('/')[-1]}", overwrite=True)


def load_pypeit_flat(
    filename: str | pathlib.Path,
    lcen: float | None = None,
    gpmm: float | int | None = None,
) -> dict[str, typing.Any]:
    """
    Load a PypeIt flat calibration into a dictionary.

    Data analysis / debugging function

    Parameters
    ----------
    filename : `str`
        Filename of the Flat file to read in
    lcen : `float`, optional
        Centeral wavelength of the grating setup [Default: None]
    gpmm : `float` or `int`, optional
        Lines per mm on the grating installed [Default: None]

    Returns
    -------
    `dict`
        Dictionary containing the various Flat products for ease of use
    """
    with astropy.io.fits.open(filename) as hdul:
        flat_dict = {}
        for hdu in hdul:
            if "EXTNAME" in hdu.header:
                flat_dict[hdu.header["EXTNAME"]] = hdu.data

        if lcen and gpmm:
            grangle, _ = deveny_grangle.compute_grangle(gpmm, lcen)
            flat_dict["GRANGLE"] = grangle

    return flat_dict


def load_pypeit_2dspec() -> None:
    """
    Load a PypeIt two-dimensional spectrum.

    Notes
    -----
    This placeholder currently performs no operation.
    """


def load_pypeit_1dspec() -> None:
    """
    Load a PypeIt one-dimensional spectrum.

    Notes
    -----
    This placeholder currently performs no operation.
    """
