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

This module contains analysis of the flexure seen in the DeVeny Spectrograph

This file contains the main driver for the analysis.
Should be run in an environment containing:
    * AstroPy
    * CCDPROC
    * NumPy
    * Matplotlib
    * SciPy
"""

# Built-In Libraries
import os
import warnings

# Third-Party Libraries
import astropy.table
import ccdproc
import numpy as np
import scipy.optimize
import scipy.signal

# Internal Imports


def flexure_driver(data_dir, grating="DV2", save_fn="test.fits"):
    """flexure_driver Driving routine for the analysis

    [extended_summary]

    Parameters
    ----------
    data_dir : `str`
        Directory where the data live
    grating : `str`, optional
        Grating designation for analysis.  [Default: DV2]
    save_fn : `str`, optional
        Filename into which to save the table.  [Default: test.fits]

    Returns
    -------
    `astropy.table.table.Table`
        Table containing the stuffs.  [If no files found, returns None]
    """
    # Create an ImageFileCollection with files matching this grating;
    #  if empty, move along
    gcl = load_images(data_dir, grating)
    if not gcl.files:
        return None

    # AstroPy Table of line positions for each image in the IFC
    table = get_line_positions(gcl)

    # Go through the identified lines and produce a set found in all images
    table = validate_lines(table)

    # Add table columns for Delta away from 0º, and Delta away from mean
    table = compute_line_deltas(table)

    # Write the validated table to disk for future use (faster analysis)
    table.write(save_fn, overwrite=True)

    # Print out information on the table to the screen
    print(table.info)

    return table


def load_images(data_dir, grating="DV2"):
    """load_images Load in the images associated with DATA_DIR and grating

    [extended_summary]

    Parameters
    ----------
    data_dir : `str`
        The directory containing the data to analyze
    grating : `str`, optional
        The grating ID to use.  [Default: DV2]
    obstype : `str`, optional
        FITS header OBSTYPE to use.  Pass None for all.  [Default: comparison]

    Returns
    -------
    `ccdproc.image_collection.ImageFileCollection`
        IFC of the files meeting the input criteria
    """
    # Dictionary
    gratid = {"DV1": "150/5000", "DV2": "300/4000", "DV5": "500/5500"}

    # Load the images of interest in to an ImageFileCollection()
    icl = ccdproc.ImageFileCollection(data_dir)

    # Clean up the telescope altitude
    for ccd, fname in icl.ccds(obstype="comparison", return_fname=True):
        ccd.header["telalt"] = np.round(ccd.header["telalt"])
        ccd.write(os.path.join(data_dir, fname), overwrite=True)

    # Return an ImageFileCollection filtered by the grating desired
    return icl.filter(grating=gratid[grating])


def get_line_positions(icl, win=11, thresh=5000.0):
    """get_line_positions Compute the line positions for the images in the icl

    [extended_summary]

    Parameters
    ----------
    icl : `ccdproc.image_collection.ImageFileCollection`
        ImageFileCollection of images to work with
    win : `int`, optional
        Window (in pixels) across which to extract the spectrum, [Default: 11]
    thresh : `float`, optional
        Line intensity (ADU) threshold for detection, [Default: 5000.]

    Returns
    -------
    `astropy.table.table.Table`
        Table of line positions with associated metadata
    """
    # Put everything into a list of dicionaties
    flex_line_positions = []

    # This will only give the x values of the fits file.
    # For each of the images,
    for ccd, fname in icl.ccds(return_fname=True):
        # For ease
        hdr = ccd.header
        # Need a lower threshold for DV5 than for DV1
        if hdr["grating"] == "500/5500":
            thresh = 1000.0
        # Check for bias frames
        if hdr["exptime"] == 0:
            continue
        print("")
        # ====================
        # Code cut-and-paste from dfocus() -- Get line centers above `thresh`
        # Parameters for DeVeny (2015 Deep-Depletion Device):
        n_spec_pix, prepix = (2048, 50)
        # Trim the image (remove top and bottom rows, pre- and post-scan pixels)
        spec2d = ccd.data[12:512, prepix : prepix + n_spec_pix]
        n_y, n_x = spec2d.shape
        trace = np.full(n_x, n_y / 2, dtype=float).reshape(
            (1, n_x)
        )  # Right down the middle
        spec1d = extract_spectrum(spec2d, trace, win)
        # Find the lines:
        centers, _ = find_lines(spec1d, thresh=thresh, minsep=17)
        cen_list = [f"{cent}" for cent in centers]
        print(f"Found {(n_cen := len(centers))} Line Centers: {cen_list}")
        # ====================

        flex_line_positions.append(
            {
                "filename": fname,
                "obserno": hdr["obserno"],
                "telalt": hdr["telalt"],
                "telaz": hdr["telaz"],
                "rotangle": hdr["rotangle"],
                "utcstart": hdr["utcstart"],
                "lampcal": hdr["lampcal"],
                "grating": hdr["grating"],
                "grangle": hdr["grangle"],
                "slitasec": hdr["slitasec"],
                "nlines": n_cen,
                "xpos": ",".join(cen_list),
            }
        )

    return astropy.table.Table(flex_line_positions)


def validate_lines(table):
    """validate_lines Validate the found lines to produce a uniform set

    The number of lines identified will vary form image to image.  This
    function validates the lines to return the set of lines found in ALL
    images for this grating.

    Parameters
    ----------
    table : `astropy.table.table.Table`
        AstroPy Table as produced by get_line_positions()

    Returns
    -------
    `astropy.table.table.Table`
        AstroPy Table identical to input except the lines are validated
    """
    print("Yay, Validation!!!!")

    # Create a variable to hold the FINAL LINES for this table
    final_lines = None
    for row in table:
        # Line centers found for this image
        cens = np.asarray([float(c) for c in row["xpos"].split(",")])

        # If this is the first one, easy...
        if final_lines is None:
            final_lines = cens
        else:
            # Remove any canonical lines not in every image
            for line in final_lines:
                # If nothing is in the same ballpark (say, 12 pixels),
                #   toss this canonical line
                if np.min(np.absolute(cens - line)) > 12.0:
                    final_lines = final_lines[final_lines != line]

    n_final = len(final_lines)
    print(f"Validated {n_final} lines.")
    # Go back through, and replace the `xpos` value in each row with those
    #  lines corresponding to the good final lines
    xpos = []
    for row in table:
        # Line centers found for this image
        cens = np.asarray([float(c) for c in row["xpos"].split(",")])

        # Keep just the lines that match the canonical lines
        keep_lines = []
        for line in final_lines:
            min_diff = np.min(np.absolute(diffs := cens - line))
            idx = np.where(np.absolute(diffs) == min_diff)
            # This is the line for this image that matches this canonical line
            keep_lines.append(cens[idx])

        # Put the array into a list to wholesale replace `xpos`
        xpos.append(np.asarray(keep_lines).flatten())

    table["nlines"] = [n_final] * len(table)
    table["xpos"] = xpos
    return table


def compute_line_deltas(table):
    """compute_line_deltas Compute line shifts and add to Table

    [extended_summary]

    Parameters
    ----------
    t : `astropy.table.table.Table`
        AstroPy Table as produced by validate_lines()
        Note: Must be VALIDATED, so `xpos` are arrays, not strings

    Returns
    -------
    `astropy.table.table.Table`
        AstroPy Table identical to validated table, with extra columns
    """

    # Things for relating shifts w.r.t. ROTANGLE = 0
    fiducial = table["xpos"][0]
    delta_to_zero = []
    for row in table:
        delta_to_zero.append(row["xpos"] - fiducial)

    # Things for relating shifts w.r.t. MEAN
    xpos = table["xpos"]
    del_mean = np.copy(xpos)
    _, nl = xpos.shape
    for line in range(nl):
        del_mean[:, line] = xpos[:, line] - np.mean(xpos[:, line])

    table["del_zero"] = delta_to_zero
    table["del_mean"] = del_mean

    return table


# Selected (trimmed) routines from `dfocus` ==================================#
#   These are from the LDTObserverTools.dfocus() code, pruned for the
#   immediate use case.


def extract_spectrum(spectrum, traces, nspix):
    """extract_spectrum Object spectral extraction routine

    [extended_summary]

    Parameters
    ----------
    spectrum : [type]
        2D spectral image
    traces : [type]
        Trace line(s) along which to extract the spectrum
    nspix : [type]
        Window width across which to extract the spectrum

    Returns
    -------
    [type]
        2-d array of spectra of individual orders
    """
    # Set # orders, size of each order based on traces dimensionality; 0 -> return
    if traces.ndim == 0:
        return 0
    norders, nx = (1, traces.size) if traces.ndim == 1 else traces.shape

    # Start out with an empty array
    spectra = np.empty((norders, nx), dtype=float)

    # Get the averaged spectra
    for io in range(norders):
        spectra[io, :] = specavg(spectrum, traces[io, :], nspix)

    return spectra


def gaussfit_func(x, a0, a1, a2, a3):
    """gaussfit_func Simple Gaussian function for fitting line profiles

    [extended_summary]

    Parameters
    ----------
    x : [type]
        Array of x values for the fit
    a0 : `float`
        Amplitude of the Gaussian
    a1 : `float`
        Mean of the Gaussian
    a2 : `float`
        Gaussian sigma
    a3 : `float`
        Background offset

    Returns
    -------
    [type]
        Array of y values corresponding to input a's and x
    """
    # Silence RuntimeWarning for overflow, this function only
    warnings.simplefilter("ignore", RuntimeWarning)

    z = (x - a1) / a2
    return a0 * np.exp(-(z**2) / 2.0) + a3


def find_lines(image, thresh=20.0, findmax=50, minsep=11, fit_window=15, verbose=False):
    """find_lines Automatically find and centroid lines in a 1-row image

    [extended_summary]

    Parameters
    ----------
    image : [type]
        [description]
    thresh : `float`, optional
        Threshold above which to indentify lines [Default: 20 DN above bkgd]
    findmax : `int`, optional
        Maximum number of lines to find [Default: 50]
    minsep : `int`, optional
        Minimum line separation for identification [Default: 11 pixels]
    fit_window : `int`, optional
        Size of the window to fit Gaussian [Default: 15 pixels]
    verbose : `bool`
        Produce verbose output?  [Default: False]

    Returns
    -------
    `tuple: float, float`
        centers: List of line centers (pixel #)
        fwhm: The computed FWHM
    """
    # Silence OptimizeWarning, this function only
    warnings.simplefilter("ignore", scipy.optimize.OptimizeWarning)

    # Define the half-window
    fhalfwin = int(np.floor(fit_window / 2))

    # Get size and flatten to 1D
    _, nx = image.shape
    spec = np.ndarray.flatten(image)

    # Find background from median value of the image:
    bkgd = np.median(spec)
    print(
        f"  Background level: {bkgd:.1f}"
        + f"   Detection threshold level: {bkgd+thresh:.1f}"
    )

    # Create empty lists to fill
    cent, fwhm = ([], [])
    j0 = 0

    # Step through the cut and identify peaks:
    for j in range(nx):

        # If we get too close to the end, skip
        if j > (nx - minsep):
            continue

        # If the spectrum at this pixel is above the THRESH...
        if spec[j] > (bkgd + thresh):

            # Mark this pixel as j1
            j1 = j

            # If this is too close to the last one, skip
            if np.abs(j1 - j0) < minsep:
                continue

            # Loop through 0-FINDMAX...  (find central pixel?)
            for jf in range(findmax):
                itmp0 = spec[jf + j]
                itmp1 = spec[jf + j + 1]
                if itmp1 < itmp0:
                    icntr = jf + j
                    break

            # If central pixel is too close to the edge, skip
            if (icntr < minsep / 2) or (icntr > (nx - minsep / 2 - 1)):
                continue

            # Set up the gaussian fitting for this line
            xmin, xmax = (icntr - fhalfwin, icntr + fhalfwin + 1)
            xx = np.arange(xmin, xmax, dtype=float)
            temp = spec[xmin:xmax]
            # Filter the SPEC to smooth it a bit for fitting
            temp = scipy.signal.medfilt(temp, kernel_size=3)

            # Run the fit, with error checking
            try:
                p0 = [1000, np.mean(xx), 3, bkgd]
                aa, _ = scipy.optimize.curve_fit(gaussfit_func, xx, temp, p0=p0)
            except RuntimeError:
                continue  # Just skip this one

            # If the width makes sense, save
            if (fw := aa[2] * 2.355) > 1.0:  # sigma -> FWHM
                cent.append(aa[1])
                fwhm.append(fw)

            # Set j0 to this pixel before looping on
            j0 = jf + j

    # Make list into an array, check again that the centers make sense
    centers = np.asarray(cent)
    c_idx = np.where(np.logical_and(centers > 0, centers <= nx))
    centers = centers[c_idx]

    if verbose:
        print(f" Number of lines: {len(centers)}")

    return (centers, fwhm)


def specavg(spectrum, trace, wsize):
    """specavg Extract an average spectrum along trace of size wsize

    [extended_summary]

    Parameters
    ----------
    spectrum : [type]
        Input Spectrum
    trace : [type]
        The trace along which to extract
    wsize : `int`
        Window size of the extraction (usually odd)

    Returns
    -------
    [type]
        One-dimensional extracted spectrum
    """
    # If ndim = 0, return, otherwise get nx
    if spectrum.ndim == 0:
        return 0
    nx = (spectrum.shape)[-1]

    speca = np.empty(nx, dtype=float)
    whalfsize = int(np.floor(wsize / 2))

    # Because of python indexing, we need to "+1" the upper limit in order
    #   to get the full wsize elements for the average
    for i in range(nx):
        speca[i] = np.average(
            spectrum[int(trace[i]) - whalfsize : int(trace[i]) + whalfsize + 1, i]
        )

    return speca.reshape((1, nx))
