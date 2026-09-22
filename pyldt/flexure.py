# SPDX-License-Identifier: MPL-2.0
#  Created on 26-Oct-2020
#  @author: tbowers
"""
PyLDT contains image calibration routines for LDT facility instruments

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

from __future__ import annotations

# Built-In Libraries
import pathlib
import warnings

# Third-Party Libraries
import astropy.table
import ccdproc
import numpy as np
import scipy.optimize
import scipy.signal
import scipy.spatial.distance

# Internal Imports


def flexure_driver(
    data_dir: str | pathlib.Path,
    grating: str = "DV2",
    save_fn: str | pathlib.Path = "test.fits",
) -> astropy.table.Table | None:
    """
    Run the DeVeny flexure analysis.

    Load comparison images, measure and validate their line positions, compute
    offsets, and save the resulting table.

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


def load_images(
    data_dir: str | pathlib.Path, grating: str = "DV2"
) -> ccdproc.ImageFileCollection:
    """
    Load comparison images for a grating.

    The returned collection is filtered by grating without modifying the
    source files.

    Parameters
    ----------
    data_dir : `str`
        The directory containing the data to analyze
    grating : `str`, optional
        The grating ID to use.  [Default: DV2]

    Returns
    -------
    `ccdproc.image_collection.ImageFileCollection`
        IFC of the files meeting the input criteria
    """
    # Dictionary
    gratid = {"DV1": "150/5000", "DV2": "300/4000", "DV5": "500/5500"}

    # Load the images of interest in to an ImageFileCollection()
    icl = ccdproc.ImageFileCollection(data_dir)

    # Return an ImageFileCollection filtered by the grating desired
    try:
        grating_name = gratid[grating]
    except KeyError as err:
        raise ValueError(
            f"Unknown grating {grating!r}; choose one of {', '.join(gratid)}."
        ) from err
    return icl.filter(grating=grating_name)


def get_line_positions(
    icl: ccdproc.ImageFileCollection, win: int = 11, thresh: float = 5000.0
) -> astropy.table.Table:
    """
    Compute line positions for images in a collection.

    Extract a central spectrum from every non-bias comparison image and store
    its detected line centers with the relevant observing metadata.

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
    # Put everything into a list of dictionaries
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
                "telalt": np.round(hdr["telalt"]),
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


def validate_lines(table: astropy.table.Table) -> astropy.table.Table:
    """
    Reduce detected lines to a set shared by every image.

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
    if len(table) == 0:
        raise ValueError("No lines are available to validate: the table is empty.")
    if any(not str(value).strip() for value in table["xpos"]):
        raise ValueError("No lines were detected in one or more input images.")

    print("Validating lines...")

    detections = [
        np.asarray([float(item) for item in str(row["xpos"]).split(",")])
        for row in table
    ]

    # Retain only baseline lines that have a distinct match in every image.
    # A nearest-neighbor assignment without reuse prevents two canonical lines
    # from collapsing onto the same detected feature.
    final_lines = detections[0]
    for centers in detections[1:]:
        distances = scipy.spatial.distance.cdist(
            final_lines[:, np.newaxis], centers[:, np.newaxis]
        )
        rows, columns = scipy.optimize.linear_sum_assignment(distances)
        matched = np.zeros(len(final_lines), dtype=bool)
        matched[rows] = distances[rows, columns] <= 12.0
        final_lines = final_lines[matched]
        if final_lines.size == 0:
            raise ValueError("No common spectral lines were found in every image.")

    n_final = len(final_lines)
    print(f"Validated {n_final} lines.")
    # Go back through, and replace the `xpos` value in each row with those
    #  lines corresponding to the good final lines
    xpos = []
    for centers in detections:
        distances = scipy.spatial.distance.cdist(
            final_lines[:, np.newaxis], centers[:, np.newaxis]
        )
        rows, columns = scipy.optimize.linear_sum_assignment(distances)
        matched = np.empty(len(final_lines), dtype=float)
        matched[rows] = centers[columns]
        xpos.append(matched)

    table["nlines"] = [n_final] * len(table)
    table["xpos"] = xpos
    return table


def compute_line_deltas(table: astropy.table.Table) -> astropy.table.Table:
    """
    Compute line shifts and add them to a table.

    Shifts are calculated relative to both the first image and the mean line
    position across all images.

    Parameters
    ----------
    table : `astropy.table.table.Table`
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


def extract_spectrum(
    spectrum: np.ndarray, traces: np.ndarray, nspix: int
) -> np.ndarray | int:
    """
    Extract spectra along one or more traces.

    Each trace is boxcar averaged over ``nspix`` spatial pixels.

    Parameters
    ----------
    spectrum : numpy.ndarray
        2D spectral image
    traces : numpy.ndarray
        Trace line(s) along which to extract the spectrum
    nspix : int
        Window width across which to extract the spectrum

    Returns
    -------
    numpy.ndarray or int
        Two-dimensional array of extracted spectra, or zero when ``traces``
        is scalar.
    """
    # Set # orders, size of each order based on traces dimensionality; 0 -> return
    if traces.ndim == 0:
        return 0
    use_traces = traces[np.newaxis, :] if traces.ndim == 1 else traces
    norders, nx = use_traces.shape

    # Start out with an empty array
    spectra = np.empty((norders, nx), dtype=float)

    # Get the averaged spectra
    for io in range(norders):
        spectra[io, :] = specavg(spectrum, use_traces[io, :], nspix)

    return spectra


def gaussfit_func(
    x: np.ndarray, a0: float, a1: float, a2: float, a3: float
) -> np.ndarray:
    """
    Evaluate a Gaussian line profile with a constant background.

    Parameters
    ----------
    x : numpy.ndarray
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
    numpy.ndarray
        Array of y values corresponding to input a's and x
    """
    z = (x - a1) / a2
    with np.errstate(over="ignore", invalid="ignore"):
        return a0 * np.exp(-(z**2) / 2.0) + a3


def find_lines(
    image: np.ndarray,
    thresh: float = 20.0,
    findmax: int = 50,
    minsep: int = 11,
    fit_window: int = 15,
    verbose: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """
    Find and centroid emission lines in a one-row image.

    Candidate peaks above the background threshold are fit with Gaussian
    profiles to obtain subpixel centers and widths.

    Parameters
    ----------
    image : numpy.ndarray
        One-row extracted spectrum.
    thresh : `float`, optional
        Threshold above which to identify lines [Default: 20 DN above bkgd]
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
    tuple of numpy.ndarray and list of float
        Array of line centers in pixels and their fitted FWHM values.
    """
    # Define the half-window
    fhalfwin = int(np.floor(fit_window / 2))

    # Get size and flatten to 1D
    spec = np.asarray(image).ravel()
    nx = spec.size

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
    for j in range(nx - 1):
        # If the spectrum at this pixel is above the THRESH...
        if spec[j] > (bkgd + thresh):
            # Mark this pixel as j1
            j1 = j

            # If this is too close to the last one, skip
            if np.abs(j1 - j0) < minsep:
                continue

            # Search only through pixels that actually remain in the spectrum.
            icntr = None
            search_stop = min(j + findmax, nx - 1)
            for candidate in range(j, search_stop):
                itmp0 = spec[candidate]
                itmp1 = spec[candidate + 1]
                if itmp1 < itmp0:
                    icntr = candidate
                    break

            # A monotonically rising tail has no local maximum to fit.
            if icntr is None:
                continue

            # If central pixel is too close to the edge, skip
            if (icntr < minsep / 2) or (icntr > (nx - minsep / 2 - 1)):
                continue

            # Set up the gaussian fitting for this line
            xmin = max(0, icntr - fhalfwin)
            xmax = min(nx, icntr + fhalfwin + 1)
            if xmax - xmin < 4:
                continue
            xx = np.arange(xmin, xmax, dtype=float)
            temp = spec[xmin:xmax]
            # Filter the SPEC to smooth it a bit for fitting
            temp = scipy.signal.medfilt(temp, kernel_size=3)

            # Run the fit, with error checking
            try:
                p0 = [1000, np.mean(xx), 3, bkgd]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", scipy.optimize.OptimizeWarning)
                    aa, _ = scipy.optimize.curve_fit(gaussfit_func, xx, temp, p0=p0)
            except (RuntimeError, ValueError):
                continue  # Just skip this one

            # If the width makes sense, save
            if (fw := aa[2] * 2.355) > 1.0:  # sigma -> FWHM
                cent.append(aa[1])
                fwhm.append(fw)

            # Set j0 to this pixel before looping on
            j0 = icntr

    # Make list into an array, check again that the centers make sense
    centers = np.asarray(cent)
    valid = np.logical_and(centers > 0, centers < nx)
    centers = centers[valid]
    fwhm = np.asarray(fwhm)[valid].tolist()

    if verbose:
        print(f" Number of lines: {len(centers)}")

    return (centers, fwhm)


def specavg(spectrum: np.ndarray, trace: np.ndarray, wsize: int) -> np.ndarray | int:
    """
    Extract an average spectrum along a trace.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Input Spectrum
    trace : numpy.ndarray
        The trace along which to extract
    wsize : `int`
        Window size of the extraction (usually odd)

    Returns
    -------
    numpy.ndarray or int
        Extracted spectrum as a row array, or zero when ``spectrum`` is
        scalar.
    """
    # If ndim = 0, return, otherwise get nx
    if spectrum.ndim == 0:
        return 0
    nx = (spectrum.shape)[-1]
    if len(trace) != nx:
        raise ValueError("Trace length must match the spectral image width.")
    if wsize < 1:
        raise ValueError("Extraction window size must be positive.")

    speca = np.empty(nx, dtype=float)
    whalfsize = int(np.floor(wsize / 2))

    # Because of python indexing, we need to "+1" the upper limit in order
    #   to get the full wsize elements for the average
    for i in range(nx):
        center = int(trace[i])
        lower = max(0, center - whalfsize)
        upper = min(spectrum.shape[0], center + whalfsize + 1)
        if lower >= upper:
            raise ValueError(f"Trace position {trace[i]} is outside the image.")
        speca[i] = np.average(spectrum[lower:upper, i])

    return speca.reshape((1, nx))
