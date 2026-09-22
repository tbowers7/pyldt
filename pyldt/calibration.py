# SPDX-License-Identifier: MPL-2.0
"""Low-level CCD calibration and FITS metadata helpers."""

from __future__ import annotations

# Astropy exposes unit attributes dynamically.
# pylint: disable=no-member

import datetime
import pathlib
import sys
import uuid

import astropy
import astropy.convolution
import astropy.io.fits
import astropy.modeling
import astropy.nddata
import astropy.units as u
import ccdproc
import numpy as np

from pyldt.errors import InputError
from pyldt.version import version

PKG_NAME = f"PyLDT {'=' * 55}"


def write_ccd_atomic(
    ccd: astropy.nddata.CCDData,
    filename: str | pathlib.Path,
    *,
    overwrite: bool = True,
) -> None:
    """Write a CCD to a sibling temporary file and atomically replace it."""
    destination = pathlib.Path(filename)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix}"
    )
    try:
        ccd.write(temporary, overwrite=False)
        if destination.exists() and not overwrite:
            raise OSError(f"File {destination} already exists.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def add_package_versions(hdr: astropy.io.fits.Header) -> astropy.io.fits.Header:
    """Add dependency and PyLDT versions to a FITS header."""
    hdr["VERSPYT"] = (
        ".".join(str(value) for value in sys.version_info[:3]),
        "Python version",
    )
    hdr["VERSAST"] = (astropy.__version__, "Astropy version")
    hdr["VERSCCD"] = (ccdproc.__version__, "CCDPROC version")
    hdr["VERSNPY"] = (np.__version__, "Numpy version")
    hdr["VERSLDT"] = (version, "PyLDT version")
    return hdr


def clean_nans(ccd: astropy.nddata.CCDData) -> astropy.nddata.CCDData:
    """Interpolate non-finite CCD data without discarding existing masks."""
    original_mask = (
        np.zeros(ccd.data.shape, dtype=bool)
        if ccd.mask is None
        else np.asarray(ccd.mask, dtype=bool).copy()
    )
    kernel = astropy.convolution.Gaussian2DKernel(x_stddev=1)
    ccd.data = astropy.convolution.interpolate_replace_nans(ccd.data, kernel)
    ccd.mask = original_mask | ~np.isfinite(ccd.data)
    if ccd.uncertainty is not None:
        ccd.uncertainty.array = astropy.convolution.interpolate_replace_nans(
            ccd.uncertainty.array, kernel
        )
    return ccd


def parse_lois_ampids(hdr: astropy.io.fits.Header) -> str:
    """Return the ordered LOIS amplifier designation from a FITS header."""
    if int(hdr["NUMAMP"]) == 1:
        return str(hdr["AMPID"]).strip()
    amp_keys = sorted(
        (key for key in hdr if key.startswith("AMPID") and key[5:].isdigit()),
        key=lambda key: int(key[5:]),
    )
    if len(amp_keys) != int(hdr["NUMAMP"]):
        raise InputError(
            f"Expected {hdr['NUMAMP']} numbered AMPID keywords; found {len(amp_keys)}."
        )
    return "".join(str(hdr[key]).strip() for key in amp_keys)


def savetime(local: bool = False) -> str:
    """Return a human-readable local or UTC timestamp."""
    now = datetime.datetime.now(None if local else datetime.UTC)
    return f"{now.isoformat(sep=' ', timespec='seconds')} {now.tzname()}"


def trim_oscan(
    ccd: astropy.nddata.CCDData,
    biassec: str,
    trimsec: str,
    oscan_order: int = 1,
) -> astropy.nddata.CCDData:
    """Subtract a modeled overscan and trim an image using FITS sections."""
    _, x_bias = ccdproc.utils.slices.slice_from_string(biassec, fits_convention=True)
    y_trim, x_trim = ccdproc.utils.slices.slice_from_string(
        trimsec, fits_convention=True
    )
    ccd = ccdproc.trim_image(ccd[y_trim.start : y_trim.stop, :])
    ccd = ccdproc.subtract_overscan(
        ccd,
        overscan=ccd[:, x_bias.start : x_bias.stop],
        median=True,
        model=astropy.modeling.models.Chebyshev1D(oscan_order),
    )
    return ccdproc.trim_image(ccd[:, x_trim.start : x_trim.stop])


def wrap_trim_oscan(
    ccd: astropy.nddata.CCDData, gain_correct: bool = True
) -> astropy.nddata.CCDData:
    """Overscan, trim, gain-correct, and assemble one or more amplifiers."""
    header = ccd.header
    if header["NUMAMP"] == 1:
        trimmed = trim_oscan(ccd, header["BIASSEC"], header["TRIMSEC"])
        if gain_correct and trimmed.unit == u.adu:
            trimmed = ccdproc.gain_correct(
                trimmed, header["GAIN"], gain_unit=u.electron / u.adu
            )
        return trimmed

    data = np.zeros_like(ccd.data, dtype=float)
    mask = np.ones_like(ccd.data, dtype=bool)
    uncertainty = None
    uncertainty_type = None
    amp_numbers = [
        key[5:]
        for key in sorted(
            header,
            key=lambda item: (
                int(item[5:]) if item.startswith("AMPID") and item[5:].isdigit() else -1
            ),
        )
        if key.startswith("AMPID") and key[5:].isdigit()
    ]
    if len(amp_numbers) != int(header["NUMAMP"]):
        raise InputError(
            f"Expected {header['NUMAMP']} numbered AMPID keywords; "
            f"found {len(amp_numbers)}."
        )
    for amp_number in amp_numbers:
        trim_key = f"TRIM{amp_number}"
        if "51:1585" in header[trim_key]:
            header[trim_key] = header[trim_key].replace("51:1585", "51:1584")
            x_offset = 1
            header["TRIMSEC"] = header["TRIMSEC"].replace("51:3121", "52:3120")
        elif "1586:3121" in header[trim_key]:
            header[trim_key] = header[trim_key].replace("1586:3121", "1587:3121")
            x_offset = -1
            header["TRIMSEC"] = header["TRIMSEC"].replace("51:3121", "52:3120")
        else:
            x_offset = 0
        y_range, x_range = ccdproc.utils.slices.slice_from_string(
            header[trim_key], fits_convention=True
        )
        chunk = trim_oscan(ccd, header[f"BIAS{amp_number}"], header[trim_key])
        if gain_correct and ccd.unit == u.adu:
            chunk = ccdproc.gain_correct(
                chunk,
                header[f"GAIN_{amp_number}"],
                gain_unit=u.electron / u.adu,
            )
        destination = np.s_[
            y_range.start : y_range.stop,
            x_range.start + x_offset : x_range.stop + x_offset,
        ]
        data[destination] = chunk.data
        mask[destination] = (
            False if chunk.mask is None else np.asarray(chunk.mask, dtype=bool)
        )
        if chunk.uncertainty is not None:
            if uncertainty is None:
                uncertainty = np.full_like(ccd.data, np.nan, dtype=float)
                uncertainty_type = type(chunk.uncertainty)
            uncertainty[destination] = chunk.uncertainty.array

    ccd.data = data
    ccd.mask = mask
    ccd.uncertainty = (
        None
        if uncertainty is None or uncertainty_type is None
        else uncertainty_type(uncertainty)
    )
    if gain_correct and ccd.unit == u.adu:
        ccd.unit = u.electron
    y_trim, x_trim = ccdproc.utils.slices.slice_from_string(
        header["TRIMSEC"], fits_convention=True
    )
    return ccdproc.trim_image(
        ccd[y_trim.start : y_trim.stop, x_trim.start : x_trim.stop]
    )
