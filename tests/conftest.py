from __future__ import annotations

import pathlib

import astropy.nddata
import astropy.units as u
import numpy as np


def write_ccd(
    path: pathlib.Path,
    name: str,
    data: np.ndarray,
    *,
    image_type: str = "object",
    filt: str = "R",
    unit: u.UnitBase = u.adu,
) -> pathlib.Path:
    """Write a minimal PyLDT-compatible CCD frame for tests."""
    ccd = astropy.nddata.CCDData(np.asarray(data, dtype=float), unit=unit)
    ccd.header["CCDSUM"] = "2 2"
    ccd.header["IMAGETYP"] = image_type
    ccd.header["FILTERS"] = filt
    filename = path / name
    ccd.write(filename)
    return filename
