from __future__ import annotations

import pathlib

import astropy.nddata
import astropy.units as u
import numpy as np
import pytest

from pyldt.imutils import make_flat_as_star


def test_make_flat_as_star_does_not_mutate_ccd_input(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    shape = (400, 1600)
    flat_data = np.full(shape, 1000.0)
    flat = astropy.nddata.CCDData(flat_data.copy(), unit=u.adu)
    for key, value in {
        "filename": "flat.0001.fits",
        "exptime": 10.0,
        "date-obs": "2026-01-01",
        "utcstart": "01:00:00",
        "ut": "01:00:00",
        "utcend": "01:00:10",
        "lst-obs": "02:00:00",
        "st": "02:00:00",
    }.items():
        flat.header[key] = value
    bias = astropy.nddata.CCDData(np.full(shape, 100.0), unit=u.adu)
    bias_filename = tmp_path / "bias.fits"
    bias.write(bias_filename)

    make_flat_as_star(flat, bias_filename, 2, verbose=False)

    np.testing.assert_array_equal(flat.data, flat_data)
    assert (tmp_path / "flat.0002.fits").is_file()
