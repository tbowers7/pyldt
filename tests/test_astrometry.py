from __future__ import annotations

import pathlib

import astropy.io.fits
import astropy.wcs
import numpy as np
import pytest

from pyldt import astrometry


def _wcs(crval: tuple[float, float]) -> astropy.wcs.WCS:
    wcs = astropy.wcs.WCS(naxis=2)
    wcs.wcs.crpix = [10.0, 10.0]
    wcs.wcs.cdelt = np.array([-0.0001, 0.0001])
    wcs.wcs.crval = crval
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def test_validate_solution_rejects_wrong_sky_center() -> None:
    solved = _wcs((180.0, 20.0))
    existing = _wcs((10.0, 20.0))

    selected, accepted = astrometry.validate_solution(solved, existing)

    assert not accepted
    assert selected is existing


def test_solve_field_stops_after_connection_failures(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingAstrometryNet:
        attempts = 0

        def solve_from_image(self, *args, **kwargs):
            type(self).attempts += 1
            raise ConnectionError("offline")

    monkeypatch.setattr(
        astrometry.astroquery.astrometry_net, "AstrometryNet", FailingAstrometryNet
    )

    with pytest.raises(ConnectionError, match="offline"):
        astrometry.solve_field(tmp_path / "missing.fits", max_attempts=2, retry_delay=0)

    assert FailingAstrometryNet.attempts == 2


def test_solve_field_tolerates_missing_scale_comment(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.ones((20, 20), dtype=float)
    header = _wcs((10.0, 20.0)).to_header()
    header["BUNIT"] = "adu"
    filename = tmp_path / "image.fits"
    astropy.io.fits.PrimaryHDU(image, header=header).writeto(filename)

    class SuccessfulAstrometryNet:
        def solve_from_image(self, *args, **kwargs):
            return header.copy()

    monkeypatch.setattr(
        astrometry.astroquery.astrometry_net, "AstrometryNet", SuccessfulAstrometryNet
    )

    astrometry.solve_field(filename, validate=False, add_scale=True)

    with astropy.io.fits.open(filename) as hdul:
        assert "SCALE" not in hdul[0].header


def test_solve_field_reads_single_scale_comment(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.ones((20, 20), dtype=float)
    header = _wcs((10.0, 20.0)).to_header()
    header["BUNIT"] = "adu"
    filename = tmp_path / "image.fits"
    astropy.io.fits.PrimaryHDU(image, header=header).writeto(filename)

    class SuccessfulAstrometryNet:
        def solve_from_image(self, *args, **kwargs):
            solved_header = header.copy()
            solved_header["COMMENT"] = "scale: 0.36 arcsec/pix"
            return solved_header

    monkeypatch.setattr(
        astrometry.astroquery.astrometry_net, "AstrometryNet", SuccessfulAstrometryNet
    )

    astrometry.solve_field(filename, validate=False, add_scale=True)

    with astropy.io.fits.open(filename) as hdul:
        assert hdul[0].header["SCALE"] == pytest.approx(0.36)
