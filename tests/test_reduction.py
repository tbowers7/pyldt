from __future__ import annotations

import pathlib

import astropy.nddata
import astropy.io.fits
import astropy.units as u
import ccdproc
import numpy as np
import pytest

import pyldt.reduction
from pyldt.reduction import ImageDirectory, InputError, LMI, imcombine

from conftest import write_ccd


def test_imcombine_default_output_accepts_collection(tmp_path: pathlib.Path) -> None:
    files = [
        write_ccd(tmp_path, f"input{index}.fits", np.arange(16).reshape(4, 4) + index)
        for index in range(3)
    ]

    imcombine(files, printstat=False)

    assert (tmp_path / "input0_comb.fits").is_file()


def test_imcombine_accepts_variadic_files(tmp_path: pathlib.Path) -> None:
    files = [
        write_ccd(tmp_path, f"input{index}.fits", np.arange(16).reshape(4, 4) + index)
        for index in range(3)
    ]

    combined = imcombine(*files, printstat=False, returnccd=True)

    assert combined is not None
    assert combined.header["NCOMBINE"] == 3


def test_constant_image_has_valid_qa_bins() -> None:
    ccd = astropy.nddata.CCDData(np.ones((4, 4)), unit=u.adu)

    bins, label = ImageDirectory.get_qa_histbins(ccd)

    assert len(bins) >= 2
    assert np.all(np.isfinite(bins))
    assert np.all(np.diff(bins) > 0)
    assert label


def test_clean_nans_handles_missing_uncertainty_and_preserves_mask() -> None:
    data = np.arange(25, dtype=float).reshape(5, 5)
    data[2, 2] = np.nan
    original_mask = np.zeros_like(data, dtype=bool)
    original_mask[0, 0] = True
    ccd = astropy.nddata.CCDData(data, unit=u.adu, mask=original_mask)

    cleaned = ImageDirectory.clean_nans(ccd)

    assert np.isfinite(cleaned.data[2, 2])
    assert cleaned.mask[0, 0]
    assert not cleaned.mask[2, 2]
    assert cleaned.uncertainty is None


def test_parse_lois_ampids_uses_only_numbered_keys() -> None:
    header = astropy.io.fits.Header(
        {"NUMAMP": 2, "AMPID": "ignored", "AMPID02": "B", "AMPID01": "A"}
    )

    assert pyldt.reduction.parse_lois_ampids(header) == "AB"


def _write_bias(path: pathlib.Path, name: str, value: int) -> pathlib.Path:
    ccd = astropy.nddata.CCDData(np.full((5, 5), value, dtype=np.int16), unit=u.adu)
    ccd.header["CCDSUM"] = "2 2"
    ccd.header["IMAGETYP"] = "bias"
    filename = path / name
    ccd.write(filename)
    return filename


def test_bias_combine_uses_only_raw_frames_from_current_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bias(tmp_path, "lmi.0001.fits", 100)
    _write_bias(tmp_path, "lmi.0002.fits", 102)
    _write_bias(tmp_path, "lmi.9999t.fits", 999)
    images = LMI(tmp_path, bin_factor=2, debug=False)
    monkeypatch.setattr(pyldt.reduction, "wrap_trim_oscan", lambda ccd, **kwargs: ccd)
    monkeypatch.setattr(images, "QA_bias", lambda *args, **kwargs: None)

    images.bias_combine(keep_orig=True, keep_trimmed=True)

    master = astropy.nddata.CCDData.read(tmp_path / "bias_bin2.fits")
    assert master.header["NCOMBINE"] == 2


def test_bias_combine_defers_raw_deletion_until_success(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = [
        _write_bias(tmp_path, "lmi.0001.fits", 100),
        _write_bias(tmp_path, "lmi.0002.fits", 102),
    ]
    images = LMI(tmp_path, bin_factor=2, debug=False)
    monkeypatch.setattr(pyldt.reduction, "wrap_trim_oscan", lambda ccd, **kwargs: ccd)
    monkeypatch.setattr(
        ccdproc,
        "combine",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("combine failed")),
    )

    with pytest.raises(RuntimeError, match="combine failed"):
        images.bias_combine()

    assert all(filename.exists() for filename in inputs)


def _write_flat_inputs(path: pathlib.Path) -> list[pathlib.Path]:
    inputs = []
    for sequence, level in enumerate((100.0, 102.0), start=1):
        data = level + np.arange(100, dtype=float).reshape(10, 10) / 100
        inputs.append(
            write_ccd(
                path,
                f"lmi.{sequence:04d}b.fits",
                data,
                image_type="sky flat",
                unit=u.electron,
            )
        )
    return inputs


def test_flat_combine_ignores_stale_normalized_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_flat_inputs(tmp_path)
    write_ccd(
        tmp_path,
        "lmi.9999n.fits",
        np.full((10, 10), 999.0),
        image_type="sky flat",
        unit=u.electron,
    )
    images = LMI(tmp_path, bin_factor=2, debug=False)
    monkeypatch.setattr(images, "QA_flat", lambda *args, **kwargs: None)

    images.flat_combine(keep_subtracted=True, keep_normalized=True)

    master = astropy.nddata.CCDData.read(tmp_path / "flat_bin2_skyflat_R.fits")
    assert master.header["NCOMBINE"] == 2


def test_flat_combine_defers_input_deletion_until_success(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_flat_inputs(tmp_path)
    images = LMI(tmp_path, bin_factor=2, debug=False)
    monkeypatch.setattr(
        ccdproc,
        "combine",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("combine failed")),
    )

    with pytest.raises(RuntimeError, match="combine failed"):
        images.flat_combine()

    assert all(filename.exists() for filename in inputs)


SKY = np.tile([0.5, 1.5], (4, 2))
DOME = np.tile([1.5, 0.5], (4, 2))
SCIENCE = np.full((4, 4), 12.0)


def _run_flat_division(
    path: pathlib.Path,
    flat_types: list[tuple[str, str, np.ndarray]],
    requested: str | None = None,
) -> astropy.nddata.CCDData:
    for tag, image_type, data in flat_types:
        write_ccd(
            path,
            f"flat_bin2_{tag}_R.fits",
            data,
            image_type=image_type,
        )
    write_ccd(path, "lmi.0001b.fits", SCIENCE)
    images = LMI(path, bin_factor=2, debug=False)
    kwargs = {} if requested is None else {"flat_type": requested}
    images.divide_by_flat(keep_subtracted=True, **kwargs)
    return astropy.nddata.CCDData.read(path / "lmi.0001f.fits")


def test_divide_by_flat_defaults_to_sky_flat(tmp_path: pathlib.Path) -> None:
    result = _run_flat_division(
        tmp_path,
        [("skyflat", "sky flat", SKY), ("domeflat", "dome flat", DOME)],
    )
    np.testing.assert_allclose(result.data, SCIENCE / SKY)


def test_divide_by_flat_uses_only_available_type(tmp_path: pathlib.Path) -> None:
    result = _run_flat_division(
        tmp_path, [("domeflat", "dome flat", DOME)], requested="skyflat"
    )
    np.testing.assert_allclose(result.data, SCIENCE / DOME)


def test_divide_by_flat_rejects_missing_type_when_ambiguous(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(InputError, match="not available"):
        _run_flat_division(
            tmp_path,
            [("skyflat", "sky flat", SKY), ("domeflat", "dome flat", DOME)],
            requested="lampflat",
        )
