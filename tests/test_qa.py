from __future__ import annotations

import pathlib

import astropy.nddata
import astropy.units as u
import ccdproc
import numpy as np
import pytest

from pyldt import qa


def test_constant_bias_plot_falls_back_when_gaussian_fit_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filenames = []
    for index in range(2):
        filename = tmp_path / f"bias{index}.fits"
        astropy.nddata.CCDData(np.ones((8, 8)), unit=u.adu).write(filename)
        filenames.append(filename)
    collection = ccdproc.ImageFileCollection(filenames=filenames)
    monkeypatch.setattr(
        qa.obstools.utils,
        "gaussfit",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no fit")),
    )

    qa.bias_plots(
        tmp_path,
        collection,
        astropy.nddata.CCDData(np.ones((8, 8)), unit=u.adu),
    )

    assert (tmp_path / "QA" / "Bias_QA.png").is_file()
    assert (tmp_path / "QA" / "Bias_QA.pdf").is_file()
