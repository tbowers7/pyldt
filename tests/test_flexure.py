from __future__ import annotations

import astropy.table
import numpy as np
import pytest

from pyldt.flexure import extract_spectrum, find_lines, specavg, validate_lines


def test_extract_spectrum_accepts_one_dimensional_trace() -> None:
    image = np.arange(50, dtype=float).reshape(5, 10)
    trace = np.full(10, 2.0)

    one_dimensional = extract_spectrum(image, trace, 3)
    two_dimensional = extract_spectrum(image, trace[np.newaxis, :], 3)

    np.testing.assert_allclose(one_dimensional, two_dimensional)


def test_find_lines_does_not_overrun_rising_spectrum() -> None:
    spectrum = np.arange(20.0).reshape(1, 20)

    centers, widths = find_lines(spectrum, thresh=-100, findmax=50, minsep=2)

    assert centers.size == 0
    assert widths == []


def test_specavg_clips_edge_windows() -> None:
    image = np.arange(20, dtype=float).reshape(2, 10)
    trace = np.zeros(10)

    extracted = specavg(image, trace, 3)

    np.testing.assert_allclose(extracted, np.mean(image[:2], axis=0)[np.newaxis, :])


def test_specavg_rejects_trace_with_wrong_length() -> None:
    with pytest.raises(ValueError, match="Trace length"):
        specavg(np.ones((5, 4)), np.ones(3), 3)


def test_validate_lines_reports_empty_detections() -> None:
    table = astropy.table.Table({"xpos": [""]})

    with pytest.raises(ValueError, match="No lines"):
        validate_lines(table)
