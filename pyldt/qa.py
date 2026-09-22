# SPDX-License-Identifier: MPL-2.0
"""Quality-assurance statistics and plots for image reduction products."""

from __future__ import annotations

import pathlib
import sys

import astropy.nddata
import astropy.stats
import ccdproc
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import obstools.utils

from pyldt import utils


def sample_pixels(data: np.ndarray, max_pixels: int = 1_000_000) -> np.ndarray:
    """Return an evenly strided, allocation-free sample of an image."""
    pixels = np.asarray(data).ravel()
    if max_pixels < 1:
        raise ValueError("max_pixels must be at least one.")
    stride = max(1, int(np.ceil(pixels.size / max_pixels)))
    return pixels[::stride]


def get_histbins(ccd: astropy.nddata.CCDData) -> tuple[np.ndarray, str]:
    """Generate robust, display-friendly histogram bins for a CCD image."""
    _, median, std = astropy.stats.sigma_clipped_stats(
        sample_pixels(ccd.data), sigma=5.0
    )
    if not np.isfinite(median):
        raise ValueError("Cannot generate QA bins for an image without finite data.")

    half_range = 5 * std if np.isfinite(std) and std > 0 else max(abs(median), 1) / 100
    raw_range = median + np.array([-half_range, half_range])
    raw_binsize = (raw_range[1] - raw_range[0]) / 100
    magnitude = 10 ** np.floor(np.log10(raw_binsize))
    candidates = magnitude * np.array([1, 2, 5, 10])
    binsize = candidates[np.searchsorted(candidates, raw_binsize)]

    plotmin = np.floor(raw_range[0] / binsize) * binsize
    plotmax = np.ceil(raw_range[1] / binsize) * binsize
    if plotmax <= plotmin:
        plotmax = plotmin + binsize
    precision = np.maximum(-np.floor(np.log10(binsize)), 0).astype(int)
    label = f"{binsize:.{precision}f}"
    return np.arange(plotmin, plotmax + binsize, binsize), label


def _plot_histograms(
    axis: matplotlib.axes.Axes,
    input_icl: ccdproc.ImageFileCollection,
    output: astropy.nddata.CCDData,
    bins: np.ndarray,
    output_label: str,
    *,
    estimates: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    for ccd in input_icl.ccds():
        axis.hist(
            sample_pixels(ccd.data),
            bins=bins,
            histtype="step",
            linewidth=0.8,
            alpha=0.35,
        )
    counts, edges, _ = axis.hist(
        sample_pixels(output.data),
        bins=bins,
        histtype="step",
        label=output_label,
        linewidth=2.0,
    )
    centers = (edges[:-1] + edges[1:]) / 2
    try:
        fit, _ = obstools.utils.gaussfit(centers, counts, estimates=estimates)
    except (RuntimeError, TypeError, ValueError):
        peak = int(np.argmax(counts))
        bin_width = abs(edges[1] - edges[0])
        fit = np.asarray([counts[peak], centers[peak], bin_width, 0.0])
    axis.plot(
        centers,
        obstools.utils.gaussian_function(centers, *fit),
        color="red",
        label=f"Gaussian Fit to {output_label}",
    )
    return fit, sample_pixels(output.data)


def _annotate_statistics(
    axis: matplotlib.axes.Axes,
    fit: np.ndarray,
    sample: np.ndarray,
    unit: str,
    typesize: float,
) -> None:
    axis.text(
        0.1,
        0.9,
        rf"Gaussian $\mu$ = {fit[1]:.2f} {unit}"
        "\n"
        rf"Gaussian $\sigma$ = {fit[2]:.2f} {unit}",
        transform=axis.transAxes,
        fontsize=typesize,
    )
    sigma = max(abs(fit[2]), sys.float_info.epsilon)
    outliers = np.sum(np.abs(sample - fit[1]) > 5 * sigma)
    axis.text(
        0.1,
        0.75,
        rf"Pixels beyond 5$\sigma$ = {outliers / sample.size * 100:.2f}%",
        transform=axis.transAxes,
        fontsize=typesize,
    )


def bias_plots(
    path: pathlib.Path,
    input_icl: ccdproc.ImageFileCollection,
    output_bias: astropy.nddata.CCDData,
    typesize: float = 8,
) -> None:
    """Write PDF and PNG QA plots for a combined bias."""
    qa_dir = (path / "QA").resolve()
    qa_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing the Bias QA plots to {qa_dir / 'Bias_QA.png'}")
    _, axis = plt.subplots()
    bins, binsize = get_histbins(next(input_icl.ccds()))
    unit = str(output_bias.header.get("BUNIT", "unitless")) or "unitless"
    fit, sample = _plot_histograms(axis, input_icl, output_bias, bins, "Combined Bias")
    _annotate_statistics(axis, fit, sample, unit, typesize)
    axis.set_ylabel(f"N pixels per {binsize} {unit} bin", fontsize=typesize)
    axis.set_xlabel(f"Pixel Value ({unit})", fontsize=typesize)
    axis.set_title(
        f"Bias Frame QA: {len(input_icl.files)} frames", fontsize=typesize + 2
    )
    axis.legend(fontsize=typesize)
    utils.set_std_tickparams(axis, typesize)
    plt.tight_layout()
    for suffix in ("pdf", "png"):
        plt.savefig(qa_dir / f"Bias_QA.{suffix}")
    plt.close()


def flat_plots(
    path: pathlib.Path,
    input_icl: ccdproc.ImageFileCollection,
    output_flat: astropy.nddata.CCDData,
    filtername: str,
    typesize: float = 8,
    flat_type: str | None = None,
) -> None:
    """Write PDF and PNG QA plots for a combined flat."""
    qa_dir = (path / "QA").resolve()
    qa_dir.mkdir(parents=True, exist_ok=True)
    flat_label = "" if flat_type is None else f"{flat_type.replace(' ', '')}_"
    qa_stem = f"Flat_{flat_label}{filtername}_QA"
    print(f"Writing the Flat QA plots to {qa_dir / f'{qa_stem}.png'}")
    _, axis = plt.subplots()
    bins, binsize = get_histbins(next(input_icl.ccds()))
    unit = str(output_flat.header.get("BUNIT", "unitless")) or "unitless"
    output_sample = sample_pixels(output_flat.data)
    counts, edges = np.histogram(output_sample, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    estimates = [np.amax(counts), centers[np.argmax(counts)], 0.1]
    fit, sample = _plot_histograms(
        axis,
        input_icl,
        output_flat,
        bins,
        "Combined Flat",
        estimates=estimates,
    )
    _annotate_statistics(axis, fit, sample, unit, typesize)
    axis.set_ylabel(f"N pixels per {binsize} {unit} bin", fontsize=typesize)
    axis.set_xlabel(f"Pixel Value ({unit})", fontsize=typesize)
    type_label = "" if flat_type is None else f", {flat_type}"
    axis.set_title(
        f"Flat Frame ({filtername} filter{type_label}) QA: "
        f"{len(input_icl.files)} frames",
        fontsize=typesize + 2,
    )
    axis.legend(fontsize=typesize)
    utils.set_std_tickparams(axis, typesize)
    plt.tight_layout()
    for suffix in ("pdf", "png"):
        plt.savefig(qa_dir / f"{qa_stem}.{suffix}")
    plt.close()
