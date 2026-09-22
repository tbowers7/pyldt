# SPDX-License-Identifier: MPL-2.0
"""Legacy command-line plotting driver for DeVeny flexure analysis."""

from __future__ import annotations

import collections.abc
import pathlib
import sys
import warnings

import astropy.table
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from pyldt.flexure import (
    compute_line_deltas,
    flexure_driver,
    get_line_positions,
    load_images,
    validate_lines,
)


def flexure_analysis(data_dir: str | pathlib.Path, rescan: bool = False) -> None:
    """Analyze and plot every supported DeVeny grating."""
    for grating in ("DV1", "DV2", "DV5"):
        save_fn = pathlib.Path(f"flex_data_{grating}.fits")
        if not rescan and save_fn.is_file():
            table = astropy.table.Table.read(save_fn)
        else:
            table = flexure_driver(data_dir, grating=grating, save_fn=save_fn)
            if table is None:
                continue
        print(table.info)
        make_plots(table, grating)


def make_plots(table: astropy.table.Table, grating: str) -> None:
    """Plot flexure measurements and fitted sinusoidal trends."""
    _, axis = plt.subplots()
    typesize = 8

    x_values, y_values, keys = construct_plotting_pairs(
        table, "rotangle", "del_zero", "telalt"
    )
    for index, (x_value, y_value) in enumerate(zip(x_values, y_values, strict=True)):
        color = f"C{index % 10}"
        axis.plot(x_value, y_value, f"{color}.")
        fit_x = np.swapaxes(np.tile(x_value, [len(y_value[0]), 1]), 0, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.optimize.OptimizeWarning)
            parameters, _ = scipy.optimize.curve_fit(
                sinusoid, fit_x.ravel(), y_value.ravel(), p0=[1, 1, 0, 0]
            )
        plot_x = np.linspace(np.min(fit_x), np.max(fit_x), 101)
        plot_y = sinusoid(plot_x, *parameters)
        axis.plot(
            plot_x,
            plot_y,
            f"{color}-",
            label=f"El = {keys[index]:.0f}" + r"$^\circ$",
        )

    axis.set_xlabel("Cassegrain Rotator Angle [deg]", fontsize=typesize)
    axis.set_ylabel(
        r"Line Center Deviation from CASS=$0^\circ$ Position [pixels]",
        fontsize=typesize,
    )
    axis.legend(loc="upper left", fontsize=typesize)
    axis.tick_params("both", labelsize=typesize, direction="in", top=True, right=True)
    plt.tight_layout()
    for suffix in ("eps", "png"):
        plt.savefig(f"flexure_analysis_{grating}.{suffix}")
    plt.close()


def sinusoid(
    x_value: np.ndarray | float,
    amplitude: float,
    _frequency: float,
    phase: float,
    offset: float,
) -> np.ndarray | float:
    """Evaluate a one-cycle-per-360-degree sinusoidal flexure model."""
    return amplitude * np.sin(np.deg2rad(x_value + phase)) + offset


def construct_plotting_pairs(
    table: astropy.table.Table, abscissa: str, ordinate: str, sort: str
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Group plotting coordinates by another table column."""
    abscissa = abscissa.lower()
    ordinate = ordinate.lower()
    sort = sort.lower()
    grouped = table.group_by(sort)
    keys = grouped.groups.keys[sort]
    x_values = []
    y_values = []
    for key in keys:
        subset = grouped.groups[keys == key]
        x_values.append(np.asarray(subset[abscissa]))
        y_values.append(np.asarray(subset[ordinate]))
    return x_values, y_values, keys


def main(args: collections.abc.Sequence[str]) -> None:
    """Run the command-line interface."""
    if len(args) < 2:
        print(f"ERROR: script {args[0]} requires the DATA_DIR to analyze.")
        return
    if not pathlib.Path(args[1]).is_dir():
        print("ERROR: DATA_DIR must be a directory containing the data to analyze.")
        return

    rescan = len(args) > 2 and args[2].casefold() == "rescan"
    first_ignored = 3 if rescan else 2
    if len(args) > first_ignored:
        print(f"WARNING: ignoring arguments: {args[first_ignored:]}")
    flexure_analysis(args[1], rescan=rescan)


if __name__ == "__main__":
    main(sys.argv)


__all__ = [
    "compute_line_deltas",
    "construct_plotting_pairs",
    "flexure_analysis",
    "get_line_positions",
    "load_images",
    "main",
    "make_plots",
    "sinusoid",
    "validate_lines",
]
