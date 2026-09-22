# SPDX-License-Identifier: MPL-2.0
"""Generic FITS image combination."""

from __future__ import annotations

import pathlib
import typing
import warnings

import astropy.nddata
import astropy.stats
import ccdproc

from pyldt import utils
from pyldt.calibration import (
    PKG_NAME,
    add_package_versions,
    clean_nans,
    savetime,
    write_ccd_atomic,
)
from pyldt.errors import InputError


def _normalize_inputs(
    infiles: tuple[str | pathlib.Path | typing.Iterable[str | pathlib.Path], ...],
    inlist: str | pathlib.Path | None,
) -> list[pathlib.Path]:
    if len(infiles) == 1 and not isinstance(infiles[0], (str, pathlib.Path)):
        files = list(infiles[0])
    else:
        files = list(infiles)
    if files and inlist is not None:
        warnings.warn(
            "Only one of files or inlist may be specified; using inlist.",
            RuntimeWarning,
        )
    if inlist is not None:
        list_path = pathlib.Path(inlist)
        files = []
        with list_path.open("r", encoding="utf-8") as file_object:
            for line in file_object:
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                filename = pathlib.Path(entry)
                files.append(
                    filename if filename.is_absolute() else list_path.parent / filename
                )
    paths = [pathlib.Path(filename) for filename in files]
    if not paths:
        raise InputError("No input files were provided for combination.")
    for filename in paths:
        if not filename.is_file():
            raise FileNotFoundError(f"File {filename} does not exist.")
    return paths


def imcombine(
    *infiles: str | pathlib.Path | typing.Iterable[str | pathlib.Path],
    inlist: str | pathlib.Path | None = None,
    outfn: str | pathlib.Path | None = None,
    del_input: bool = False,
    combine: str | None = None,
    printstat: bool = True,
    overwrite: bool = True,
    returnccd: bool = False,
    mem_limit: float = 8.192e9,
) -> astropy.nddata.CCDData | None:
    """Sigma-clipped combine FITS images supplied directly or through a list."""
    files = _normalize_inputs(infiles, inlist)
    if combine is None:
        combine = "median"
    elif combine not in ("median", "mean"):
        raise InputError("combine must be either 'median' or 'mean'.")

    collection = ccdproc.ImageFileCollection(filenames=files)
    if printstat:
        for image, filename in collection.ccds(return_fname=True):
            minimum, maximum, mean, stddev = utils.mmms(image)
            print(
                f"{filename}:: Min: {minimum:.2f} Max: {maximum:.2f} "
                f"Mean: {mean:.2f} Stddev: {stddev:.2f}"
            )
    if len(files) < 3:
        warnings.warn(
            "Proper combination requires at least three input images; "
            "proceeding regardless.",
            RuntimeWarning,
        )

    combined = clean_nans(
        ccdproc.combine(
            collection.files,
            method=combine,
            sigma_clip=True,
            sigma_clip_dev_func=astropy.stats.mad_std,
            mem_limit=mem_limit,
        )
    )
    combined.header.set(
        "ncombine", len(collection.files), "# of input images in combination"
    )
    combined.header["HISTORY"] = PKG_NAME
    combined.header["HISTORY"] = "Combined image created: " + savetime()
    combined.header["HISTORY"] = (
        f"{combine.title()} combined {len(collection.files)} files:"
    )
    for filename in collection.files:
        combined.header["HISTORY"] = pathlib.Path(filename).name
    combined.header = add_package_versions(combined.header)
    if returnccd:
        return combined

    if outfn is None:
        first_file = files[0]
        outfn = first_file.with_name(f"{first_file.stem}_comb{first_file.suffix}")
    print(f"Saving combined image as {outfn}")
    write_ccd_atomic(combined, outfn, overwrite=overwrite)
    if del_input:
        for filename in files:
            filename.unlink()
    return None


__all__ = ["imcombine"]
