# SPDX-License-Identifier: MPL-2.0
#  Created on 26-Oct-2020
#  @author: tbowers
"""
PyLDT contains tools for data from Lowell Observatory facility instruments

Lowell Discovery Telescope (Lowell Observatory: Flagstaff, AZ)
http://www.lowell.edu

The high-level image calibration routines in this module are designed for easy
and simple-minded calibration of images from the Lowell Observatory's facility
instruments.  In particular, this module is designed to reduce `imager` data.
All `spectroscopic` data should be reduced with the PypeIt data reduction
pipeline (see https://pypeit.readthedocs.io/en/release/index.html).

Instruments currently supported in this module are the Large Monolithic Imager
(LMI) at LDT, and the NASA42 imager at the Hall 42" on Anderson Mesa.  As the
Peggy Johnson 1m (PJ1M) at Anderson Mesa comes online, this module will be
expanded to include its imager(s).

The top-level classes take in a directory of data and can process them using
class methods to produce calibrated data for use with the data analysis
software of your choosing.
"""

from __future__ import annotations

# Built-In Libraries
import functools
import pathlib
import shutil
import typing
import warnings

# 3rd Party Libraries
import astropy.io.fits
import astropy.nddata
import astropy.stats
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
import ccdproc
import numpy as np
from tqdm import tqdm

# Internal Imports
from pyldt import qa
from pyldt.calibration import (
    PKG_NAME,
    add_package_versions,
    clean_nans,
    parse_lois_ampids,
    savetime,
    trim_oscan,
    wrap_trim_oscan,
    write_ccd_atomic,
)
from pyldt.combine import imcombine
from pyldt.errors import InputError

# Define API
__all__ = [
    "LMI",
    "NASA42",
    "imcombine",
    "parse_lois_ampids",
    "savetime",
    "trim_oscan",
    "wrap_trim_oscan",
]


def _control_warnings(
    method: typing.Callable[..., typing.Any],
) -> typing.Callable[..., typing.Any]:
    """Apply an image directory's warning preference only for one method call."""

    @functools.wraps(method)
    def wrapped(
        self: "ImageDirectory", *args: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        with warnings.catch_warnings():
            if not self.show_warnings:
                warnings.simplefilter("ignore", AstropyWarning)
                warnings.simplefilter("ignore", UserWarning)
            return method(self, *args, **kwargs)

    return wrapped


class ImageDirectory:
    """
    Internal class, parent of all imager classes

    This base class contains collective metadata for a single night's data
    images.  Child classes modify or extend this class for specific differences
    between the various imaging cameras.

    Parameters
    ----------
    path : :obj:`str` or :obj:`pathlib.Path`
        Path to the directory containing the images to be reduced.
    mem_limit : :obj:`float`, optional
        Memory limit for the image combination routine  (Default: 8.192e9 bytes)
    debug : :obj:`bool`, optional
        Print debugging statements?  (Default: True)
    show_warnings : :obj:`bool`, optional
        Show warning messages?  (Default: False)
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        mem_limit: float = 8.192e9,
        debug: bool = True,
        show_warnings: bool = False,
    ) -> None:
        """
        Initialize a directory of images.

        Parameters
        ----------
        path : str or pathlib.Path
            Directory containing images to reduce.
        mem_limit : float, optional
            Memory limit in bytes for image combination.
        debug : bool, optional
            Print processing diagnostics.
        show_warnings : bool, optional
            Show Astropy and user warnings instead of suppressing them.
        """
        # Settings that determine how the class functions
        self.debug = debug
        self.show_warnings = show_warnings

        # Metadata related to all files in this directory
        self.path = pathlib.Path(path)
        if self.debug:
            print(f"Processing images in {self.path}")
        self.mem_limit = mem_limit
        # Attributes that need to be specified for the instrument
        self.biassec = None
        self.trimsec = None
        self.prefix = None
        self.bin_factor = None
        self.binning = None
        # Generic filenames
        self.zerofn = "bias.fits"
        # Create Placeholder for initial ImageFileCollection for the directory
        self.icl = None

    @staticmethod
    def write_ccd_atomic(
        ccd: astropy.nddata.CCDData,
        filename: str | pathlib.Path,
        *,
        overwrite: bool = True,
    ) -> None:
        """Write a CCD to a sibling temporary file and atomically replace it."""
        write_ccd_atomic(ccd, filename, overwrite=overwrite)

    def image_file_collection(
        self, *args: typing.Any, **kwargs: typing.Any
    ) -> ccdproc.ImageFileCollection:
        """Create an image collection under this instance's warning policy."""
        with warnings.catch_warnings():
            if not self.show_warnings:
                warnings.simplefilter("ignore", AstropyWarning)
                warnings.simplefilter("ignore", UserWarning)
            return ccdproc.ImageFileCollection(*args, **kwargs)

    @_control_warnings
    def inspect_images(self) -> None:
        """
        Inspect the images in the specified directory

        Inspects the images in the specified directory, and loads in the
        default BIASSEC and TRIMSEC values (if not specified at Class
        instantiation).  For folders of DeVeny data, also modifies the FILTREAR
        keyword and adds a GRAT_ID keyword containing the DVx name of the
        grating.
        """

        # Print a helpful statement that image inspection is happening
        if self.debug:
            print("Inspecting the images in this directory...")

        # Check that binning is set
        if self.binning is None:
            raise InputError("Binning not set.")
        if self.debug:
            print(f"Binning is: {self.binning.replace(' ','x')}")

        # Refresh the ImageFileCollection
        self.icl.refresh()

        # Set up a progress bar, so we can see how the process is going...
        prog_bar = tqdm(
            total=len(self.icl.files), unit="frame", unit_scale=False, colour="cyan"
        )

        # Loop through files...
        for ccd, fname in self.icl.ccds(ccdsum=self.binning, return_fname=True):
            # Check for empty trimsec/biassec attributes, pull from header
            try:
                if self.biassec is None:
                    self.biassec = ccd.header["biassec"]
                if self.trimsec is None:
                    self.trimsec = ccd.header["trimsec"]
            except KeyError as err:
                warnings.warn(str(err), RuntimeWarning)

            # Add Header Section Specifying pipeline
            ccd.header["DRP_NAME"] = ("PyLDT", "Data Reduction Pipeline " + "=" * 15)
            ccd.header = self.add_package_versions(ccd.header)

            # Fix NASA42 headers stuff
            if ccd.header.get("LCAMMOD", None) == "nasa42":
                ccd.header["INSTRUME"] = ("NASA42", "Instrument")
                ccd.header["FILTERS"] = (
                    ccd.header.get("FILTNAME", ""),
                    "Telescope Filter Name",
                )
                if ccd.header.get("NUMAMP", 1) == 1:
                    # Set GAIN and RDNOISE to that for AMP 1
                    ccd.header["GAIN"] = (
                        ccd.header.get("AGAIN_01", 1.0),
                        "Gain for amplifier 01",
                    )
                    ccd.header["RDNOISE"] = (
                        ccd.header.get("ARDNS_01", 0.0),
                        "Read Noise for amplifier 01",
                    )
                    # Fix the TRIMSEC & BIASSEC keywords to remove the lowest
                    #  60 rows (unbinned) because they stink
                    for section in ["TRIMSEC", "BIASSEC"]:
                        xsec, ysec = ccd.header[section].strip("[]").split(",")
                        ymin, ymax = np.array(ysec.split(":"), dtype=int)
                        ymin = 60 // ccd.header["ADELY_01"] + 1
                        ccd.header[section] = f"[{xsec},{ymin}:{ymax}]"

            # Fix depricated FITS keyword
            if "RADECSYS" in ccd.header:
                ccd.header.rename_keyword("RADECSYS", "RADESYSa")
            self.write_ccd_atomic(ccd, self.path / fname)

            # Update the progress bar
            prog_bar.update(1)
        # Close the progress bar, end of loop
        prog_bar.close()

    def copy_raw(self, overwrite: bool = False) -> None:
        """
        Copy raw FITS files to a ``raw`` backup directory.

        If a directory containing the raw data is not extant, create it and copy
        all FITS files there as a backup.

        Parameters
        ----------
        overwrite : bool, optional
            Replace backup files that already exist.
        """

        raw_data = pathlib.Path(self.path, "raw")
        raw_data.mkdir(exist_ok=True)

        # Copy files to raw_data, overwrite if requested
        for img in sorted(self.path.glob(f"{self.prefix}.????.fits")):
            if (not raw_data.joinpath(img.name).exists()) or overwrite:
                print(f"Copying {img} to {raw_data}...")
                shutil.copy2(img, raw_data)

    @_control_warnings
    def bias_combine(
        self,
        keep_orig: bool = False,
        keep_trimmed: bool = False,
        gain_correct: bool = True,
    ) -> None:
        """
        Find and combine bias frames with the selected binning.

        Each input bias is overscan-subtracted and trimmed before the frames
        are averaged with sigma clipping.

        Parameters
        ----------
        keep_orig : :obj:`bool`, optional
            Keep the original (`i.e.`, input) files?  (Default: False)
        keep_trimmed : :obj:`bool`, optional
            Keep the trimmed (`i.e.`, intermediate) files?  (Default: False)
        gain_correct : :obj:`bool`, optional
            Multiply by the CCD gain before combining?  (Default: True)
        """

        if self.binning is None:
            raise InputError("Binning not set.")
        if self.debug:
            print(
                "Trimming and combining bias frames with binning "
                f"{self.binning.replace(' ','x')} into {self.zerofn}..."
            )

        # First, refresh the ImageFileCollection
        self.icl.refresh()

        # Set up a progress bar, so we can see how the process is going...
        try:
            bias_files = self.icl.files_filtered(imagetyp="bias")
        except TypeError:
            print("No bias frames found!")
            return
        prog_bar = tqdm(
            total=len(bias_files), unit="frame", unit_scale=False, colour="#808080"
        )

        # Loop through files, tracking this invocation's products explicitly so
        # stale trimmed frames cannot enter the master bias.
        trimmed_files = []
        original_bias_files = []
        for ccd, file_name in self.icl.ccds(
            ccdsum=self.binning, imagetyp="bias", bitpix=16, return_fname=True
        ):
            # Fit the overscan section, subtract it, then trim the image
            ccd = wrap_trim_oscan(ccd, gain_correct=gain_correct)

            # Update the header
            ccd.header["HISTORY"] = PKG_NAME
            ccd.header["HISTORY"] = "Trimmed bias saved: " + savetime()
            ccd.header["HISTORY"] = f"Original filename: {file_name}"
            ccd.header = self.add_package_versions(ccd.header)

            # Save the result (suffix = 't'); delete the input file
            trimmed_fn = self.path / f"{file_name[:-5]}t{file_name[-5:]}"
            self.write_ccd_atomic(ccd, trimmed_fn)
            trimmed_files.append(trimmed_fn)
            original_bias_files.append(self.path / file_name)

            # Update the progress bar
            prog_bar.update(1)
        # Close the progress bar, end of loop
        prog_bar.close()

        # Collect the trimmed biases
        t_bias_cl = ccdproc.ImageFileCollection(filenames=trimmed_files)

        # If no trimmed biases, return now
        if not t_bias_cl.files:
            return

        if self.debug:
            print("Doing average combine now...")

        # Perform the NaN-cleaned combination
        comb_bias = self.clean_nans(
            ccdproc.combine(
                [self.path / fn for fn in t_bias_cl.files],
                method="average",
                sigma_clip=True,
                sigma_clip_dev_func=astropy.stats.mad_std,
                mem_limit=self.mem_limit,
            )
        )

        # Make the QA plot(s)
        self.QA_bias(t_bias_cl, comb_bias)

        # Add FITS keyword NCOMBINE and HISTORY
        comb_bias.header.set(
            "ncombine", len(t_bias_cl.files), "# of input images in combination"
        )
        comb_bias.header["HISTORY"] = "Combined bias created: " + savetime()
        comb_bias.header["HISTORY"] = (
            "Average combined " + f"{len(t_bias_cl.files)} files:"
        )
        for fname in t_bias_cl.files:
            comb_bias.header["HISTORY"] = fname

        # Save the result; delete the input files
        comb_bias.header = self.add_package_versions(comb_bias.header)
        self.write_ccd_atomic(comb_bias, self.path / self.zerofn)
        if not keep_orig:
            for filename in original_bias_files:
                filename.unlink()
        if not keep_trimmed:
            for filename in trimmed_files:
                filename.unlink()

    @_control_warnings
    def bias_subtract(self, keep_orig: bool = False, gain_correct: bool = True) -> None:
        """
        Subtract the combined bias from the images.

        Images are overscan-subtracted, trimmed, and optionally gain-corrected
        before the combined bias is removed.

        Parameters
        ----------
        keep_orig : :obj:`bool`, optional
            Keep the original (`i.e.`, input) files?  (Default: False)
        gain_correct : :obj:`bool`, optional
            Multiply by the CCD gain before returning?  (Default: True)
        """

        if self.binning is None:
            raise InputError("Binning not set.")
        if self.debug:
            print("Subtracting bias from remaining images...")

        # Refresh the ImageFileCollection
        self.icl.refresh()

        # Load the appropriate bias frame to subtract
        if not self.path.joinpath(self.zerofn).is_file():
            self.bias_combine()
        try:
            combined_bias = astropy.nddata.CCDData.read(self.path / self.zerofn)
        except FileNotFoundError:
            # Just skip the bias subtraction
            print(f"Skipping bias subtraction for lack of {self.zerofn}")
            return

        # Set up a progress bar, so we can see how the process is going...
        prog_bar = tqdm(
            total=len(self.icl.files), unit="frame", unit_scale=False, colour="blue"
        )

        # Loop through files. Retained raw biases are calibration inputs, not
        # targets for bias subtraction. Defer deletion until the entire stage
        # succeeds.
        processed_inputs = []
        for ccd, file_name in self.icl.ccds(
            ccdsum=self.binning, bitpix=16, return_fname=True
        ):
            if str(ccd.header.get("imagetyp", "")).strip().casefold() == "bias":
                continue

            # Fit the overscan section, subtract it, then trim the image
            ccd = wrap_trim_oscan(ccd, gain_correct=gain_correct)

            # Subtract combined bias
            ccd = ccdproc.subtract_bias(ccd, combined_bias)

            # Update the header
            ccd.header["HISTORY"] = PKG_NAME
            ccd.header["HISTORY"] = "Bias-subtracted image saved: " + savetime()
            ccd.header["HISTORY"] = f"Subtracted bias: {self.zerofn}"
            ccd.header["HISTORY"] = f"Original filename: {file_name}"

            # Save the result (suffix = 'b'); delete input file
            ccd.header = self.add_package_versions(ccd.header)
            self.write_ccd_atomic(ccd, self.path / f"{file_name[:-5]}b{file_name[-5:]}")
            processed_inputs.append(self.path / file_name)

            # Update the progress bar
            prog_bar.update(1)
        # Close the progress bar, end of loop
        prog_bar.close()
        if not keep_orig:
            for filename in processed_inputs:
                filename.unlink()

    @_control_warnings
    def flat_combine(
        self,
        keep_subtracted: bool = False,
        keep_normalized: bool = False,
        norm_use_center_only: bool = True,
    ) -> None:
        """
        Combine flat field frames

        Combine the flat frames for each flat type and filter in the directory
        with a given binning.  Basic emulation of IRAF's flatcombine.  Produces
        a combined flat image saved with the appropriate filename for each
        flat type and filter found.

        Parameters
        ----------
        keep_subtracted : :obj:`bool`, optional
            Keep the bias-subtracted (`i.e.`, input) image?  (Default: False)
        keep_normalized : :obj:`bool`, optional
            Keep the normalized (`i.e.`, intermediate) image?  (Default: False)
        norm_use_center_only : :obj:`bool`, optional
            Use the center 50% (by pixels) of the image only for flat
            normalization?  (Default: True)
        """

        # Load the list of bias-subtracted data frames -- check binning
        bsub_cl = ccdproc.ImageFileCollection(
            self.path, glob_include=f"{self.prefix}.*b.fits"
        )

        if not bsub_cl.files:
            print("No bias-subtracted frames.  Skipping flat combine...")
            return

        if self.binning is None:
            raise InputError("Binning not set.")
        if self.debug:
            print("Normalizing flat field frames...")

        # Filter here to get # of images for progress bar
        flat_cl = bsub_cl.filter(
            ccdsum=self.binning, imagetyp=r"[a-z]+\s*flat", regex_match=True
        )
        print(f"Flat types found: {sorted(set(flat_cl.summary['imagetyp']))}")

        # Set up a progress bar, so we can see how the process is going...
        prog_bar = tqdm(
            total=len(flat_cl.files), unit="frame", unit_scale=False, colour="yellow"
        )

        # Normalize flat field images by the mean value. Keep explicit track of
        # the products made by this invocation so stale intermediates from an
        # earlier or interrupted run cannot enter the combination.
        normalized_files = []
        subtracted_files = []
        for ccd, flat_fn in flat_cl.ccds(return_fname=True):
            # Get the indices of the region over which to measure the mean
            if norm_use_center_only:
                # Select the inner 50% of pixels in the image
                cen = np.array(ccd.data.shape) // 2
                delta = (cen // np.sqrt(2)).astype(int)
                img_slice = np.s_[
                    cen[0] - delta[0] : cen[0] + delta[0] + 1,
                    cen[1] - delta[1] : cen[1] + delta[1] + 1,
                ]
            else:
                img_slice = np.s_[:, :]

            # Perform the division (in a NaN-safe manner)
            ccd = ccd.divide(
                np.nanmean(ccd[img_slice]) * u.Unit(ccd.header["BUNIT"]),
                handle_meta="first_found",
            )

            # Update the header
            ccd.header["HISTORY"] = "Normalized flat saved: " + savetime()
            ccd.header["HISTORY"] = f"Previous filename: {flat_fn}"

            # Save the result (suffix = 'n'); delete the input file
            ccd.header = self.add_package_versions(ccd.header)
            normalized_fn = self.path / f"{flat_fn[:-6]}n{flat_fn[-5:]}"
            self.write_ccd_atomic(ccd, normalized_fn)
            normalized_files.append(normalized_fn)
            subtracted_files.append(self.path / flat_fn)

            # Update the progress bar
            prog_bar.update(1)
        # Close the progress bar, end of loop
        prog_bar.close()

        # Load the list of normalized flat field images
        if not normalized_files:
            print("No flats to be combined.")
            return

        norm_cl = ccdproc.ImageFileCollection(filenames=normalized_files)
        if norm_cl.files:
            # Combine flat field frames separately for each flat type and filter.
            # In particular, do not mix (for example) sky and dome flats taken
            # through the same filter.
            flat_types = sorted(set(norm_cl.summary["imagetyp"]))
            for flat_type in flat_types:
                type_cl = norm_cl.filter(imagetyp=flat_type)
                unique_filters = sorted(set(type_cl.summary["filters"]))
                flat_type_tag = flat_type.replace(" ", "")

                for filt in unique_filters:
                    flats = type_cl.files_filtered(filters=filt, include_path=True)

                    print(
                        f"Combining {len(flats)} {flat_type} frames "
                        f"for filter {filt}..."
                    )
                    # Perform the NaN-cleaned combination
                    cflat = self.clean_nans(
                        ccdproc.combine(
                            flats,
                            method="average",
                            sigma_clip=True,
                            sigma_clip_dev_func=astropy.stats.mad_std,
                            mem_limit=self.mem_limit,
                        )
                    )

                    # Make the QA plot(s)
                    self.QA_flat(
                        ccdproc.ImageFileCollection(filenames=flats),
                        cflat,
                        filt,
                        flat_type=flat_type,
                    )

                    # Add FITS keyword NCOMBINE and HISTORY
                    cflat.header.set(
                        "ncombine", len(flats), "# of input images in combination"
                    )
                    cflat.header["HISTORY"] = PKG_NAME
                    cflat.header["HISTORY"] = "Combined flat created: " + savetime()
                    cflat.header["HISTORY"] = (
                        "Average combined " + f"{len(flats)} files:"
                    )
                    for fname in flats:
                        # Remove the path portion of the filename for the HISTORY
                        cflat.header["HISTORY"] = pathlib.Path(fname).name

                    # Build filename, save, remove input files
                    flat_fn = f"flat_bin{self.bin_factor}_{flat_type_tag}_{filt}.fits"
                    if self.debug:
                        print(f"Saving combined flat as {flat_fn}")
                    cflat.header = self.add_package_versions(cflat.header)
                    self.write_ccd_atomic(cflat, self.path / flat_fn)

            # Delete inputs only after every master flat has been created
            # successfully, leaving a recoverable state if any group fails.
            if not keep_subtracted:
                for filename in subtracted_files:
                    filename.unlink()
            if not keep_normalized:
                for filename in normalized_files:
                    filename.unlink()

        else:
            print("No flats to be combined.")

    @_control_warnings
    def divide_by_flat(
        self, flat_type: str = "skyflat", keep_subtracted: bool = False
    ) -> None:
        """
        Divide frames by the appropriate flatfield

        Divides all LMI science frames by the appropriate flat field image
        This method is LMI-specific, rather than being wrapper for a more
        general function.  Basic emulation of IRAF's ccdproc/flatcor function.

        Parameters
        ----------
        flat_type : :obj:`str`, optional
            Flat type to use when more than one type is available. Spaces and
            capitalization are ignored when matching. If only one flat type is
            available, it is used regardless of this value. (Default:
            ``"skyflat"``)
        keep_subtracted : :obj:`bool`, optional
            Keep the bias-subtracted (`i.e.`, input) image?  (Default: False)
        """
        # Load the list of combined flats and bias-subtracted data frames
        flat_cl = ccdproc.ImageFileCollection(
            self.path, glob_include=f"flat_bin{self.bin_factor}_*.fits"
        )
        sci_cl = ccdproc.ImageFileCollection(
            self.path, glob_include=f"{self.prefix}.*b.fits"
        )

        # If either IFC is empty, return now
        if not sci_cl.files or not flat_cl.files:
            print("No flats and/or no science images.  Skipping flat divide...")
            return

        # Select exactly one type of flat. If there is only one available type,
        # use it even when a different type was requested.
        available_flat_types = sorted(set(flat_cl.summary["imagetyp"]))
        flat_types_by_tag = {
            "".join(str(available_type).split()).casefold(): available_type
            for available_type in available_flat_types
        }
        if len(available_flat_types) == 1:
            selected_flat_type = available_flat_types[0]
        else:
            requested_tag = "".join(flat_type.split()).casefold()
            try:
                selected_flat_type = flat_types_by_tag[requested_tag]
            except KeyError as err:
                available = ", ".join(str(item) for item in available_flat_types)
                raise InputError(
                    f"Flat type {flat_type!r} is not available. "
                    f"Choose one of: {available}."
                ) from err

        flat_cl = flat_cl.filter(imagetyp=selected_flat_type)
        if self.debug:
            print(f"Using {selected_flat_type} frames for flat correction.")

        # Loop through the filters present. Defer removal of the input science
        # frames until every requested filter has completed successfully.
        corrected_inputs = []
        for filt in sorted(set(flat_cl.summary["filters"])):
            # Load in the combined flat for this filter
            if self.debug:
                print(f"Dividing science frames by combined flat for filter: {filt}")
            combined_flat, mflat_fn = next(
                flat_cl.ccds(ccdsum=self.binning, filters=filt, return_fname=True)
            )

            # Set up a progress bar, so we can see how the process is going
            sci_filt_files = sci_cl.files_filtered(filters=filt)
            prog_bar = tqdm(
                total=len(sci_filt_files),
                unit="frame",
                unit_scale=False,
                colour="#D8BFD8",
            )

            # Loop through the science frames to correct
            for ccd, sci_fn in sci_cl.ccds(
                ccdsum=self.binning, filters=filt, return_fname=True
            ):
                # Divide by combined flat
                ccd = ccdproc.flat_correct(ccd, combined_flat)

                # Update the header
                ccd.header["flatcor"] = True
                ccd.header["HISTORY"] = PKG_NAME
                ccd.header["HISTORY"] = "Flat-corrected image saved: " + savetime()
                ccd.header["HISTORY"] = f"Divided by flat: {mflat_fn}"
                ccd.header["HISTORY"] = f"Previous filename: {sci_fn}"

                # Save the result (suffix = 'f'); delete the input file
                ccd.header = self.add_package_versions(ccd.header)
                self.write_ccd_atomic(
                    ccd,
                    self.path / f"{sci_fn[:-6]}f{sci_fn[-5:]}",
                )
                corrected_inputs.append(self.path / sci_fn)

                # Update the progress bar
                prog_bar.update(1)
            # Close the progress bar, end of loop
            prog_bar.close()
        if not keep_subtracted:
            for filename in corrected_inputs:
                filename.unlink()

    def QA_bias(  # pylint: disable=invalid-name
        self,
        input_icl: ccdproc.ImageFileCollection,
        output_bias: astropy.nddata.CCDData,
        typesize: float = 8,
    ) -> None:
        """Produce QA plots for a bias combination."""
        qa.bias_plots(self.path, input_icl, output_bias, typesize)

    def QA_flat(  # pylint: disable=invalid-name
        self,
        input_icl: ccdproc.ImageFileCollection,
        output_flat: astropy.nddata.CCDData,
        filtername: str,
        typesize: float = 8,
        flat_type: str | None = None,
    ) -> None:
        """Produce QA plots for a flat combination."""
        qa.flat_plots(
            self.path,
            input_icl,
            output_flat,
            filtername,
            typesize,
            flat_type,
        )

    @staticmethod
    def sample_pixels(data: np.ndarray, max_pixels: int = 1_000_000) -> np.ndarray:
        """Return an evenly strided sample of an image."""
        return qa.sample_pixels(data, max_pixels)

    @staticmethod
    def get_qa_histbins(
        ccd: astropy.nddata.CCDData,
    ) -> tuple[np.ndarray, str]:
        """Generate robust histogram bins for a CCD image."""
        return qa.get_histbins(ccd)

    @staticmethod
    def add_package_versions(
        hdr: astropy.io.fits.Header,
    ) -> astropy.io.fits.Header:
        """Add dependency and PyLDT versions to a FITS header."""
        return add_package_versions(hdr)

    @staticmethod
    def clean_nans(ccd: astropy.nddata.CCDData) -> astropy.nddata.CCDData:
        """Interpolate non-finite data without discarding existing masks."""
        return clean_nans(ccd)


class LMI(ImageDirectory):
    """
    Manage calibration of a directory containing LMI data.

    Instrument-specific defaults are combined with the common reduction
    workflow implemented by :class:`ImageDirectory`.

    Parameters
    ----------
    path : :obj:`str` or :obj:`pathlib.Path`
        Path to the directory containing the images to be reduced.
    biassec : :obj:`str`, optional
        The IRAF-style overscan region to be subtracted from each frame.
        If unspecified, use the values suggested in the LMI User Manual.
    trimsec : :obj:`str`, optional
        The IRAF-style image region to be retained in each frame.
        If unspecified, use the values suggested in the LMI User Manual.
    bin_factor : :obj:`int`, optional
        The binning factor used to create the image(s) to be processed.
        (Default: 2)
    mem_limit : :obj:`float`, optional
        Memory limit for the image combination routine  (Default: 8.192e9 bytes)
    **kwargs : Any
        Additional options passed to :class:`ImageDirectory`.
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        biassec: str | None = None,
        trimsec: str | None = None,
        bin_factor: int = 2,
        mem_limit: float = 8.192e9,
        **kwargs: typing.Any,
    ) -> None:
        """
        Initialize an LMI image directory.

        Parameters
        ----------
        path : str or pathlib.Path
            Directory containing LMI images.
        biassec : str, optional
            IRAF-style overscan section.
        trimsec : str, optional
            IRAF-style retained image section.
        bin_factor : int, optional
            Detector binning factor.
        mem_limit : float, optional
            Memory limit in bytes for image combination.
        **kwargs : Any
            Additional options passed to :class:`ImageDirectory`.
        """
        # SUPER-INIT!!!
        super().__init__(path, mem_limit=mem_limit, **kwargs)

        # Load up the instance attributes
        self.bin_factor = int(bin_factor)
        self.binning = f"{self.bin_factor} {self.bin_factor}"
        self.biassec = biassec
        self.trimsec = trimsec

        # Define file prefix & standard filenames
        self.prefix = "lmi"
        self.zerofn = f"bias_bin{self.bin_factor}.fits"

        # Load initial ImageFileCollection
        self.icl = self.image_file_collection(
            self.path, glob_include=f"{self.prefix}.????.fits"
        )

    def process_all(self) -> None:
        """
        Run every calibration step for the directory.

        The result of running this method will be to process all of the images
        in the specified directory (and given binning) through all of the basic
        calibration steps.  The procedure is:
            * copy_raw() -- Make a copy of the raw data in a safe place
            * inspect_images() -- Make sure the relevant metadata is set
            * bias_combine() -- Combine the bias frames into a Calibration bias
            * bias_subtract() -- Subtract the bias & overscan from all frames
            * flat_combine() -- Combine flat fields of a given filter
            * divide_by_flat() -- Divide science frames by the appropriate flat
        """
        self.copy_raw()
        self.inspect_images()
        self.bias_combine()
        self.bias_subtract()
        self.flat_combine()
        self.divide_by_flat()


class NASA42(ImageDirectory):
    """
    Manage calibration of a directory containing NASA42 data.

    Instrument-specific naming rules are combined with the common reduction
    workflow implemented by :class:`ImageDirectory`.

    Parameters
    ----------
    path : :obj:`str` or :obj:`pathlib.Path`
        Path to the directory containing the images to be reduced.
    biassec : :obj:`str`, optional
        The IRAF-style overscan region to be subtracted from each frame.
        If unspecified, use the values suggested in the LMI User Manual.
    trimsec : :obj:`str`, optional
        The IRAF-style image region to be retained in each frame.
        If unspecified, use the values suggested in the LMI User Manual.
    bin_factor : :obj:`int`, optional
        The binning factor used to create the image(s) to be processed.
        (Default: 2)
    mem_limit : :obj:`float`, optional
        Memory limit for the image combination routine  (Default: 8.192e9 bytes)
    prefix : :obj:`str`, optional
        The file prefix to use.  If ``None``, the prefix will be pulled from the
        first file in the directory.  (Default: None)
    **kwargs : Any
        Additional options passed to :class:`ImageDirectory`.
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        biassec: str | None = None,
        trimsec: str | None = None,
        bin_factor: int = 2,
        mem_limit: float = 8.192e9,
        prefix: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """
        Initialize a NASA42 image directory.

        Parameters
        ----------
        path : str or pathlib.Path
            Directory containing NASA42 images.
        biassec : str, optional
            IRAF-style overscan section.
        trimsec : str, optional
            IRAF-style retained image section.
        bin_factor : int, optional
            Detector binning factor.
        mem_limit : float, optional
            Memory limit in bytes for image combination.
        prefix : str, optional
            Input filename prefix; inferred when omitted.
        **kwargs : Any
            Additional options passed to :class:`ImageDirectory`.
        """
        # SUPER-INIT!!!
        super().__init__(path, mem_limit=mem_limit, **kwargs)

        # Load up the instance attributes
        self.bin_factor = int(bin_factor)
        self.binning = f"{self.bin_factor} {self.bin_factor}"
        self.biassec = biassec
        self.trimsec = trimsec

        # Define file prefix & standard filenames
        # File prefix -- NASA42 files prefix with the UT date
        if prefix is None:
            # Look at all the 20*.fits files in this directory, and choose
            # Note: This will need to be updated for the year 2100
            fitsfiles = sorted(self.path.glob("20*.????.fits"))
            if fitsfiles:
                self.prefix = fitsfiles[0].name.split(".")[0]
            else:
                raise InputError(
                    "Could not infer the NASA42 filename prefix: no raw FITS "
                    "files matched '20*.????.fits'."
                )
        else:
            self.prefix = prefix
        if self.debug:
            print(f"Directory prefix: {self.prefix}")
        self.zerofn = f"bias_bin{self.bin_factor}.fits"

        # Load initial ImageFileCollection
        self.icl = self.image_file_collection(
            self.path, glob_include=f"{self.prefix}.????.fits"
        )

    def process_all(self) -> None:
        """
        Run every calibration step for the directory.

        The result of running this method will be to process all of the images
        in the specified directory (and given binning) through all of the basic
        calibration steps.  The procedure is:
            * copy_raw() -- Make a copy of the raw data in a safe place
            * inspect_images() -- Make sure the relevant metadata is set
            * bias_combine() -- Combine the bias frames into a Calibration bias
            * bias_subtract() -- Subtract the bias & overscan from all frames
            * flat_combine() -- Combine flat fields of a given filter
            * divide_by_flat() -- Divide science frames by the appropriate flat
        """
        self.copy_raw()
        self.inspect_images()
        self.bias_combine()
        self.bias_subtract()
        self.flat_combine()
        self.divide_by_flat()
