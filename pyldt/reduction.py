# -*- coding: utf-8 -*-
#
#  This file is part of PyLDT.
#
#   This Source Code Form is subject to the terms of the Mozilla Public
#   License, v. 2.0. If a copy of the MPL was not distributed with this
#   file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 26-Oct-2020
#
#  @author: tbowers

"""PyLDT contains image calibration routines for LDT facility instruments

Lowell Discovery Telescope (Lowell Observatory: Flagstaff, AZ)
http://www.lowell.edu

The high-level image calibration routines in this module are designed for easy
and simple-minded calibration of images from the LDT's facility instruments.

These currently include the Large Monolithic Imager (LMI), and the DeVeny
Optical Spectrograph (formerly the KPNO White Spectrograph).

The top-level classes take in a directory of data and can process them using
class methods to produce calibrated data for use with the data analysis
software of your choosing.
"""

# Built-In Libraries
import datetime
import pathlib
import shutil
import sys
import warnings

# 3rd Party Libraries
import astropy.convolution
import astropy.modeling
import astropy.nddata
import astropy.stats
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
import ccdproc
import numpy as np
from tqdm import tqdm

# Internal Imports
from pyldt import utils

# Define API
__all__ = ["LMI", "DeVeny", "imcombine", "savetime", "trim_oscan", "wrap_trim_oscan"]


# Global Variables
PKG_NAME = f"PyLDT {'='*55}"  # For header metadata printing


class ImageDirectory:
    """Internal class, parent of LMI & DeVeny.
    Contains collective metadata for a single night's data images.  This
    parent class is modified for specific differences between LMI and DeVeny.

    Args:
        path (:TYPE:`str`)
            Path to the directory containing the images to be reduced.
        mem_limit (:TYPE:`float`), optional
            Memory limit for the image combination routine
            (Default: 8.192e9 bytes)
        debug (:TYPE:`bool`)
            Description.
        show_warnings (:TYPE:`bool`)
            Description.
    """

    def __init__(self, path, mem_limit=8.192e9, debug=True, show_warnings=False):
        # Settings that determine how the class functions
        self.debug = debug
        # Unless specified, suppress AstroPy warnings
        if not show_warnings:
            warnings.simplefilter("ignore", AstropyWarning)
            warnings.simplefilter("ignore", UserWarning)

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
        self.gratings = None
        # Generic filenames
        self.zerofn = "bias.fits"
        # Create Placeholder for initial ImageFileCollection for the directory
        self.icl = None

    def inspectimages(self, binning=None, deveny=False):
        """Inspect the images in the specified directory

        Inspects the images in the specified directory, and loads in the
        default BIASSEC and TRIMSEC values (if not specified at Class
        instantiation).  For folders of DeVeny data, also modifies the FILTREAR
        keyword and adds a GRAT_ID keyword containing the DVx name of the
        grating.

        :param binning: The CCD binning of the image
        :param deveny: Is this a directory of DeVeny data?
        :return: None
        """

        # Print a helpful statement that image inspection is happening
        if self.debug:
            print("Inspecting the images in this directory...")

        # Check that binning is set
        if binning is None:
            raise InputError("Binning not set.")
        if self.debug:
            print(f"Binning is: {binning.replace(' ','x')}")

        if deveny:
            # Break out key/value lists for gratings
            grating_ids = list(self.gratings.keys())
            grating_kwds = list(self.gratings.values())

        # Refresh the ImageFileCollection
        self.icl.refresh()

        # Set up a progress bar, so we can see how the process is going...
        prog_bar = tqdm(
            total=len(self.icl.files), unit="frame", unit_scale=False, colour="cyan"
        )

        # Loop through files,
        for ccd, fname in self.icl.ccds(ccdsum=binning, return_fname=True):
            # Check for empty trimsec/biassec attributes, pull from header
            try:
                if self.biassec is None:
                    self.biassec = ccd.header["biassec"]
                if self.trimsec is None:
                    self.trimsec = ccd.header["trimsec"]
            except KeyError as err:
                warnings.warn(str(err), RuntimeWarning)

            # If DeVeny, adjust the FILTREAR FITS keyword to make it play nice
            #   Also, create GRAT_ID keyword containing DVx grating ID
            if deveny:
                ccd.header["filtrear"] = ccd.header["filtrear"].split(" (")[0]
                grname = grating_ids[grating_kwds.index(ccd.header["grating"])]
                ccd.header.set("grat_id", grname, "Grating ID Name", after="grating")
            # Fix depricated FITS keyword
            if "RADECSYS" in ccd.header:
                ccd.header.rename_keyword("RADECSYS", "RADESYSa")
            ccd.write(self.path.joinpath(fname), overwrite=True)

            # Update the progress bar
            prog_bar.update(1)
        # Close the progress bar, end of loop
        prog_bar.close()

    def copy_raw(self, overwrite=False):
        """Copy raw FITS files to subdirectory 'raw' for safekeeping.
        If a directory containing the raw data is not extant, create it and copy
        all FITS files there as a backup.
        :return: None
        """

        raw_data = pathlib.Path(self.path, "raw")
        raw_data.mkdir(exist_ok=True)

        # Copy files to raw_data, overwrite if requested
        for img in sorted(self.path.glob(f"{self.prefix}.????.fits")):
            if (not raw_data.joinpath(img.name).exists()) or overwrite:
                print(f"Copying {img} to {raw_data}...")
                shutil.copy2(img, raw_data)

    def biascombine(
        self, binning=None, output="bias.fits", keep_trimmed=False, gain_correct=True
    ):
        """Finds and combines bias frames with the indicated binning

        :param binning:
        :param output:
        :param gain_correct:
        :return:
        """
        if binning is None:
            raise InputError("Binning not set.")
        if self.debug:
            print(
                "Trimming and combining bias frames with binning "
                f"{binning.replace(' ','x')} into {output}..."
            )

        # First, refresh the ImageFileCollection
        self.icl.refresh()

        # Set up a progress bar, so we can see how the process is going...
        bias_files = self.icl.files_filtered(imagetyp="bias")
        prog_bar = tqdm(
            total=len(bias_files), unit="frame", unit_scale=False, colour="#808080"
        )

        # Loop through files,
        for ccd, file_name in self.icl.ccds(
            ccdsum=binning, imagetyp="bias", bitpix=16, return_fname=True
        ):
            # Fit the overscan section, subtract it, then trim the image
            ccd = wrap_trim_oscan(ccd, gain_correct=gain_correct)

            # Update the header
            ccd.header["HISTORY"] = PKG_NAME
            ccd.header["HISTORY"] = "Trimmed bias saved: " + savetime()
            ccd.header["HISTORY"] = f"Original filename: {file_name}"

            # Save the result (suffix = 't'); delete the input file
            ccd.header = self.add_package_versions(ccd.header)
            ccd.write(
                self.path.joinpath(f"{file_name[:-5]}t{file_name[-5:]}"), overwrite=True
            )
            self.path.joinpath(file_name).unlink()

            # Update the progress bar
            prog_bar.update(1)
        # Close the progress bar, end of loop
        prog_bar.close()

        # Collect the trimmed biases
        t_bias_cl = ccdproc.ImageFileCollection(
            self.path, glob_include=f"{self.prefix}.*t.fits"
        )

        # If we have a fresh list of trimmed biases to work with...
        if t_bias_cl.files:
            if self.debug:
                print("Doing average combine now...")
            comb_bias = ccdproc.combine(
                [self.path.joinpath(fn) for fn in t_bias_cl.files],
                method="average",
                sigma_clip=True,
                sigma_clip_dev_func=astropy.stats.mad_std,
                mem_limit=self.mem_limit,
            )

            # Clean the NaN's
            comb_bias = self.clean_nans(comb_bias)

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
            comb_bias.write(self.path.joinpath(output), overwrite=True)
            if not keep_trimmed:
                for fname in t_bias_cl.files:
                    self.path.joinpath(fname).unlink()

    def bias_subtract(self, gain_correct=True):
        """

        gain_correct : `bool`
            Multiply by the CCD gain before returning?  [Default: True]


        :return:
        """
        if self.binning is None:
            raise InputError("Binning not set.")
        if self.debug:
            print("Subtracting bias from remaining images...")

        # Refresh the ImageFileCollection
        self.icl.refresh()

        # Load the appropriate bias frame to subtract
        if not self.path.joinpath(self.zerofn).is_file():
            self.biascombine(binning=self.binning)
        try:
            combined_bias = astropy.nddata.CCDData.read(self.path.joinpath(self.zerofn))
        except FileNotFoundError:
            # Just skip the bias subtraction
            print(f"Skipping bias subtraction for lack of {self.zerofn}")
            return

        # Set up a progress bar, so we can see how the process is going...
        prog_bar = tqdm(
            total=len(self.icl.files), unit="frame", unit_scale=False, colour="blue"
        )

        # Loop through files,
        for ccd, file_name in self.icl.ccds(
            ccdsum=self.binning, bitpix=16, return_fname=True
        ):
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
            ccd.write(
                self.path.joinpath(f"{file_name[:-5]}b{file_name[-5:]}"), overwrite=True
            )
            self.path.joinpath(file_name).unlink()

            # Update the progress bar
            prog_bar.update(1)
        # Close the progress bar, end of loop
        prog_bar.close()

    @staticmethod
    def add_package_versions(hdr):
        """Add or update the depedendent package versions

        Include the version information for dependent packages in the FITS
        headers for the purposes for debugging if something changes in the
        underlying infrastructure.  By comparing package version numbers,
        it may be possible to pinpoint when a change in a dependancy causes
        problems in this package's output.

        Parameters
        ----------
        hdr : :obj:`astropy.io.fits.Header`
            The FITS header to which to add/update package version info

        Returns
        -------
        :obj:`astropy.io.fits.Header`
            The updated FITS header
        """
        # Add Python, Astropy, CCDPROC, and Numpy version numbers
        hdr["VERSPYT"] = (
            ".".join([str(v) for v in sys.version_info[:3]]),
            "Python version",
        )
        hdr["VERSAST"] = (astropy.__version__, "Astropy version")
        hdr["VERSCCD"] = (ccdproc.__version__, "CCDPROC version")
        hdr["VERSNPY"] = (np.__version__, "Numpy version")

        return hdr

    @staticmethod
    def clean_nans(ccd: astropy.nddata.CCDData) -> astropy.nddata.CCDData:
        """Clean the NaN's from a CCDData object by interpolation

        This method performs a cleaning of NaN values in a ``CCDData`` object.
        The issue is not simply removing NaN's in the data attribute, but also
        adjusting the mask and uncertainty attributes to align with the
        cleaned data.

        The replacement algorithm is provided by
        :func:`astropy.convolution.interpolate_replace_nans`, used alongside
        a 2D Gaussian kernel with sigma = 1 pixel.  This effectively replaces
        NaN values with a smoothed average of the surrounding pixels.

        Parameters
        ----------
        ccd : :obj:`astropy.nddata.CCDData`
            The input ``CCDData`` object to be cleaned.

        Returns
        -------
        :obj:`astropy.nddata.CCDData`
            The resulting cleaned ``CCDData`` object.  The cleaning is done
            in place.
        """
        # Clean up the image by interpolating over NaN's:
        before_nan = (~np.isfinite(ccd.data)).sum()
        ccd.data = astropy.convolution.interpolate_replace_nans(
            ccd.data, kernel := astropy.convolution.Gaussian2DKernel(x_stddev=1)
        )

        # Update the image mask
        ccd.mask = ~np.isfinite(ccd.data)
        print(
            "   Number of initial / final NaN pixels:   "
            f"{before_nan} / {ccd.mask.sum()}"
        )

        # Update the image uncertainty by smoothing, too
        ccd.uncertainty.array = astropy.convolution.interpolate_replace_nans(
            ccd.uncertainty.array, kernel
        )

        # Return the updated CCDData object
        return ccd


class LMI(ImageDirectory):
    """Class call for a folder of LMI data to be calibrated.

    Args:
        path (:TYPE:`str`)
            Path to the directory containing the images to be reduced.
        biassec (:TYPE:`str`)
            The IRAF-style overscan region to be subtracted from each frame.
            If unspecified, use the values suggested in the LMI User Manual.
        trimsec (:TYPE:`str`)
            The IRAF-style image region to be retained in each frame.
            If unspecified, use the values suggested in the LMI User Manual.
        bin_factor (:TYPE:`int`)
            The binning factor used to create the image(s) to be processed.
            [Default: 2]
        mem_limit (:TYPE:`float`), optional
            Memory limit for the image combination routine
            (Default: 8.192e9 bytes)
    """

    def __init__(
        self,
        path,
        biassec=None,
        trimsec=None,
        bin_factor=2,
        mem_limit=8.192e9,
        **kwargs,
    ):
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
        self.icl = ccdproc.ImageFileCollection(
            self.path, glob_include=f"{self.prefix}.*.fits"
        )

    def process_all(self):
        """Process all of the images in this directory (with given binning)
        The result of running this method will be to process all of the images
        in the specified directory (and given binning) through all of the basic
        calibration steps.  The procedure is:
            * copy_raw() -- Make a copy of the raw data in a safe place
            * insepct_images() -- Make sure the relevant metadata is set
            * bias_combine() -- Combine the bias frames into a Calibration bias
            * bias_subtract() -- Subtract the bias & overscan from all frames
            * flat_combine() -- Combine flat fields of a given filter
            * divide_flat() -- Divide all science frames by the appropriate flat
        :return: None
        """
        self.copy_raw()
        self.inspect_images()
        self.bias_combine()
        self.bias_subtract()
        self.flat_combine()
        self.divide_flat()

    def inspect_images(self):
        """Checks that the relevant metadata is set
        Looks to ensure BIASSEC and TRIMSEC values are properly set
        :return: None
        """
        self.inspectimages(self.binning)

    def bias_combine(self, keep_trimmed=False, gain_correct=True):
        """Combine the bias frames in the directory with a given binning
        Basic emulation of IRAF's zerocombine.  Produces a combined bias image
        saved with the appropriate filename.
        :return: None
        """
        self.biascombine(
            self.binning,
            output=self.zerofn,
            keep_trimmed=keep_trimmed,
            gain_correct=gain_correct,
        )

    def flat_combine(self, keep_subtracted=False, keep_normalized=False):
        """Finds and combines flat frames with the indicated binning

        :return: None
        """

        # Load the list of bias-subtracted data frames -- check binning
        bsub_cl = ccdproc.ImageFileCollection(
            self.path, glob_include=f"{self.prefix}.*b.fits"
        )

        if not bsub_cl.files:
            print("Nothing to be done for flat_combine()!")
            return

        if self.debug:
            print("Normalizing flat field frames...")

        # Filter here to get # of images for progress bar
        flat_cl = bsub_cl.filter(
            ccdsum=self.binning, imagetyp="[a-z]+ flat", regex_match=True
        )
        # Set up a progress bar, so we can see how the process is going...
        prog_bar = tqdm(
            total=len(flat_cl.files), unit="frame", unit_scale=False, colour="yellow"
        )

        # Normalize flat field images by the mean value
        for ccd, flat_fn in flat_cl.ccds(return_fname=True):
            # Perform the division (in a NaN-safe manner)
            ccd = ccd.divide(np.nanmean(ccd), handle_meta="first_found")

            # Update the header
            ccd.header["HISTORY"] = "Normalized flat saved: " + savetime()
            ccd.header["HISTORY"] = f"Previous filename: {flat_fn}"

            # Save the result (suffix = 'n'); delete the input file
            ccd.header = self.add_package_versions(ccd.header)
            ccd.write(
                self.path.joinpath(f"{flat_fn[:-6]}n{flat_fn[-5:]}"), overwrite=True
            )
            if not keep_subtracted:
                self.path.joinpath(flat_fn).unlink()

            # Update the progress bar
            prog_bar.update(1)
        # Close the progress bar, end of loop
        prog_bar.close()

        # Load the list of normalized flat field images
        norm_cl = ccdproc.ImageFileCollection(
            self.path, glob_include=f"{self.prefix}.*n.fits"
        )
        if norm_cl.files:
            # Create a unique list of the filter collection found in this set
            filters = list(norm_cl.summary["filters"])
            unique_filters = sorted(list(set(filters)))

            # Combine flat field frames for each filt in unique_filters
            for filt in unique_filters:
                flats = norm_cl.files_filtered(filters=filt, include_path=True)

                print(f"Combining flats for filter {filt}...")
                cflat = ccdproc.combine(
                    flats,
                    method="average",
                    sigma_clip=True,
                    sigma_clip_dev_func=astropy.stats.mad_std,
                    mem_limit=self.mem_limit,
                )

                # Clean the NaN's
                cflat = self.clean_nans(cflat)

                # Add FITS keyword NCOMBINE and HISTORY
                cflat.header.set(
                    "ncombine", len(flats), "# of input images in combination"
                )
                cflat.header["HISTORY"] = PKG_NAME
                cflat.header["HISTORY"] = "Combined flat created: " + savetime()
                cflat.header["HISTORY"] = "Median combined " + f"{len(flats)} files:"
                for fname in flats:
                    # Remove the path portion of the filename for the HISTORY
                    cflat.header["HISTORY"] = fname[fname.rfind("/") + 1 :]

                # Build filename, save, remove input files
                flat_fn = f"flat_bin{self.bin_factor}_{filt}.fits"
                if self.debug:
                    print(f"Saving combined flat as {flat_fn}")
                cflat.header = self.add_package_versions(cflat.header)
                cflat.write(self.path.joinpath(flat_fn), overwrite=True)
                for fname in flats:
                    # Path name is already included
                    if not keep_normalized:
                        pathlib.Path(fname).unlink()

        else:
            print("No flats to be combined.")

    def divide_flat(self):
        """Divides all LMI science frames by the appropriate flat field image
        This method is LMI-specific, rather than being wrapper for a more
        general function.  Basic emulation of IRAF's ccdproc/flatcor function.
        :return: None
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

        # Check to be sure there are, indeed, flats...
        if flat_cl.files:
            # Loop through the filters present
            for filt in sorted(list(flat_cl.summary["filters"])):
                # Load in the combined flat for this filter
                if self.debug:
                    print(f"Dividing by combined flat for filter: {filt}")
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
                    ccdproc.flat_correct(ccd, combined_flat)

                    # Update the header
                    ccd.header["flatcor"] = True
                    ccd.header["HISTORY"] = PKG_NAME
                    ccd.header["HISTORY"] = "Flat-corrected image saved: " + savetime()
                    ccd.header["HISTORY"] = f"Divided by flat: {mflat_fn}"
                    ccd.header["HISTORY"] = f"Previous filename: {sci_fn}"

                    # Save the result (suffix = 'f'); delete the input file
                    ccd.header = self.add_package_versions(ccd.header)
                    ccd.write(
                        self.path.joinpath(f"{sci_fn[:-6]}f{sci_fn[-5:]}"),
                        overwrite=True,
                    )
                    self.path.joinpath(sci_fn).unlink()

                    # Update the progress bar
                    prog_bar.update(1)
                # Close the progress bar, end of loop
                prog_bar.close()


class DeVeny(ImageDirectory):
    """Class call for a folder of DeVeny data to be calibrated."""

    def __init__(self, path, biassec=None, trimsec=None, prefix=None, multilamp=False):
        """__init__: Initialize DeVeny class.
        Args:
           path (:TYPE:`str`)
                Path to the directory containing the images to be reduced.
           biassec (:TYPE:`str`)
                The IRAF-style overscan region to be subtracted from each frame.
                If unspecified, use the values suggested in the LMI User Manual.
           trimsec (:TYPE:`str`)
                The IRAF-style image region to be retained in each frame.
                If unspecified, use the values suggested in the LMI User Manual.
        """

        super().__init__(path)
        self.bin_factor = 1
        self.binning = f"{self.bin_factor} {self.bin_factor}"
        self.multilamp = multilamp

        # Set the BIASSEC and TRIMSEC appropriately
        self.biassec = "[2101:2144,5:512]" if biassec is None else biassec
        self.trimsec = "[54:  2096,5:512]" if trimsec is None else trimsec

        # File prefix -- DeVeny files prefix with the UT date
        if prefix is None:
            # Look at all the 20*.fits files in this directory, and choose
            # Note: This will need to be updated for the year 2100
            fitsfiles = self.path.glob("20*.????.fits")
            if fitsfiles:
                slashind = fitsfiles[0].rfind("/")
                self.prefix = fitsfiles[0][slashind + 1 : slashind + 9]
        else:
            self.prefix = prefix
        if self.debug:
            print(f"Directory prefix: {self.prefix}")

        # Define standard filenames
        self.zerofn = "bias.fits"

        # Define the gratings
        self.gratings = {
            "DV1": "150/5000",
            "DV2": "300/4000",
            "DV3": "300/6750",
            "DV4": "400/8500",
            "DV5": "500/5500",
            "DV6": "600/4900",
            "DV7": "600/6750",
            "DV8": "831/8000",
            "DV9": "1200/5000",
            "DV10": "2160/5000",
            "DVxx": "UNKNOWN",
        }

        self.icl = ccdproc.ImageFileCollection(
            self.path, glob_include=f"{self.prefix}.*.fits"
        )

    def process_all(self, no_flat=False, force_copy=False):
        """Process all of the images in this directory (with given binning)
        The result of running this method will be to process all of the images
        in the specified directory (and given binning) through all of the basic
        calibration steps.  The procedure is:
            * copy_raw() -- Make a copy of the raw data in a safe place
            * insepct_images() -- Make sure the relevant metadata is set
            * bias_combine() -- Combine the bias frames into a Calibration bias
            * bias_subtract() -- Subtract the bias & overscan from all frames
            * flat_combine() -- Combine flat fields of a given filter
            * divide_flat() -- Divide all science frames by the appropriate flat
        :return: None
        """
        self.copy_raw(overwrite=force_copy)
        self.inspect_images()
        self.bias_combine()
        self.bias_subtract()
        if not no_flat:
            self.flat_combine()

    def inspect_images(self):
        """Checks that the relevant metadata is set
        Looks to ensure BIASSEC and TRIMSEC values are properly set
        Also cleans up the FILTREAR FITS keyword (removes parenthetical value)
        :return: None
        """
        self.inspectimages(self.binning, deveny=True)

    def bias_combine(self):
        """Combine the bias frames in the directory with a given binning
        Basic emulation of IRAF's zerocombine.  Produces a combined bias image
        saved with the appropriate filename.
        :return: None
        """
        self.biascombine(self.binning, output=self.zerofn)

    def flat_combine(self):
        """

        :return:
        """
        if self.debug:
            print("Combining flats...")

        # Load the list of bias-subtracted data frames
        bsub_cl = ccdproc.ImageFileCollection(
            self.path, glob_include=f"{self.prefix}.*b.fits"
        )

        # Find just the flats
        # NOTE: When CCDPROC filters an ImgFileCol, the resulting filenames
        #       have the path attached.  This allows for generality, but will
        #       need to be accounted for.
        flats_cl = bsub_cl.filter(imagetyp="dome flat")

        # Check that we have any
        if flats_cl.files:
            # In case more than one grating was used (unlikely except eng)
            for grname in list(set(list(flats_cl.summary["grat_id"]))):
                # Filter the ImgFileColl to include only this grating
                gr_cl = flats_cl.filter(grat_id=grname)

                # In case more than one grating tilt angle was used (possible)
                for gra in list(set(list(gr_cl.summary["grangle"]))):
                    # Filter the ImgFileColl to include only this tilt
                    gra_cl = gr_cl.filter(grangle=gra)

                    # In case more than one order-blocking filter was used (???)
                    for filt in list(set(list(gra_cl.summary["filtrear"]))):
                        # Filter the ImgFileColl to include only this filter
                        filt_cl = gra_cl.filter(filtrear=filt)

                        # For engineering, possibly use different lamps for comp
                        if self.multilamp:
                            lamps = list(set(list(filt_cl.summary["comment"])))
                        else:
                            lamps = ["domelamp"]
                        if self.debug:
                            print(f"Flat lamps used: {lamps}")

                        for this_lamp in lamps:
                            if self.multilamp:
                                lamp_cl = filt_cl.filter(comment=this_lamp)
                                lname = (
                                    "_TRING" if this_lamp[0:3] == "Top" else "_FLOOD"
                                )
                            else:
                                lamp_cl = filt_cl
                                lname = ""

                            # Actually do the flat combining
                            cflat = ccdproc.combine(
                                lamp_cl.files,
                                method="average",
                                sigma_clip=True,
                                sigma_clip_dev_func=astropy.stats.mad_std,
                                mem_limit=self.mem_limit,
                            )

                            # Clean the NaN's
                            cflat = self.clean_nans(cflat)

                            # Add FITS keyword NCOMBINE and HISTORY
                            cflat.header.set(
                                "ncombine",
                                len(lamp_cl.files),
                                "# of input images in combination",
                            )
                            cflat.header["HISTORY"] = PKG_NAME
                            cflat.header["HISTORY"] = (
                                "Combined flat " + "created: " + savetime()
                            )
                            cflat.header[
                                "HISTORY"
                            ] = f"Median combined {len(lamp_cl.files)} files:"
                            for fname in lamp_cl.files:
                                # Note: These filenames have the path attached,
                                #       via the .filter() method of ImgFileCol.
                                # Include just the pathless filename.
                                cflat.header["HISTORY"] = fname[fname.rfind("/") + 1 :]

                            # Build filename, save, remove input files
                            flat_fn = f"flat_{grname}_{gra}_{filt}{lname}.fits"
                            if self.debug:
                                print(f"Saving combined flat as {flat_fn}")
                            cflat.header = self.add_package_versions(cflat.header)
                            cflat.write(self.path.joinpath(flat_fn), overwrite=True)
                            for fname in lamp_cl.files:
                                # Note: These filenames have the path already
                                #       attached, via the .filter() method of
                                #       ImgFileCol.
                                pathlib.Path(fname).unlink()
        else:
            print("No flats to be combined.")


# Error Classes
class Error(Exception):
    """Base class for exceptions in this module."""


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        super().__init__()
        self.message = message


# Non-class function definitions =============================================#
def imcombine(
    *infiles,
    inlist=None,
    outfn=None,
    del_input=False,
    combine=None,
    printstat=True,
    overwrite=True,
    returnccd=False,
    mem_limit=8.192e9,
):
    """Combine a collection of images
    This function (crudely) emulates the IRAF imcombine function.  Pass in a
    list of images to be combined, and the result is written to disk with an
    optionally specified output filename.
    :param infiles: `list`: List of filenames to combine
    :param inlist: `str`: Filename of text file listing images to be combined
    :param outfn: `str`: Filename to write combined image.  Default: append
                         '_comb' to first filename in the input list.
    :param del_input: `bool`: Delete the input files after combination.
                              Default: `false`
    :param combine: `str`: Combine method.  'median' (default), or 'mean'
    :param printstat: `bool`: Print image statistics to screen
    :param overwrite: `bool`: Overwrite the output file.  Default: True
    :return: None (Unless returnccd=True, then it returns the combined CCDData)
    """

    # Unpack the single-item tuple *infiles
    if len(infiles) > 0:
        (files,) = infiles
    else:
        files = []

    # Check for inputs
    if len(files) > 0 and inlist is not None:
        warnings.warn(
            "Only one of files or inlist may be specified, not both.", RuntimeWarning
        )
        print("Using the `inlist` for file combination.")

    # Read in the text list inlist, if specified
    if inlist is not None:
        with open(inlist, "r", encoding="utf-8") as f_obj:
            files = []
            for line in f_obj:
                files.append(pathlib.Path(line.rstrip()))
    # Ensure the files are pathlib.Path objects
    else:
        files = [pathlib.Path(fn) for fn in files]

    # Check that specified input files exist
    for fname in files:
        if not fname.is_file():
            raise FileNotFoundError(f"File {fname} does not exist.")

    # Determine combine method (default = 'median')
    if combine != "median" or combine != "mean":
        combine = "median"

    # Create an ImgFileColl using the input files
    file_cl = ccdproc.ImageFileCollection(filenames=files)

    if printstat:
        # Print out the statistics, for clarity
        for img, fname in file_cl.ccds(return_fname=True):
            mini, maxi, mean, stdv = utils.mmms(img)
            print(
                f"{fname}:: Min: {mini:.2f} Max: {maxi:.2f} "
                + f"Mean: {mean:.2f} Stddev: {stdv:.2f}"
            )

    # Check for proper file list
    if len(files) < 3:
        warnings.warn(
            "Proper combination requires at least three input images.  "
            "Proceeding regardless...",
            RuntimeWarning,
        )

    # Run the combination
    comb_ccd = ccdproc.combine(
        file_cl.files,
        method=combine,
        sigma_clip=True,
        sigma_clip_dev_func=astropy.stats.mad_std,
        mem_limit=mem_limit,
    )

    comb_ccd = ImageDirectory.clean_nans(comb_ccd)

    # Add FITS keyword NCOMBINE and add HISTORY
    comb_ccd.header.set(
        "ncombine", len(file_cl.files), "# of input images in combination"
    )
    comb_ccd.header["HISTORY"] = PKG_NAME
    comb_ccd.header["HISTORY"] = "Combined image created: " + savetime()
    comb_ccd.header["HISTORY"] = (
        f"{combine.title()} combined " + f"{len(file_cl.files)} files:"
    )
    for fname in file_cl.files:
        comb_ccd.header["HISTORY"] = (
            fname.name if isinstance(fname, pathlib.Path) else fname
        )

    # If returnccd is True, return now before thinking about saving.
    if returnccd:
        return comb_ccd

    # Build filename (if not specified in call), save, remove input files
    if outfn is None:
        outfn = f"{files[0][:-5]}_comb{files[0][-5:]}"
    print(f"Saving combined image as {outfn}")
    comb_ccd.header = ImageDirectory.add_package_versions(comb_ccd.header)
    comb_ccd.write(outfn, overwrite=overwrite)
    if del_input:
        for fname in files:
            fname.unlink()
    return None


def parse_lois_ampids(hdr):
    """parse_lois_ampids Parse the LOIS amplifier IDs

    LOIS is particular about how it records which amplifiers are used to read
    out the CCD.  Most of the time, users will use a single amplifier, whose ID
    is recorded in the 'AMPID' FITS keyword.  If, however, more than one
    amplifier is used, 'AMPID' is not present, and the amplifier combination
    must be reconstructed from the present 'AMPIDnn' keywords.

    Parameters
    ----------
    hdr : `astropy.io.fits.Header`
        The FITS header for which the amplifier IDs are to be parsed

    Returns
    -------
    `str`
        The amplifier designation(s) used
    """
    # Basic 1-amplifier case:
    if int(hdr["NUMAMP"]) == 1:
        return f"{hdr['AMPID'].strip()}"

    # Else, parse out all of the "AMPIDnn" keywords, join and return
    return "".join([val.strip() for kwd, val in hdr.items() if "AMPID" in kwd])


def savetime(local=False):
    """Shortcut to return the current UT timestamp in a useful form
    :return: `str`: UT timestamp in format %Y-%m-%d %H:%M:%S
    """
    if local:
        return f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} MST'
    return f'{datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UT'


def trim_oscan(ccd, biassec, trimsec, model=None):
    """Subtract the overscan region and trim image to desired size.
    The CCDPROC function subtract_overscan() expects the TRIMSEC of the image
    (the part you want to keep) to span the entirety of one dimension, with the
    BIASSEC (overscan section) being at the end of the other dimension.
    Both LMI and DeVeny have edge effects on all sides of their respective
    chips, and so the TRIMSEC and BIASSEC do not meet the expectations of
    subtract_overscan().
    Therefore, this function is a wrapper to first remove the undesired ROWS
    from top and bottom, then perform the subtract_overscan() fitting and
    subtraction, followed by trimming off the now-spent overscan region.
    Args:
        ccd (:TYPE:`internal link or datatype`)
            Description.
        biassec (:TYPE:`str`)
            Description.
        trimsec (:TYPE:`str`)
            Description.
        model (:TYPE:internal link or datatype`)
            Description.
    Returns:
        ccd (:TYPE:`internal link or datatype`)
            Description.
    """

    # Convert the FITS bias & trim sections into slice classes for use
    _, x_b = ccdproc.utils.slices.slice_from_string(biassec, fits_convention=True)
    y_t, x_t = ccdproc.utils.slices.slice_from_string(trimsec, fits_convention=True)

    # First trim off the top & bottom rows
    ccd = ccdproc.trim_image(ccd[y_t.start : y_t.stop, :])

    # Model & Subtract the overscan
    if model is None:
        # Chebyshev 1st order function
        model = astropy.modeling.models.Chebyshev1D(1)
    else:
        # Really nothing different here... figure out how to incorporate others
        model = astropy.modeling.models.Chebyshev1D(1)
    ccd = ccdproc.subtract_overscan(
        ccd, overscan=ccd[:, x_b.start : x_b.stop], median=True, model=model
    )

    # Trim the overscan & return
    return ccdproc.trim_image(ccd[:, x_t.start : x_t.stop])


def wrap_trim_oscan(ccd, gain_correct=True):
    """wrap_trim_oscan Wrap the trim_oscan() function to handle multiple amps

    This function will perform the magic of stitching together multi-amplifier
    reads.  There may be instrument-specific issues related to this, but it is
    likely that only LMI will ever bet read out in multi-amplifier mode.

    TODO: Whether here or somewhere else, should convert things to electrons
          via the GAIN.  Might not be necessary within the context of Roz, but
          will be necessary for science frame analysis with multiple amplifier
          reads.

    Parameters
    ----------
    ccd : `astropy.nddata.CCDData`
        The CCDData object upon which to operate
    gain_correct : `bool`
        Multiply by the CCD gain before returning?  [Default: True]

    Returns
    -------
    trimmed : `astropy.nddata.CCDData`
        The properly trimmed and overscan-subtracted CCDData object,
        optionally gain corrected
    """
    # Shorthand
    hdr = ccd.header

    # The "usual" case, pass-through from `trim_oscan()`, with optional
    #   gain correction
    if hdr["NUMAMP"] == 1:
        trimmed = trim_oscan(ccd, hdr["BIASSEC"], hdr["TRIMSEC"])
        if gain_correct and trimmed.unit == u.adu:
            trimmed = ccdproc.gain_correct(
                trimmed, hdr["GAIN"], gain_unit=u.electron / u.adu
            )
        return trimmed

    # The multi-amplifier case is a little more involved.
    # First, make an empty FLOAT array for the trim output
    float_array = np.zeros_like(ccd.data, dtype=float)

    # Use the individual amplifier BIAS and TRIM sections to process
    amp_nums = [kwd[-2:] for kwd in hdr.keys() if "AMPID" in kwd]
    for amp_num in amp_nums:
        # Totally hacking tweak of the situation for 2x2 binning:
        if "51:1585" in hdr[f"TRIM{amp_num}"]:
            hdr[f"TRIM{amp_num}"] = hdr[f"TRIM{amp_num}"].replace("51:1585", "51:1584")
            xstart = xstop = 1
            hdr["TRIMSEC"] = hdr["TRIMSEC"].replace("51:3121", "52:3120")
        elif "1586:3121" in hdr[f"TRIM{amp_num}"]:
            hdr[f"TRIM{amp_num}"] = hdr[f"TRIM{amp_num}"].replace(
                "1586:3121", "1587:3121"
            )
            xstart = xstop = -1
            hdr["TRIMSEC"] = hdr["TRIMSEC"].replace("51:3121", "52:3120")
        else:
            xstart = xstop = 0
        yrange, xrange = ccdproc.utils.slices.slice_from_string(
            hdr[f"TRIM{amp_num}"], fits_convention=True
        )
        chunk = trim_oscan(ccd, hdr[f"BIAS{amp_num}"], hdr[f"TRIM{amp_num}"])
        if gain_correct and ccd.unit == u.adu:
            chunk = ccdproc.gain_correct(
                chunk, hdr[f"GAIN_{amp_num}"], gain_unit=u.electron / u.adu
            )

        float_array[
            yrange.start : yrange.stop, xrange.start + xstart : xrange.stop + xstop
        ] = chunk.data

    ccd.data = float_array
    if gain_correct and ccd.unit == u.adu:
        ccd.unit = u.electron

    # Return the final trimmed image
    ytrim, xtrim = ccdproc.utils.slices.slice_from_string(
        hdr["TRIMSEC"], fits_convention=True
    )
    return ccdproc.trim_image(ccd[ytrim.start : ytrim.stop, xtrim.start : xtrim.stop])
