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
from __future__ import division, print_function, absolute_import
from datetime import datetime
import glob
import os
from pathlib import Path
import shutil
import warnings

# Numpy
import numpy as np

# Astropy and CCDPROC
from astropy.modeling import models
from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy.utils.exceptions import AstropyWarning
import ccdproc as ccdp
from ccdproc.utils.slices import slice_from_string

# Intrapackage
from .utils import *


class _Images:
    """Internal class, parent of LMI & DeVeny."""

    def __init__(self, path, debug=True, show_warnings=False):
        """__init__: Initialize the internal _Images class.
        Args:
            path (:TYPE:`str`)
                Path to the directory containing the images to be reduced.
            debug (:TYPE:`bool`)
                Description.
            show_warnings (:TYPE:`bool`)
                Description.
        """
        self.path = path
        self.debug = debug
        if self.debug:
            print(self.path)
        # Unless specified, suppress AstroPy warnings
        if not show_warnings:
            warnings.simplefilter('ignore', AstropyWarning)
            warnings.simplefilter('ignore', UserWarning)
        # Add attributes that need to be specified for the instrument
        self.biassec = None
        self.trimsec = None
        self.prefix = None
        self.bin_factor = None
        # Generic filenames
        self.zerofn = 'bias.fits'

    def copy_raw(self, overwrite=False):
        """Copy raw FITS files to subdirectory 'raw' for safekeeping.
        If a directory containing the raw data is not extant, create it and copy
        all FITS files there as a backup.
        :return:
        """
        raw_data = Path(self.path, 'raw')
        if not raw_data.exists():
            raw_data.mkdir(exist_ok=True)
            new_raw = True
        else:
            new_raw = False
        # Copy files to raw_data
        if new_raw or (not new_raw and overwrite):
            if self.debug:
                print(self.path + '/' + self.prefix + '.*.fits')
            for img in glob.iglob(self.path + '/' + self.prefix + '.*.fits'):
                print(f'Copying {img} to {raw_data}...')
                shutil.copy2(img, raw_data)

    def _biascombine(self, binning=None, output="bias.fits"):
        """Finds and combines bias frames with the indicated binning

        :param binning:
        :param output:
        :return:
        """
        if binning is None:
            raise
        if self.debug:
            print(binning, output)

        # First, build the file collection of images to be trimmed
        file_cl = ccdp.ImageFileCollection(
            self.path, glob_include=self.prefix + '.*.fits')
        # Loop through files,
        for ccd, file_name in file_cl.ccds(ccdsum=binning, bitpix=16,
                                           imagetyp='bias', return_fname=True):

            # Check for empty trimsec/biassec attributes, pull from header
            if self.biassec is None:
                self.biassec = ccd.header['biassec']
            if self.trimsec is None:
                self.trimsec = ccd.header['trimsec']

            # Fit the overscan section, subtract it, then trim the image
            ccd = _trim_oscan(ccd, self.biassec, self.trimsec)

            # Update the header
            ccd.header['HISTORY'] = 'Trimmed bias saved: ' + \
                                    f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UT '
            ccd.header['HISTORY'] = f'Original filename: {file_name}'

            # Save the result; delete the input file
            ccd.write(f'{file_name[:-5]}t{file_name[-5:]}', overwrite=True)
            os.remove(file_name)

        # Collect the trimmed biases
        t_bias_cl = ccdp.ImageFileCollection(
            self.path, glob_include=self.prefix + '*t.fits')

        # If we have a fresh list of trimmed biases to work with...
        if t_bias_cl.files:

            if self.debug:
                print(f"Combining bias frames with binning {binning}...")
            combined_bias = ccdp.combine(t_bias_cl.files, method='median',
                                         sigma_clip=True,
                                         sigma_clip_low_thresh=5,
                                         sigma_clip_high_thresh=5,
                                         sigma_clip_func=np.ma.median,
                                         sigma_clip_dev_func=mad_std,
                                         mem_limit=4e9)

            # Add FITS keyword BIASCOMB and add HISTORY
            combined_bias.header['biascomb'] = True
            combined_bias.header['HISTORY'] = 'Combined bias created: ' + \
                                              f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UT'
            for f in t_bias_cl.files:
                combined_bias.header['HISTORY'] = f

            # Save the result; delete the input files
            combined_bias.write(output, overwrite=True)
            for f in t_bias_cl.files:
                os.remove(f)

    def _biassubtract(self, binning=None, deveny=False):
        """

        :param binning:
        :return:
        """
        if binning is None:
            raise

        # First, build the file collection of images to be trimmed
        file_cl = ccdp.ImageFileCollection(
            self.path, glob_include=self.prefix + '.*.fits')

        # Load the appropriate bias frame to subtract
        if not os.path.isfile(self.zerofn):
            self.biascombine()
        combined_bias = CCDData.read(self.zerofn)

        # Loop through files,
        for ccd, file_name in file_cl.ccds(ccdsum=binning, bitpix=16,
                                           return_fname=True):
            if self.debug:
                print(file_name, ccd.header['NAXIS2'], ccd.header['NAXIS1'])

            # Check for empty trimsec/biassec attributes, pull from header
            if self.biassec is None:
                self.biassec = ccd.header['biassec']
            if self.trimsec is None:
                self.trimsec = ccd.header['trimsec']

            # Fit the overscan section, subtract it, then trim the image
            ccd = _trim_oscan(ccd, self.biassec, self.trimsec)

            # Subtract master bias
            ccd = ccdp.subtract_bias(ccd, combined_bias)

            # If DeVeny, adjust the FILTREAR FITS keyword to make it play nice
            if deveny:
                if len(ccd.header['filtrear']) == 9:
                    ccd.header['filtrear'] = ccd.header['filtrear'][0:5]

            # Update the header
            ccd.header['HISTORY'] = 'Bias-subtracted image saved: ' + \
                                    f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UT'
            ccd.header['HISTORY'] = f'Subtracted bias: {self.zerofn}'
            ccd.header['HISTORY'] = f'Original filename: {file_name}'

            # Save the result; delete input file
            ccd.write(f'{file_name[:-5]}b{file_name[-5:]}', overwrite=True)
            os.remove(file_name)


class LMI(_Images):
    """Class call for a folder of LMI data to be reduced.

    """

    def __init__(self, path, biassec=None, trimsec=None, bin_factor=2):
        """__init__: Initialize LMI class.
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
        _Images.__init__(self, path)
        self.bin_factor = int(bin_factor)
        self.binning = f'{self.bin_factor} {self.bin_factor}'

        # Set the BIASSEC and TRIMSEC appropriately FOR 2x2 BINNING
        if self.bin_factor == 2:
            self.biassec = '[3100:3124, 3:3079]' if biassec is None else biassec
            self.trimsec = '[30:3094,   3:3079]' if trimsec is None else trimsec
        else:
            self.biassec = biassec
            self.trimsec = trimsec

        # File prefix
        self.prefix = 'lmi'
        # Define standard filenames
        self.zerofn = f'bias_bin{self.bin_factor}.fits'

    def bias_combine(self):
        """

        :return:
        """
        self._biascombine(self.binning, output=self.zerofn)

    def bias_subtract(self):
        """

        :return:
        """
        self._biassubtract(self.binning)

    def flat_combine(self):
        """Finds and combines flat frames with the indicated binning

        :return:
        """

        # Load the list of bias-subtracted data frames
        bsub_cl = ccdp.ImageFileCollection(
            self.path, glob_include=self.prefix + '.*b.fits')

        # Normalize flat field images by the mean value
        for flat_type in ['sky flat', 'dome flat']:
            for ccd, flat_fn in bsub_cl.ccds(ccdsum=self.binning,
                                             imagetyp=flat_type,
                                             return_fname=True):
                # Perform the division
                ccd = ccd.divide(np.mean(ccd), handle_meta='first_found')

                # Update the header
                ccd.header['HISTORY'] = 'Normalized flat saved: ' + \
                                        f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UT'
                ccd.header['HISTORY'] = f'Previous filename: {flat_fn}'

                # Write out the file to a '*n.fits'; delete input file
                ccd.write(f'{flat_fn[:-6]}n{flat_fn[-5:]}', overwrite=True)
                os.remove(flat_fn)

        # Load the list of normalized flat field images
        norm_cl = ccdp.ImageFileCollection(
            self.path, glob_include=self.prefix + '.*n.fits')
        if norm_cl.files:

            # List the collection of filters found in this set
            filters = list(norm_cl.summary['filters'])

            # Determine unique, and produce a list to iterate on
            unique_filters = list(set(filters))

            # Combine for each filt in unique_filters
            for filt in unique_filters:

                flats = norm_cl.files_filtered(filters=filt,
                                               include_path=True)

                print(f"Combining flats for filter {filt}...")
                combined_flat = ccdp.combine(flats, method='median',
                                             sigma_clip=True,
                                             sigma_clip_low_thresh=5,
                                             sigma_clip_high_thresh=5,
                                             sigma_clip_func=np.ma.median,
                                             sigma_clip_dev_func=mad_std,
                                             mem_limit=4e9)

                # Add FITS keyword FLATCOMB and add HISTORY
                combined_flat.header['flatcomb'] = True
                combined_flat.header['HISTORY'] = 'Combined flat created: ' + \
                                                  f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UT'
                for fn in flats:
                    combined_flat.header['HISTORY'] = fn

                # Make a filename to save, save, remove input files
                flat_fn = f'flat_bin{self.bin_factor}_{filt}.fits'
                if self.debug:
                    print(f'Saving combined flat as {flat_fn}')
                combined_flat.write(flat_fn, overwrite=True)
                for fn in flats:
                    os.remove(fn)

        else:
            print("No flats to be combined.")

    def divide_flat(self):
        """Divides all LMI science frames by the appropriate flat field image
        This method is LMI-specific, rather than being wrapper for a more
        general function.
        :return:
        """

        # Load the list of master flats and bias-subtracted data frames
        flat_cl = ccdp.ImageFileCollection(
            self.path, glob_include=f'flat_bin{self.bin_factor}_*.fits')
        sci_cl = ccdp.ImageFileCollection(
            self.path, glob_include=self.prefix + '*b.fits')

        # Check to be sure there are, indeed, flats...
        if flat_cl.files:
            # Loop through the filters present
            for filt in list(flat_cl.summary['filters']):

                if self.debug:
                    print(f'Dividing by master flat for filter: {filt}')
                master_flat, mflat_fn = next(
                    flat_cl.ccds(ccdsum=self.binning, filters=filt,
                                 return_fname=True))

                # Loop through the science frames to correct
                for ccd, sci_fn in sci_cl.ccds(ccdsum=self.binning,
                                               filters=filt,
                                               return_fname=True):

                    if self.debug:
                        print(f'Flat correcting file {sci_fn}')

                    # Divide by master flat
                    ccdp.flat_correct(ccd, master_flat, add_keyword=True)

                    # Update the header
                    ccd.header['flatcor'] = True
                    ccd.header['HISTORY'] = 'Flat-corrected image saved: ' + \
                                            f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UT'
                    ccd.header['HISTORY'] = f'Divided by flat: {mflat_fn}'
                    ccd.header['HISTORY'] = f'Previous filename: {sci_fn}'

                    # Write out the file to a '*f.fits', remove input file
                    ccd.write(f'{sci_fn[:-6]}f{sci_fn[-5:]}', overwrite=True)
                    os.remove(sci_fn)

    def process_all(self):
        """

        :return:
        """
        self.copy_raw()
        self.bias_combine()
        self.bias_subtract()
        self.flat_combine()
        self.divide_flat()


class DeVeny(_Images):
    """Class call for a folder of DeVeny data to be reduced.

    """

    def __init__(self, path, biassec=None, trimsec=None, prefix=None,
                 multilamp=False):
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
        _Images.__init__(self, path)
        self.bin_factor = 1
        self.binning = f'{self.bin_factor} {self.bin_factor}'
        self.multilamp = multilamp

        # Set the BIASSEC and TRIMSEC appropriately
        self.biassec = '[2101:2144,5:512]' if biassec is None else biassec
        self.trimsec = '[54:  2096,5:512]' if trimsec is None else trimsec

        # File prefix -- DeVeny files prefix with the UT date
        if prefix is None:
            # Look at all the 20*.fits files in this directory, and choose
            fitsfiles = glob.glob(self.path + '/' + '20*.fits')
            if fitsfiles:
                slashind = fitsfiles[0].rfind('/')
                self.prefix = fitsfiles[0][slashind + 1:slashind + 9]
        else:
            self.prefix = prefix
        print(self.prefix)
        # Define standard filenames
        self.zerofn = 'bias.fits'

        # Define the gratings
        self.DV1 = "150/5000"
        self.DV2 = "300/4000"
        self.DV3 = "300/6750"
        self.DV4 = "400/8500"
        self.DV5 = "500/5500"
        self.DV6 = "600/4900"
        self.DV7 = "600/6750"
        self.DV8 = "831/8000"
        self.DV9 = "1200/5000"
        self.DV10 = "2160/5000"

    def bias_combine(self):
        """

        :return:
        """
        self._biascombine(self.binning, output=self.zerofn)

    def bias_subtract(self):
        """

        :return:
        """
        self._biassubtract(self.binning, deveny=True)

    def flat_combine(self):
        """

        :return:
        """
        if self.debug:
            print("Combining flats...")

        # Load the list of bias-subtracted data frames
        bsub_cl = ccdp.ImageFileCollection(
            self.path, glob_include=self.prefix + '.*b.fits')

        # Find just the flats
        flats_cl = bsub_cl.filter(imagetyp="dome flat")

        # Check that we have any
        if flats_cl.files:

            # Determine which grating(s) are included in this directory
            gratings = list(set(list(flats_cl.summary['grating'])))

            # In case more than one grating was used (unlikely except engineering)
            for g in gratings:

                grname = None
                if g == self.DV1:
                    grname = 'DV1'
                elif g == self.DV2:
                    grname = 'DV2'
                elif g == self.DV3:
                    grname = 'DV3'
                elif g == self.DV4:
                    grname = 'DV4'
                elif g == self.DV5:
                    grname = 'DV5'
                elif g == self.DV6:
                    grname = 'DV6'
                elif g == self.DV7:
                    grname = 'DV7'
                elif g == self.DV8:
                    grname = 'DV8'
                elif g == self.DV9:
                    grname = 'DV9'
                elif g == self.DV10:
                    grname = 'DV10'

                # Filter the ImageFileCollection to include only this grating
                gr_cl = flats_cl.filter(grating=g)

                # Determine which grating tilt angles were used
                grangles = list(set(list(gr_cl.summary['grangle'])))

                # In case more than one grating tilt angle was used (possible)
                for gra in grangles:

                    # Filter the ImageFileCollection to include only this tilt
                    gra_cl = gr_cl.filter(grangle=gra)

                    # Determine which order-blocking filters were used
                    filtrears = list(set(list(gra_cl.summary['filtrear'])))

                    # In case more than one order-blocking filter was used (???)
                    for filt in filtrears:

                        # Filter the ImageFileCollection to include only this filter
                        filt_cl = gra_cl.filter(filtrear=filt)

                        # For engineering, possible use different lamps for comp
                        if self.multilamp:
                            lamps = list(set(list(filt_cl.summary['comment'])))
                        else:
                            lamps = []

                        print(lamps)

                        for lamp in lamps:

                            if self.multilamp:
                                lamp_cl = filt_cl.filter(comment=lamp)
                                if lamp[0:3] == 'Top':
                                    lname = '_TRING'
                                else:
                                    lname = '_FLOOD'
                            else:
                                lamp_cl = filt.cl
                                lname = ''

                            comb_flat = ccdp.combine(lamp_cl.files,
                                                     method='median',
                                                     sigma_clip=True,
                                                     sigma_clip_low_thresh=5,
                                                     sigma_clip_high_thresh=5,
                                                     sigma_clip_func=np.ma.median,
                                                     sigma_clip_dev_func=mad_std,
                                                     mem_limit=4e9)

                            # Add FITS keyword FLATCOMB and add HISTORY
                            comb_flat.header['flatcomb'] = True
                            comb_flat.header[
                                'HISTORY'] = 'Combined flat created: ' + \
                                             f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UT'
                            for fn in lamp_cl.files:
                                comb_flat.header['HISTORY'] = fn

                            # Make a filename to save, save, remove input files
                            flat_fn = f'flat_{grname}_{gra}_{filt}{lname}.fits'
                            if self.debug:
                                print(f'Saving combined flat as {flat_fn}')
                            comb_flat.write(flat_fn, overwrite=True)
                            for fn in lamp_cl.files:
                                os.remove(fn)
        else:
            print("No flats to be combined.")

    def process_all(self):
        """

        :return:
        """
        self.copy_raw()
        self.bias_combine()
        self.bias_subtract()
        self.flat_combine()


def _trim_oscan(ccd, biassec, trimsec, model=None):
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
    yb, xb = slice_from_string(biassec, fits_convention=True)
    yt, xt = slice_from_string(trimsec, fits_convention=True)

    # First trim off the top & bottom rows
    ccd = ccdp.trim_image(ccd[yt.start:yt.stop, :], add_keyword=True)

    # Model & Subtract the overscan
    if model is None:
        model = models.Chebyshev1D(1)  # Chebyshev 1st order function
    else:
        model = models.Chebyshev1D(1)  # Figure out how to incorporate others
    ccd = ccdp.subtract_overscan(ccd, overscan=ccd[:, xb.start:xb.stop],
                                 median=True, model=model, add_keyword=True)

    # Trim the overscan & return
    return ccdp.trim_image(ccd[:, xt.start:xt.stop], add_keyword=True)


def imcombine(files, del_input=False, median=False, mean=False, printstat=True):
    """

    :param files:
    :param del_input:
    :param median:
    :param mean:
    :return:
    """

    # Check for proper file list
    if len(files) < 3:
        print("Script requires start and stop image #s to combine.")
        raise Exception()
    for f in files:
        if not os.path.isfile(f):
            raise Exception()

    # Determine combine method
    if median is True:
        combmethod = 'median'
    elif mean is True:
        combmethod = 'mean'
    else:
        combmethod = 'median'

    # Let's get combining!
    file_cl = ccdp.ImageFileCollection(filenames=files)

    if printstat:
        # Print out the statistics, for clarity
        for img, fn in file_cl.ccds(return_fname=True):
            mini, maxi, mean, stdv = mmms(img)
            print(
                f'{fn} Min: {mini:.2f} Max: {maxi:.2f} Mean: {mean:.2f} Stddev: {stdv:.2f}')

    comb_img = ccdp.combine(file_cl.files, method=combmethod, sigma_clip=True,
                            sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                            sigma_clip_func=np.ma.median,
                            sigma_clip_dev_func=mad_std, mem_limit=4e9)

    comb_fn = f'{files[0][:-5]}_comb{files[0][-5:]}'

    comb_img.header['combined'] = True
    comb_img.header['n_comb'] = len(file_cl.files)
    for i, fn in enumerate(file_cl.files):
        comb_img.header[f'comb{i + 1:04}'] = fn

    comb_img.write(comb_fn, overwrite=True)

    if del_input:
        for f in files:
            os.remove(f)
    pass


def main():
    """
    This is the main body function.
    """
    pass


if __name__ == "__main__":
    main()
