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


class _ImageDirectory:
    """Internal class, parent of LMI & DeVeny.
    Contains collective metadata for a single night's data images.  This
    parent class is modified for specific differences between LMI and DeVeny.
    """

    def __init__(self, path, debug=True, show_warnings=False):
        """__init__: Initialize the internal _ImageDirectory class.
        Args:
            path (:TYPE:`str`)
                Path to the directory containing the images to be reduced.
            debug (:TYPE:`bool`)
                Description.
            show_warnings (:TYPE:`bool`)
                Description.
        """
        # Settings that determine how the class functions
        self.debug = debug
        # Unless specified, suppress AstroPy warnings
        if not show_warnings:
            warnings.simplefilter('ignore', AstropyWarning)
            warnings.simplefilter('ignore', UserWarning)

        # Metadata related to all files in this directory
        self.path = path
        if self.debug:
            print(self.path)
        # Attributes that need to be specified for the instrument
        self.biassec = None
        self.trimsec = None
        self.prefix = None
        self.bin_factor = None
        self.binning = None
        self.gratings = None
        # Generic filenames
        self.zerofn = 'bias.fits'
        # Create Placeholder for initial ImageFileCollection for the directory
        self._file_cl = None

    def _inspectimages(self, binning=None, deveny=False):
        """

        :param binning:
        :param deveny:
        :return:
        """
        # Check that binning is set
        if binning is None:
            raise
        if self.debug:
            print(f'Binning is: {binning}')

        if deveny:
            # Break out key/value lists for gratings
            grating_ids = list(self.gratings.keys())
            grating_kwds = list(self.gratings.values())

        # Refresh the ImageFileCollection
        self._file_cl.refresh()

        # Loop through files,
        for ccd, fname in self._file_cl.ccds(ccdsum=binning, return_fname=True):
            # Check for empty trimsec/biassec attributes, pull from header
            if self.biassec is None:
                self.biassec = ccd.header['biassec']
            if self.trimsec is None:
                self.trimsec = ccd.header['trimsec']
            # If DeVeny, adjust the FILTREAR FITS keyword to make it play nice
            #   Also, create GRAT_ID keyword containing DVx grating ID
            if deveny:
                if len(ccd.header['filtrear']) == 9:
                    ccd.header['filtrear'] = ccd.header['filtrear'][0:5]
                grname = grating_ids[grating_kwds.index(ccd.header['grating'])]
                ccd.header.set('grat_id', grname, 'Grating ID Name',
                               after='grating')
                ccd.write(f'{self.path}/{fname}', overwrite=True)

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
        if new_raw or overwrite:
            pattern = self.path + '/' + self.prefix + '.*.fits'
            if self.debug:
                print(pattern)
            for img in glob.iglob(pattern):
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

        # First, refresh the ImageFileCollection
        self._file_cl.refresh()

        # Loop through files,
        for ccd, file_name in self._file_cl.ccds(ccdsum=binning,
                                                 bitpix=16,
                                                 imagetyp='bias',
                                                 return_fname=True):
            # Fit the overscan section, subtract it, then trim the image
            ccd = _trim_oscan(ccd, self.biassec, self.trimsec)

            # Update the header
            ccd.header['HISTORY'] = 'Trimmed bias saved: ' + _savetime()
            ccd.header['HISTORY'] = f'Original filename: {file_name}'

            # Save the result (suffix = 't'); delete the input file
            ccd.write(f'{self.path}/{file_name[:-5]}t{file_name[-5:]}',
                      overwrite=True)
            os.remove(f'{self.path}/{file_name}')

        # Collect the trimmed biases
        t_bias_cl = ccdp.ImageFileCollection(
            self.path, glob_include=f'{self.prefix}.*t.fits')

        # If we have a fresh list of trimmed biases to work with...
        if t_bias_cl.files:

            if self.debug:
                print(f"Combining bias frames with binning {binning}...")
            comb_bias = ccdp.combine(
                [f'{self.path}/{fn}' for fn in t_bias_cl.files],
                method='median',
                sigma_clip=True,
                sigma_clip_low_thresh=5,
                sigma_clip_high_thresh=5,
                sigma_clip_func=np.ma.median,
                sigma_clip_dev_func=mad_std,
                mem_limit=4e9)

            # Add FITS keyword BIASCOMB and add HISTORY
            comb_bias.header['biascomb'] = True
            comb_bias.header['HISTORY'] = 'Combined bias created: ' + \
                                          _savetime()

            for f in t_bias_cl.files:
                comb_bias.header['HISTORY'] = f

            # Save the result; delete the input files
            comb_bias.write(f'{self.path}/{output}', overwrite=True)
            for f in t_bias_cl.files:
                os.remove(f'{self.path}/{f}')

    def bias_subtract(self):
        """

        :return:
        """
        if self.binning is None:
            raise

        # Refresh the ImageFileCollection
        self._file_cl.refresh()

        # Load the appropriate bias frame to subtract
        if not os.path.isfile(f'{self.path}/{self.zerofn}'):
            self._biascombine(binning=self.binning)
        combined_bias = CCDData.read(f'{self.path}/{self.zerofn}')

        # Loop through files,
        for ccd, file_name in self._file_cl.ccds(ccdsum=self.binning,
                                                 bitpix=16,
                                                 return_fname=True):
            if self.debug:
                print(file_name, ccd.header['NAXIS2'], ccd.header['NAXIS1'])

            # Fit the overscan section, subtract it, then trim the image
            ccd = _trim_oscan(ccd, self.biassec, self.trimsec)

            # Subtract master bias
            ccd = ccdp.subtract_bias(ccd, combined_bias)

            # Update the header
            ccd.header['HISTORY'] = 'Bias-subtracted image saved: ' + \
                                    _savetime()
            ccd.header['HISTORY'] = f'Subtracted bias: {self.zerofn}'
            ccd.header['HISTORY'] = f'Original filename: {file_name}'

            # Save the result (suffix = 'b'); delete input file
            ccd.write(f'{self.path}/{file_name[:-5]}b{file_name[-5:]}',
                      overwrite=True)
            os.remove(f'{self.path}/{file_name}')


class LMI(_ImageDirectory):
    """Class call for a folder of LMI data to be calibrated.

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
        _ImageDirectory.__init__(self, path)
        self.bin_factor = int(bin_factor)
        self.binning = f'{self.bin_factor} {self.bin_factor}'

        # Set the BIASSEC and TRIMSEC appropriately FOR 2x2 BINNING
        if self.bin_factor == 2:
            self.biassec = '[3100:3124, 3:3079]' if biassec is None else biassec
            self.trimsec = '[30:3094,   3:3079]' if trimsec is None else trimsec
        else:
            self.biassec = biassec
            self.trimsec = trimsec

        # Define file prefix & standard filenames
        self.prefix = 'lmi'
        self.zerofn = f'bias_bin{self.bin_factor}.fits'

        # Load initial ImageFileCollection
        self._file_cl = ccdp.ImageFileCollection(
            self.path, glob_include=f'{self.prefix}.*.fits')

    def process_all(self):
        """Process all of the images in this directory (with given binning)
        The result of running this method will be to process all of the images
        in the specified directory (and given binning) through all of the basic
        calibration steps.  The procedure is:
            * copy_raw() -- Make a copy of the raw data in a safe place
            * insepct_images() -- Make sure the relevant metadata is set
            * bias_combine() -- Combine the bias frames into a master bias
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
        self._inspectimages(self.binning)

    def bias_combine(self):
        """Combine the bias frames in the directory with a given binning
        Basic emulation of IRAF's zerocombine.  Produces a combined bias image
        saved with the appropriate filename.
        :return: None
        """
        self._biascombine(self.binning, output=self.zerofn)

    def flat_combine(self):
        """Finds and combines flat frames with the indicated binning

        :return: None
        """

        # Load the list of bias-subtracted data frames
        bsub_cl = ccdp.ImageFileCollection(
            self.path, glob_include=f'{self.prefix}.*b.fits')

        # Normalize flat field images by the mean value
        for flat_type in ['sky flat', 'dome flat']:
            for ccd, flat_fn in bsub_cl.ccds(ccdsum=self.binning,
                                             imagetyp=flat_type,
                                             return_fname=True):
                # Perform the division
                ccd = ccd.divide(np.mean(ccd), handle_meta='first_found')

                # Update the header
                ccd.header['HISTORY'] = 'Normalized flat saved: ' + _savetime()
                ccd.header['HISTORY'] = f'Previous filename: {flat_fn}'

                # Save the result (suffix = 'n'); delete the input file
                ccd.write(f'{self.path}/{flat_fn[:-6]}n{flat_fn[-5:]}',
                          overwrite=True)
                os.remove(f'{self.path}/{flat_fn}')

        # Load the list of normalized flat field images
        norm_cl = ccdp.ImageFileCollection(
            self.path, glob_include=f'{self.prefix}.*n.fits')
        if norm_cl.files:

            # Create a unique list of the filter collection found in this set
            filters = list(norm_cl.summary['filters'])
            unique_filters = list(set(filters))

            # Combine flat field frames for each filt in unique_filters
            for filt in unique_filters:

                flats = norm_cl.files_filtered(filters=filt,
                                               include_path=True)

                print(f"Combining flats for filter {filt}...")
                cflat = ccdp.combine(flats,
                                     method='median',
                                     sigma_clip=True,
                                     sigma_clip_low_thresh=5,
                                     sigma_clip_high_thresh=5,
                                     sigma_clip_func=np.ma.median,
                                     sigma_clip_dev_func=mad_std,
                                     mem_limit=4e9)

                # Add FITS keyword FLATCOMB and add HISTORY
                cflat.header['flatcomb'] = True
                cflat.header['HISTORY'] = 'Combined flat created: ' + \
                                          _savetime()
                for fn in flats:
                    # Remove the path portion of the filename for the HISTORY
                    cflat.header['HISTORY'] = fn[fn.rfind('/') + 1:]

                # Build filename, save, remove input files
                flat_fn = f'flat_bin{self.bin_factor}_{filt}.fits'
                if self.debug:
                    print(f'Saving combined flat as {flat_fn}')
                cflat.write(f'{self.path}/{flat_fn}', overwrite=True)
                for fn in flats:
                    # Path name is already included
                    os.remove(f'{fn}')

        else:
            print("No flats to be combined.")

    def divide_flat(self):
        """Divides all LMI science frames by the appropriate flat field image
        This method is LMI-specific, rather than being wrapper for a more
        general function.  Basic emulation of IRAF's ccdproc/flatcor function.
        :return: None
        """

        # Load the list of master flats and bias-subtracted data frames
        flat_cl = ccdp.ImageFileCollection(
            self.path, glob_include=f'flat_bin{self.bin_factor}_*.fits')
        sci_cl = ccdp.ImageFileCollection(
            self.path, glob_include=f'{self.prefix}.*b.fits')

        # Check to be sure there are, indeed, flats...
        if flat_cl.files:
            # Loop through the filters present
            for filt in list(flat_cl.summary['filters']):

                # Load in the master flat for this filter
                if self.debug:
                    print(f'Dividing by master flat for filter: {filt}')
                master_flat, mflat_fn = next(flat_cl.ccds(ccdsum=self.binning,
                                                          filters=filt,
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
                                            _savetime()
                    ccd.header['HISTORY'] = f'Divided by flat: {mflat_fn}'
                    ccd.header['HISTORY'] = f'Previous filename: {sci_fn}'

                    # Save the result (suffix = 'f'); delete the input file
                    ccd.write(f'{self.path}/{sci_fn[:-6]}f{sci_fn[-5:]}',
                              overwrite=True)
                    os.remove(f'{self.path}/{sci_fn}')


class DeVeny(_ImageDirectory):
    """Class call for a folder of DeVeny data to be calibrated.

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
        _ImageDirectory.__init__(self, path)
        self.bin_factor = 1
        self.binning = f'{self.bin_factor} {self.bin_factor}'
        self.multilamp = multilamp

        # Set the BIASSEC and TRIMSEC appropriately
        self.biassec = '[2101:2144,5:512]' if biassec is None else biassec
        self.trimsec = '[54:  2096,5:512]' if trimsec is None else trimsec

        # File prefix -- DeVeny files prefix with the UT date
        if prefix is None:
            # Look at all the 20*.fits files in this directory, and choose
            # Note: This will need to be updated for the year 2100
            fitsfiles = glob.glob(self.path + '/' + '20*.fits')
            if fitsfiles:
                slashind = fitsfiles[0].rfind('/')
                self.prefix = fitsfiles[0][slashind + 1:slashind + 9]
        else:
            self.prefix = prefix
        if self.debug:
            print(f'Directory prefix: {self.prefix}')
        # Define standard filenames
        self.zerofn = 'bias.fits'

        # Define the gratings
        self.gratings = {"DV1": "150/5000",
                         "DV2": "300/4000",
                         "DV3": "300/6750",
                         "DV4": "400/8500",
                         "DV5": "500/5500",
                         "DV6": "600/4900",
                         "DV7": "600/6750",
                         "DV8": "831/8000",
                         "DV9": "1200/5000",
                         "DV10": "2160/5000",
                         "DVxx": "UNKNOWN"}

        self._file_cl = ccdp.ImageFileCollection(
            self.path, glob_include=f'{self.prefix}.*.fits')

    def process_all(self):
        """Process all of the images in this directory (with given binning)
        The result of running this method will be to process all of the images
        in the specified directory (and given binning) through all of the basic
        calibration steps.  The procedure is:
            * copy_raw() -- Make a copy of the raw data in a safe place
            * insepct_images() -- Make sure the relevant metadata is set
            * bias_combine() -- Combine the bias frames into a master bias
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

    def inspect_images(self):
        """Checks that the relevant metadata is set
        Looks to ensure BIASSEC and TRIMSEC values are properly set
        Also cleans up the FILTREAR FITS keyword (removes parenthetical value)
        :return: None
        """
        self._inspectimages(self.binning, deveny=True)

    def bias_combine(self):
        """Combine the bias frames in the directory with a given binning
        Basic emulation of IRAF's zerocombine.  Produces a combined bias image
        saved with the appropriate filename.
        :return: None
        """
        self._biascombine(self.binning, output=self.zerofn)

    def flat_combine(self):
        """

        :return:
        """
        if self.debug:
            print("Combining flats...")

        # Load the list of bias-subtracted data frames
        bsub_cl = ccdp.ImageFileCollection(
            self.path, glob_include=f'{self.prefix}.*b.fits')

        # Find just the flats
        # NOTE: When CCDPROC filters an ImgFileCol, the resulting filenames
        #       have the path attached.  This allows for generality, but will
        #       need to be accounted for.
        flats_cl = bsub_cl.filter(imagetyp="dome flat")

        # Check that we have any
        if flats_cl.files:

            # In case more than one grating was used (unlikely except eng)
            for grname in list(set(list(flats_cl.summary['grat_id']))):

                # Filter the ImgFileColl to include only this grating
                gr_cl = flats_cl.filter(grat_id=grname)

                # In case more than one grating tilt angle was used (possible)
                for gra in list(set(list(gr_cl.summary['grangle']))):

                    # Filter the ImgFileColl to include only this tilt
                    gra_cl = gr_cl.filter(grangle=gra)

                    # In case more than one order-blocking filter was used (???)
                    for filt in list(set(list(gra_cl.summary['filtrear']))):

                        # Filter the ImgFileColl to include only this filter
                        filt_cl = gra_cl.filter(filtrear=filt)

                        # For engineering, possibly use different lamps for comp
                        if self.multilamp:
                            lamps = list(set(list(filt_cl.summary['comment'])))
                        else:
                            lamps = ['domelamp']
                        if self.debug:
                            print(f'Flat lamps used: {lamps}')

                        for this_lamp in lamps:

                            if self.multilamp:
                                lamp_cl = filt_cl.filter(comment=this_lamp)
                                lname = '_TRING' if this_lamp[0:3] == 'Top' \
                                    else '_FLOOD'
                            else:
                                lamp_cl = filt_cl
                                lname = ''

                            # Actually do the flat combining
                            cflat = ccdp.combine(lamp_cl.files,
                                                 method='median',
                                                 sigma_clip=True,
                                                 sigma_clip_low_thresh=5,
                                                 sigma_clip_high_thresh=5,
                                                 sigma_clip_func=np.ma.median,
                                                 sigma_clip_dev_func=mad_std,
                                                 mem_limit=4e9)

                            # Add FITS keyword FLATCOMB and add HISTORY
                            cflat.header['flatcomb'] = True
                            cflat.header['HISTORY'] = 'Combined flat ' + \
                                                      'created: ' + _savetime()
                            for fn in lamp_cl.files:
                                # Note: These filenames have the path attached,
                                #       via the .filter() method of ImgFileCol.
                                # Include just the pathless filename.
                                cflat.header['HISTORY'] = fn[fn.rfind('/') + 1:]

                            # Build filename, save, remove input files
                            flat_fn = f'flat_{grname}_{gra}_{filt}{lname}.fits'
                            if self.debug:
                                print(f'Saving combined flat as {flat_fn}')
                            cflat.write(f'{self.path}/{flat_fn}',
                                        overwrite=True)
                            for fn in lamp_cl.files:
                                # Note: These filenames have the path already
                                #       attached, via the .filter() method of
                                #       ImgFileCol.
                                os.remove(f'{fn}')
        else:
            print("No flats to be combined.")


# Non-class function definitions
def imcombine(*files, inlist=None, outfn=None, del_input=False, combine=None,
              printstat=True):
    """Combine a collection of images
    This function (crudely) emulates the IRAF imcombine function.  Pass in a
    list of images to be combined, and the result is written to disk with an
    optionally specified output filename.
    :param files: `list`: List of filenames to combine
    :param inlist: `str`: Filename of text file listing images to be combined
    :param outfn: `str`: Filename to write combined image.  Default: append
                         '_comb' to first filename in the input list.
    :param del_input: `bool`: Delete the input files after combination.
                              Default: `false`
    :param combine: `str`: Combine method.  'median' (default), or 'mean'
    :param printstat: `bool`: Print image statistics to screen
    :return: None
    """

    # Check for inputs
    if len(files) > 0 and inlist is not None:
        print("Only one of files or inlist may be specified, not both.")
        raise Exception()

    # Read in the text list inlist, if specified
    if inlist is not None:
        with open(inlist, 'r') as f:
            files = []
            for line in f:
                files.append(line.rstrip())

    # Check for proper file list
    if len(files) < 3:
        print("Combination requires at least three input images.")
        raise Exception()

    # Check that specified input files exist
    for f in files:
        if not os.path.isfile(f):
            print(f"File {f} does not exist.")
            raise Exception()

    # Determine combine method (default = 'median')
    if combine != 'median' or combine != 'mean':
        combine = 'median'

    # Create an ImgFileColl using the input files
    file_cl = ccdp.ImageFileCollection(filenames=files)

    if printstat:
        # Print out the statistics, for clarity
        for img, fn in file_cl.ccds(return_fname=True):
            mini, maxi, mean, stdv = mmms(img)
            print(f'{fn}:: Min: {mini:.2f} Max: {maxi:.2f} ' +
                  f'Mean: {mean:.2f} Stddev: {stdv:.2f}')

    comb_img = ccdp.combine(file_cl.files,
                            method=combine,
                            sigma_clip=True,
                            sigma_clip_low_thresh=5,
                            sigma_clip_high_thresh=5,
                            sigma_clip_func=np.ma.median,
                            sigma_clip_dev_func=mad_std,
                            mem_limit=4e9)

    # Add FITS keyword COMBINED and add HISTORY
    comb_img.header['combined'] = True
    comb_img.header['HISTORY'] = 'Combined image created: ' + _savetime()
    for fn in file_cl.files:
        comb_img.header['HISTORY'] = fn

    # Build filename (if not specified in call), save, remove input files
    if outfn is None:
        outfn = f'{files[0][:-5]}_comb{files[0][-5:]}'
    print(f'Saving combined image as {outfn}')
    comb_img.write(f'{outfn}', overwrite=True)
    if del_input:
        for f in files:
            os.remove(f'{f}')


def _savetime():
    """Shortcut to return the current UT timestamp in a useful form
    :return: `str`: UT timestamp in format %Y-%m-%d %H:%M:%S
    """
    return f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UT'


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


def main():
    """
    This is the main body function.
    """
    pass


if __name__ == "__main__":
    main()
