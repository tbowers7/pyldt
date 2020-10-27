# -*- coding: utf-8 -*-
#
#  This file is part of PyLDT.
#
#  PyLDT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  any later version.
#
#  PyLDT is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with PyLDT.  If not, see <https://www.gnu.org/licenses/>.
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
import glob
import os
import shutil
import warnings
from pathlib import Path

# Numpy
import numpy as np

# Astropy and CCDPROC
from astropy.modeling import models
from astropy.stats import mad_std
from astropy.utils.exceptions import AstropyWarning
import ccdproc as ccdp
from ccdproc.utils.slices import slice_from_string


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
        # Define various named constants
        self.BIN11 = '1 1'
        self.BIN22 = '2 2'
        self.BIN33 = '3 3'
        self.BIN44 = '4 4'

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

    def biascombine(self, binning=None, output="Bias.fits"):
        """Finds and combines bias frames with the indicated binning

        :param binning:
        :param output:
        :return:
        """
        # First, build the file collection
        file_cl = ccdp.ImageFileCollection(self.path,
                                           glob_include=self.prefix + '.*.fits')
        # Loop through files,
        for ccd, file_name in file_cl.ccds(ccdsum=binning, bitpix=16,
                                           imagetyp='bias', return_fname=True):
            # Construct the name of the trimmed file
            trim_name = '{0}t{1}'.format(file_name[:-5], file_name[-5:])
            # Fit the overscan section, subtract it, then trim the image
            ccd = trim_oscan(ccd, self.biassec, self.trimsec)
            # Save the result
            ccd.write(trim_name, overwrite=True)
            # Delete the input file to save space
            os.remove(file_name)

        # Collect the trimmed biases
        t_bias_cl = ccdp.ImageFileCollection(self.path, glob_include='*t.fits')

        # If we have a fresh list of trimmed biases to work with...
        if len(t_bias_cl.files) != 0:

            if self.debug:
                print(f"Combining bias frames with binning {binning}...")
            combined_bias = ccdp.combine(t_bias_cl.files, method='median',
                                         sigma_clip=True,
                                         sigma_clip_low_thresh=5,
                                         sigma_clip_high_thresh=5,
                                         sigma_clip_func=np.ma.median,
                                         sigma_clip_dev_func=mad_std,
                                         mem_limit=4e9)
            # Add FITS keyword BIASCOMB
            combined_bias.meta['biascomb'] = True
            combined_bias.write(output, overwrite=True)
            # Delete input files to save space
            for f in t_bias_cl.files:
                os.remove(f)


class LMI(_Images):
    """Class call for a folder of LMI data to be reduced.

    """

    def __init__(self, path, biassec=None, trimsec=None):
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
        # Set the BIASSEC and TRIMSEC appropriately
        if biassec is None:
            self.biassec = '[3100:3124, 3:3079]'
        else:
            self.biassec = biassec
        if trimsec is None:
            self.trimsec = '[30:3094,   3:3079]'
        else:
            self.trimsec = trimsec
        # File prefix
        self.prefix = 'lmi'
        # Define standard filenames
        self.ZEROFN1 = 'Zero_1x1.fits'
        self.ZEROFN2 = 'Zero_2x2.fits'
        self.ZEROFN3 = 'Zero_3x3.fits'
        self.ZEROFN4 = 'Zero_4x4.fits'


class DeVeny(_Images):
    """Class call for a folder of DeVeny data to be reduced.

    """

    def __init__(self, path, biassec=None, trimsec=None):
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
        # Set the BIASSEC and TRIMSEC appropriately
        if biassec is None:
            self.biassec = '[2101:2144,5:512]'
        else:
            self.biassec = biassec
        if trimsec is None:
            self.trimsec = '[54:  2096,5:512]'
        else:
            self.trimsec = trimsec
        # Define standard filenames
        self.ZEROFN = 'Zero.fits'


def function1(arg1, debug=True):
    """One line description.
    Explanatory paragraph. Link example
    `Paramiko <http://docs.paramiko.org/en/latest/>`_
    Internal reference example
    :func:`dataservants.utils.files.getDirListing`.
    Args:
        arg1 (:TYPE:`internal link or datatype`)
            Description.
        debug (:obj:`bool`)
            Bool to trigger additional debugging outputs. Defaults to False.
    Returns:
        ret1 (:TYPE:`internal link or datatype`)
            Description.
            .. code-block:: python
                stuff = things
    """
    if debug:
        print('I am debugging!')
    return arg1


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
    yb, xb = slice_from_string(biassec, fits_convention=True)
    yt, xt = slice_from_string(trimsec, fits_convention=True)

    # First trim off the top & bottom rows
    ccd = ccdp.trim_image(ccd[yt.start:yt.stop, :])

    # Model & Subtract the overscan
    if model is None:
        model = models.Chebyshev1D(1)  # Chebyshev 1st order function
    else:
        model = models.Chebyshev1D(1)  # Figure out how to incorporate others
    ccd = ccdp.subtract_overscan(ccd, overscan=ccd[:, xb.start:xb.stop],
                                 median=True, model=model)

    # Trim the overscan & return
    return ccdp.trim_image(ccd[:, xt.start:xt.stop])


def main():
    """
    This is the main body function.
    """
    pass


if __name__ == "__main__":
    main()
