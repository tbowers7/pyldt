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
from pathlib import Path

# Astropy and CCDPROC
from astropy.modeling import models
import ccdproc as ccdp
from ccdproc.utils.slices import slice_from_string


class _Images:
    """Internal class, parent of LMI & DeVeny."""

    def __init__(self, path, debug=True):
        self.path = Path(path)
        self.debug = debug
        if self.debug:
            print(self.path)
        # Add attributes that need to be specified for the instrument
        self.biassec = None
        self.trimsec = None

        pass


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
        if biassec is None:
            self.biassec = '[3100:3124, 3:3079]'
        else:
            self.biassec = biassec
        if trimsec is None:
            self.trimsec = '[30:3094,   3:3079]'
        else:
            self.trimsec = trimsec


class DeVeny(_Images):
    """Class call for a folder of DeVeny data to be reduced.

    """

    def __init__(self, path):
        _Images.__init__(self, path)
        pass


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
