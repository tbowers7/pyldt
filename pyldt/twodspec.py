# -*- coding: utf-8 -*-
#
#  This file is part of PyLDT.
#
#   This Source Code Form is subject to the terms of the Mozilla Public
#   License, v. 2.0. If a copy of the MPL was not distributed with this
#   file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 26-Jan-2021
#
#  @author: tbowers

"""PyLDT contains image calibration routines for LDT facility instruments

Lowell Discovery Telescope (Lowell Observatory: Flagstaff, AZ)
http://www.lowell.edu

This file contains routines for extracting spectra from 2D spectrographic
images.  They are intended to operate in a manner similar to their IRAF
namesakes.
"""

# Built-In Libraries
import os
import warnings

# 3rd Party Libraries
from astropy.nddata import CCDData
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
import ccdproc as ccdp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

# Internal Imports

# Suppress the warnings:
#    WARNING: FITSFixedWarning: RADECSYS= 'FK5 ' / Astrometric System
#    the RADECSYS keyword is deprecated, use RADESYSa. [astropy.wcs.wcs]
# Silence the AstropyDeprecationWarning:
#    block_replicate was moved to the astropy.nddata.blocks module.
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', AstropyWarning)
warnings.simplefilter('ignore', AstropyDeprecationWarning)


def twodspec_apextract(file_name, stype='model', write_1d=True,
                       del_input=False, return_hdr=False):
    """Aperture extraction.  Do something like IRAF's twodspec.axpextract()

    Routine only accepts single-order spectra at the moment... could be
    expanded in the future, as desired.  Follow the example in PyDeVeny.

    :param file_name: Name of file to be apextract'd
    :param stype: Type of spectrum (like DeVeny's swext in dextract)
    :param write_1d: Write out the 1D Spectrum to disk? [Default: True]
    :param del_input: Also delete input file? [Default: False]
    :param return_hdr: Return the FITS header? [Default: False]
    :return: spectrum, pixnum [,header] (Spectrum, Pixel Number [,FITS header])
    """

    # Check for existance of file
    if not os.path.isfile(file_name):
        print("File either does not exist or cannot be read.")
        return None

    # Read in said file
    ccd = CCDData.read(file_name)
    print(f"Shape of ccd: {ccd.shape}")
    ny, nx = ccd.shape
    print(f"nx = {nx}, ny = {ny}")

    # Define the aperture to extract
    trace, width = twodspec_apdefine(ccd, stype=stype)

    # Case out stype:
    if stype == 'star':
        pass
    elif stype == 'model':
        # Model Spectra (flats, model arcs, etc.)
        spectrum = ccdp.block_average(ccd, [ny,1])  # Squash, keeping header
        halfwin = int(np.floor(width/2))

        # Because of python indexing, we need to "+1" the upper limit in order
        #   to get the full wsize elements for the average
        for i in range(nx):
            spectrum.data[0,i] = np.average(
                ccd.data[int(trace[i]) - halfwin :
                         int(trace[i]) + halfwin + 1, i])

    else:
        pass

    if write_1d:
        # Write one-dimensional spectrum to disk
        oned_fn = f"{file_name[:-5]}_1d{file_name[-5:]}"
        spectrum.write(oned_fn, overwrite=True)
        if del_input:
            os.remove(file_name)

    # Create running pixel number
    pixnum = np.asarray(range(nx)).flatten()

    # Return tuple based on input parameter
    if return_hdr:
        return np.asarray(spectrum).flatten(), pixnum, ccd.header
    return np.asarray(spectrum).flatten(), pixnum


def twodspec_apdefine(ccd, stype='star'):
    """Apdefine -- define an aperture

    :param ccd: Input CCDData object
    :param stype: Type of spectrum for which to define an aperture
    :return:
    """

    # Get image dimensions
    ny, nx = ccd.shape

    # Case out spectrum types
    if stype == 'star':
        pass

    elif stype == 'model':
        # Model Spectra (flats, model arcs, etc.)
        # Trace down the middle, with a window pix wide
        trace = np.full(nx, ny/2, dtype=float)
        width = 201

    else:
        pass

    return trace, width


#=============================================================================
# Graphics helper functions

def draw_figure(canvas, figure):
    """draw_figure [summary]

    [extended_summary]

    Parameters
    ----------
    canvas : [type]
        [description]
    figure : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def delete_fig_agg(fig_agg):
    """delete_fig_agg [summary]

    [extended_summary]

    Parameters
    ----------
    fig_agg : [type]
        [description]
    """
    fig_agg.get_tk_widget().forget()
    plt.close('all')
