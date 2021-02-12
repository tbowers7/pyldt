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
from __future__ import division, print_function, absolute_import
from datetime import datetime
import glob
import os
from pathlib import Path
import shutil
import sys
import warnings

# Numpy & Similar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg
from scipy.signal import find_peaks

# Astropy and CCDPROC
from astropy.io import fits
from astropy.modeling import models
from astropy.nddata import CCDData, block_reduce
from astropy.stats import mad_std
from astropy.utils.exceptions import AstropyWarning
import ccdproc as ccdp
from ccdproc.utils.slices import slice_from_string

#from rascal.calibrator import Calibrator
#from rascal.util import load_calibration_lines


# Intrapackage
from .utils import *

# Boilerplate variables
__author__ = 'Timothy P. Ellsworth Bowers'
__copyright__ = 'Copyright 2021'
__credits__ = ['Lowell Observatory']
__license__ = 'MPL-2.0'
__version__ = '0.2.0'
__email__ = 'tbowers@lowell.edu'
__status__ = 'Development Status :: 2 - Pre-Alpha'

## Suppress the warning:
##    WARNING: FITSFixedWarning: RADECSYS= 'FK5 ' / Astrometric System
##    the RADECSYS keyword is deprecated, use RADESYSa. [astropy.wcs.wcs]
warnings.simplefilter('ignore', AstropyWarning)
warnings.simplefilter('ignore', UserWarning)



def twodspec_response(flatfn, function='spline3', order=1):
    """Do something like IRAF's twodspec.longlist.response()

    :param flatfn: Filename of the calibrated flat field image to be reponse'd.
    :param function: Fitting function
    :param order: Order of fitting function [Default: 1]
    :return: 
    """

    # Available fitting functions:
    funcs = ['spline3']

    # Read in the calibrated image from disk
    ccd = CCDData.read(flatfn)
    print(f"Shape of ccd: {ccd.shape}")
    ny, nx = ccd.shape
    print(f"nx = {nx}, ny = {ny}")

    # Collapse the spatial dimension
    spec = block_reduce(ccd, [ny,1], func=np.mean).flatten()
    print(f"Shape of spec: {spec.shape}")

    # Compute the running pixel numbers for the abscissa
    pixnum = np.asarray(range(nx), dtype=int)    
    

    #=========================================================================
    # Create GUI to interactively fit the response

    # Color scheme
    sg.theme('purple')

    # Window layout
    row1 = [sg.Canvas(size=(600,350), key='-PLOT-'), 
            sg.Canvas(size=(600,350), key='-RESID-')]
    row2 = [sg.Text("Fit Function:"), 
            sg.Drop(values=(funcs),auto_size_text=True,
                    default_value=funcs[0],key='function'), 
            sg.Text("      Order of Fit:"), 
            sg.Input(key='-ORDER-',size=(6,1)), 
            sg.Button("Fit"), sg.Button("Apply"), sg.Button("Quit")]
    row3 = [sg.Text(size=(50,1), key='-MESSAGE-', text_color='dark red')]

    # Create the Window
    window = sg.Window(
             "twodspec.response fit",
             [row1, row2, row3],
             location=(0,0),
             finalize=True,
             element_justification="center",
             font="Helvetica 14")

    # Create the basic figues
    ptitle = f"{ccd.header['OBSTYPE']} -- Grating: {ccd.header['GRAT_ID']}" + \
        f"   Grangle: {ccd.header['GRANGLE']}"

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(pixnum,spec, label="Data")
    ax.set_ylim(ymin=0)
    plt.xlabel('Colunmn #')
    plt.ylabel('Mean Value (DN)')
    plt.legend(loc='upper left')
    plt.title(ptitle)
 
    # Add the plot to the window
    fig_agg = draw_figure(window["-PLOT-"].TKCanvas, fig)
    fig_aggr = None
    window['-MESSAGE-'].update("Enter order of fitting function to proceed")

    flat_fit = None

    # Run the window
    while True:

        # On event, read it in
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Quit':
            break

        elif event == 'Fit':
            function = values["function"]
            try:
                order = int(values["-ORDER-"])
            except ValueError:
                window['-MESSAGE-'].update(
                    "Please enter an integer for the order.")
                continue
            
            # Parse out fitting function
            if function == 'spline3':
                print("Fitting cubic spline to the data")

                # Fit the spline3 with specified order
                flat_fit, knots, errmsg = spline3(pixnum, spec, order)

                # Print error message to window if not blank
                if errmsg != '':
                    window['-MESSAGE-'].update(
                        errmsg)
                    continue

            else:
                print("Only 'spline3' currently implemented.  Try again.")

            # Redraw the figures
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(pixnum,spec, label="Data")
            ax.plot(pixnum,flat_fit, 'r-', label="Fit")
            ax.set_ylim(ymin=0)
            plt.vlines(knots, 0, np.max(spec), 'lightgray', linestyles='dashed')
            plt.xlabel('Colunmn #')
            plt.ylabel('Mean Value (DN)')
            plt.legend(loc='upper left')
            plt.title(ptitle)

            if fig_agg is not None:
                delete_fig_agg(fig_agg)
            fig_agg = draw_figure(window["-PLOT-"].TKCanvas, fig)

            figr, axr = plt.subplots(figsize=(6,4))
            axr.plot(pixnum,spec/flat_fit, 'r-', label="Ratio")
            axr.plot(pixnum,spec/spec, 'b--')
            plt.xlabel('Colunmn #')
            plt.ylabel('Ratio')
            plt.legend(loc='upper left')
            lower, upper = plt.ylim()
            print(f"Y limits: {lower} {upper}")
            print(f"Proposed upper Y limit: {np.maximum(0.2, lower)}")
            print(f"Proposed lower Y limit: {np.minimum(5, upper)}")
            axr.set_ylim(ymax=np.minimum(2, upper), 
                            ymin=np.maximum(0.5, lower))
            lower, upper = plt.ylim()
            plt.vlines(knots, lower, upper, 'lightgray', linestyles='dashed')
            plt.title(f"Residual plot for {function}, order: {order}")

            if fig_aggr is not None:
                delete_fig_agg(fig_aggr)
            fig_aggr = draw_figure(window["-RESID-"].TKCanvas, figr)

            window['-MESSAGE-'].update(
                "Refit if necessary, or click 'Apply' to divide the flat.")

        elif event == 'Apply':

            # Do the division of ccd and save to file
            if flat_fit is None:
                window['-MESSAGE-'].update(
                    "You must attempt a fit before applying.")
            else:
                print("Doing the division now... quitting.")
                break




    window.close()


def twodspec_apextract(file_name, stype='model', write_1d=True,
                       del_input=False):
    """Aperture extraction.  Do something like IRAF's twodspec.axpextract()

    Routine only accepts single-order spectra at the moment... could be
    expanded in the future, as desired.  Follow the example in PyDeVeny.

    :param file_name: Name of file to be apextract'd
    :param stype: Type of spectrum (like DeVeny's swext in dextract)
    :param write_1d: Write out the 1D Spectrum to disk? [Default: True]
    :param del_input: Also delete input file? [Default: False]
    :return: spectrum, pixnum (Spectrum, and Running Pixel Number)
    """

    # Check for existance of file        
    if not os.path.isfile(file_name):
        print("File either does not exist or cannot be read.")
        return

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
        oned_fn = '{0}_1d{1}'.format(file_name[:-5],file_name[-5:])
        spectrum.write(oned_fn, overwrite=True)
        if del_input:
            os.remove(file_name)

    pixnum = np.asarray(range(nx)).flatten()  # Running pixel number
    return spectrum, pixnum


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


def twodspec_identify():
    # Open the example file
    spectrum2D = fits.open("20201003.0038b_comb_1d.fits")[0].data
    npix = spectrum2D.shape[1]

    # Get the median along the spectral direction
    spectrum = np.median(spectrum2D, axis=0)

    print(spectrum.shape)

    # Load the Lines from library
    atlas = load_calibration_lines(elements=["Ne","Ar","Hg","Cd"])

    #print(atlas)

    # Get the spectral lines
    peaks, _ = find_peaks(spectrum)

    print(peaks)
    print(peaks[1:] - peaks[:-1])

    print(type(peaks),type(atlas))


    pixnum = np.arange(npix)  # Running pixel number
    ## Figure!
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(pixnum,np.transpose(spectrum))
    #ax.plot(pixnum,flat_fit,'r-')
    ax.set_ylim(ymin=0)
    #plt.title(file_name)
    plt.show()






    # Set up the Calibrator object
    c = Calibrator(peaks, npix)

    # Solve for the wavelength calibration
    best_p = c.fit()

    # Produce the diagnostic plot
    c.plot_fit(spectrum, best_p)




def twodspec_reidentify():
    pass




#=============================================================================
# Graphics helper functions

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def delete_fig_agg(fig_agg):
    fig_agg.get_tk_widget().forget()
    plt.close('all')
