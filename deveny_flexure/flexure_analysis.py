# -*- coding: utf-8 -*-
#
#  This file is part of ______.
#
#   This Source Code Form is subject to the terms of the Mozilla Public
#   License, v. 2.0. If a copy of the MPL was not distributed with this
#   file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 08-Jul-2021
#
#  @author: tbowers, bshafransky

"""Analysis of the flexure seen in the DeVeny Spectrograph (LDT)

This file contains the main driver for the analysis.
Should be run in an environment containing:
    * AstroPy
    * CCDPROC
    * NumPy
    * Matplotlib
    * SciPy

Run from the command line:
% python flexure_analysis.py DATA_DIR [rescan]

"""

import os
import ccdproc as ccdp
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import warnings

# Get local routines
from from_dfocus import *


def flexure_analysis(data_dir, rescan=False):
    """flexure_analysis Driving routine for the analysis

    [extended_summary]

    Parameters
    ----------
    data_dir : `str`
        Directory where the data live
    rescan : `bool`
        Forcably rescan and refit the files
    """    

    for grating in ['DV1','DV2','DV5']:

        save_fn = f"flex_data_{grating}.fits"

        # Check to see if we saved the AstroPy table to FITS...
        if not rescan and os.path.isfile(save_fn):
            table = Table.read(save_fn)

        else:
            # Create an ImageFileCollection with files matching this grating;
            #  if empty, move along
            gcl = load_images(data_dir, grating)
            if len(gcl.files) == 0:
                continue

            # AstroPy Table of line positions for each image in the IFC
            table = get_line_positions(gcl)

            # Go through the identified lines and produce a set found in all images
            table = validate_lines(table)

            # Add table columns for Delta away from 0º, and Delta away from mean
            table = compute_line_deltas(table)

            # Write the validated table to disk for future use (faster analysis)
            table.write(save_fn, overwrite=True)

        # Print out information on the table to the screen
        print(table.info)

        # Analyze!!!
        make_plots(table, grating)

    return


def load_images(data_dir, grating):
    """load_images Load in the images associated with DATA_DIR and grating
    
    [extended_summary]

    Parameters
    ----------
    data_dir : `str`
        The directory containing the data to analyze
    grating : `str`
        The grating ID to use

    Returns
    -------
    `ccdproc.image_collection.ImageFileCollection`
        IFC of the files meeting the input criteria
    """    
    # Dictionary
    gratid = {'DV1':'150/5000', 'DV2':'300/4000', 'DV5':'500/5500'}

    # Load the images of interest in to an ImageFileCollection()
    icl = ccdp.ImageFileCollection(data_dir)

    # Clean up the telescope altitude
    for ccd, fn in icl.ccds(return_fname=True):
        ccd.header['telalt'] = np.round(ccd.header['telalt'])
        ccd.write(os.path.join(data_dir,fn), overwrite=True)

    # Return an ImageFileCollection filtered by the grating desired
    return icl.filter(grating=gratid[grating])


def get_line_positions(icl, win=11, thresh=5000.):
    """get_line_positions Compute the line positions for the images in the icl

    [extended_summary]

    Parameters
    ----------
    icl : `ccdproc.image_collection.ImageFileCollection`
        ImageFileCollection of images to work with
    win : `int`, optional
        Window (in pixels) across which to extract the spectrum, [Default: 11]
    thresh : `float`, optional
        Line intensity (ADU) threshold for detection, [Default: 5000.]

    Returns
    -------
    `astropy.table.table.Table`
        Table of line positions with associated metadata
    """    
    # Put everything into a list of dicionaties
    flex_line_positions = []

    # This will only give the x values of the fits file.
    # For each of the images,
    for ccd, fname in icl.ccds(return_fname=True):
        # Need a lower threshold for DV5 than for DV1
        if ccd.header['grating'] == '500/5500':
            thresh = 1000.
        # Check for bias frames
        if ccd.header['exptime'] == 0:
            continue
        print("")
        #====================
        # Code cut-and-paste from dfocus() -- Get line centers above `thresh`
        # Parameters for DeVeny (2015 Deep-Depletion Device):
        n_spec_pix, prepix = (2048, 50)
        # Trim the image (remove top and bottom rows, pre- and post-scan pixels)
        spec2d = ccd.data[12:512,prepix:prepix+n_spec_pix]
        ny, nx = spec2d.shape
        trace = np.full(nx, ny/2, dtype=float).reshape((1,nx)) # Right down the middle
        spec1d = extract_spectrum(spec2d, trace, win)
        # Find the lines:
        centers, _ = find_lines(spec1d, thresh=thresh, minsep=17)
        nc = len(centers)
        cen_list = [f'{cent}' for cent in centers]
        print(f"Found {nc} Line Centers: {cen_list}")
        #====================

        # For ease
        h = ccd.header
        # For saving the table to disk
        cen_str = ','.join(cen_list)

        flex_line_positions.append({'filename':fname,
                                    'obserno': h['obserno'],
                                    'telalt':h['telalt'],
                                    'telaz':h['telaz'],
                                    'rotangle':h['rotangle'],
                                    'utcstart':h['utcstart'],
                                    'lampcal':h['lampcal'],
                                    'grating':h['grating'],
                                    'grangle':h['grangle'],
                                    'slitasec':h['slitasec'],
                                    'nlines':nc,
                                    'xpos':cen_str})

    t = Table(flex_line_positions)
    return t


def validate_lines(t):
    """validate_lines Validate the found lines to produce a uniform set

    The number of lines identified will vary form image to image.  This
    function validates the lines to return the set of lines found in ALL
    images for this grating.

    Parameters
    ----------
    t : `astropy.table.table.Table`
        AstroPy Table as produced by get_line_positions()

    Returns
    -------
    `astropy.table.table.Table`
        AstroPy Table identical to input except the lines are validated
    """    
    print("Yay, Validation!!!!")
    nl = t['nlines']
    # print(f"Mean # of lines found: {np.mean(nl)}  Min: {np.min(nl)}  Max: {np.max(nl)}")

    # Create a variable to hold the FINAL LINES for this table
    final_lines = None
    for row in t:
        # Line centers found for this image
        cens = np.asarray([float(c) for c in row['xpos'].split(',')])

        # If this is the first one, easy...
        if final_lines is None:
            final_lines = cens
        else:
            # Remove any canonical lines not in every image
            for line in final_lines:
                # If nothing is in the same ballpark (say, 12 pixels), 
                #   toss this canonical line
                if np.min(np.absolute(cens - line)) > 12.: 
                    final_lines = final_lines[final_lines != line]

    n_final = len(final_lines)
    print(f"Validated {n_final} lines.")
    # Go back through, and replace the `xpos` value in each row with those
    #  lines corresponding to the good final lines
    xpos = []
    for i, row in enumerate(t):
        # Line centers found for this image
        cens = np.asarray([float(c) for c in row['xpos'].split(',')])

        # Keep just the lines that match the canonical lines
        keep_lines = []
        for line in final_lines:
            min_diff = np.min(np.absolute(diffs := cens - line))
            idx = np.where(np.absolute(diffs) == min_diff)
            # This is the line for this image that matches this canonical line
            keep_lines.append(cens[idx])

        # Put the array into a list to wholesale replace `xpos`
        xpos.append(np.asarray(keep_lines).flatten())

    t['nlines'] = [n_final] * len(t)
    t['xpos'] = xpos
    return t


def compute_line_deltas(t):
    """compute_line_deltas Compute line shifts and add to Table

    [extended_summary]

    Parameters
    ----------
    t : `astropy.table.table.Table`
        AstroPy Table as produced by validate_lines()
        Note: Must be VALIDATED, so `xpos` are arrays, not strings

    Returns
    -------
    `astropy.table.table.Table`
        AstroPy Table identical to validated table, with extra columns
    """

    # Things for relating shifts w.r.t. ROTANGLE = 0
    fiducial = t['xpos'][0]
    delta_to_zero = []
    for row in t:
        delta_to_zero.append(row['xpos'] - fiducial)

    # Things for relating shifts w.r.t. MEAN
    xpos = t['xpos']
    del_mean = np.copy(xpos)
    _, nl = xpos.shape
    for line in range(nl):
        del_mean[:,line] = xpos[:,line] - np.mean(xpos[:,line])

    t['del_zero'] = delta_to_zero
    t['del_mean'] = del_mean

    return t


def make_plots(t, grating):
    """make_plots Make plots of the data... with subcalls to fitting functions

    [extended_summary]

    Parameters
    ----------
    t : `astropy.table.table.Table`
        AstroPy Table for this grating, containing all the data!
    grating : `str`
        Grating name, for labeling plots and creating filenames
    """
    # Silence OptimizeWarning
    warnings.simplefilter('ignore', optimize.OptimizeWarning)

    # Set up the plotting environment
    _, ax = plt.subplots()
    tsz = 8

    x,y,k = construct_plotting_pairs(t, 'rotangle', 'del_zero', 'telalt')
    for i in range(len(x)):
        xp, yp = (x[i], y[i])
        ax.plot(xp,yp,f"C{i if i < 10 else i-10}.")
        xp = np.swapaxes(np.tile(xp,[len(yp[0]),1]),0,1)
        # print(f"Shapes: {xp.shape} {yp.shape}")

        par, _ = optimize.curve_fit(sinusoid, xp.flatten(), yp.flatten(), p0=[1, 1, 0, 0])
        xpl = np.arange(101) * (np.max(xp) - np.min(xp)) /100. + np.min(xp)
        ypl = sinusoid(xpl, par[0], par[1], par[2], par[3])
        label = f"El = {k[i]:.0f}"+r'$^\circ$'#+f", A={par[0]:.1f} B={par[1]:.2f} C={par[2]:.1f} D={par[3]:.1f}"
        ax.plot(xpl, ypl, f"C{i if i < 10 else i-10}-", label=label)

    ax.set_xlabel('Cassegrain Rotator Angle [deg]', fontsize=tsz)
    ax.set_ylabel(r'Line Center Deviation from CASS=$0^\circ$ Position [pixels]', fontsize=tsz)

    # Final adjustments and save figure
    ax.legend(loc='upper left', fontsize=tsz)
    ax.tick_params('both', labelsize=tsz, direction='in', top=True, right=True)
    plt.tight_layout()
    plt.savefig(f"flexure_analysis_{grating}.eps")
    plt.savefig(f"flexure_analysis_{grating}.png")

    return


def sinusoid(x, a, b, c, d):
    """sinusoid Sine Function

    [extended_summary]

    Parameters
    ----------
    x : `float`
        Abscissa (degrees)
    a : `float`
        Amplitude
    b : `float`
        Angular Frequency
    c : `float`
        Phase Offset (degrees)
    d : `float`
        Vertical offset

    Returns
    -------
    `float``
        Ordinate
    """
    # Fix angular frequency at 1
    b = 1.0
    return a * np.sin(b * x*np.pi/180. + c*np.pi/180.) + d


def construct_plotting_pairs(t, abs, ord, sort):
    """construct_plotting_pairs Construct plotting pairs from table

    Extract the requested things from the table to construct plotting pairs
    that can be directly input into ax.plot(), sorted by some other key

    Parameters
    ----------
    t : `astropy.table.table.Table`
        The table thingie we've been passing around
    abs : `str`
        FITS keyword for the data to appear on the abscissa
    ord : `str`
        FITS keyword for the data to appear on the ordinate
    sort : `str`
        FITS keyword for the data to sort by
    """
    # Clean the input keywords --> lower()
    abs = abs.lower()
    ord = ord.lower()
    sort = sort.lower()

    # Set blank lists to be filled
    x = []
    y = []

    # Index the table based on the `sort` parameter
    t_by_sort = t.group_by(sort)
    for key in t_by_sort.groups.keys[sort]:
        mask = (t_by_sort.groups.keys[sort] == key)
        sub_t = t_by_sort.groups[mask]
        print(f"Key: {key}")
        sub_t.pprint()
        x.append(np.asarray(sub_t[abs]))
        y.append(np.asarray(sub_t[ord]))

    return x, y, t_by_sort.groups.keys[sort]




#==============================================================================
def main(args):
    """Main driving routine

    Call should be of form:
    % python flexure_analysis.py DATA_DIR [rescan]
    """
    from os import path

    # Exit if command-line arguments aren't valid
    if len(args) == 1:
        print(f"ERROR: scrpit {args[0]} requires the DATA_DIR to analyze.")
        return
    if not path.isdir(args[1]):
        print(f"ERROR: DATA_DIR must be a directory containing the data to analyze.")
        return

    # Check for RESCAN argument
    if len(args) > 2 and args[2] == 'rescan':
        rescan = True
    else:
        rescan = False

    # Announce that we're putting our fingers in our ears and saying "LALALALA"
    if len(args) > 2 and not rescan:
        print(f"WARNING: I'm ignoring the following arguments: {args[2:]}")
    elif len(args) > 3 and rescan:
        print(f"WARNING: I'm ignoring the following arguments: {args[3:]}")

    # Run the analysis
    flexure_analysis(args[1], rescan=rescan)

    return


if __name__ == '__main__':
    import sys
    main(sys.argv)
