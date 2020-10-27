### Import the necessary packages

## Built-in Libraries
from pathlib import Path
import shutil
import warnings
import os

## Numpy
import numpy as np

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from patsy import dmatrix
import statsmodels.api as sm

## Astropy
from astropy.nddata import CCDData
from astropy import units as u
from astropy.modeling import models
from astropy.stats import mad_std
from astropy.utils.exceptions import AstropyWarning


## CCDPROC
import ccdproc as ccdp

## Local libraries



###========================================================================
### Define named things
BIN11 = '1 1'
ZEROFN = 'Zero.fits'
PREFIX = '20201003'
DV1 = "150/5000"
DV2 = "300/4000"
DV3 = "300/6750"
DV4 = "400/8500"
DV5 = "500/5500"
DV6 = "600/4900"
DV7 = "600/6750"
DV8 = "831/8000"
DV9 = "1200/5000"




BIASSEC='[2101:2144,5:512]'
TRIMSEC='[54:  2096,5:512]'

## Suppress the warning:
##    WARNING: FITSFixedWarning: RADECSYS= 'FK5 ' / Astrometric System
##    the RADECSYS keyword is deprecated, use RADESYSa. [astropy.wcs.wcs]
warnings.simplefilter('ignore', AstropyWarning)
warnings.simplefilter('ignore', UserWarning)


### Now to gather the flats and combnine by filter
print("Okay, let's look at the flats...")

# Load the list of bias-subtracted data frames
bsub_cl = ccdp.ImageFileCollection('.', glob_include=PREFIX+'.*b.fits')


## If we haven't done this yet...
if len(bsub_cl.files_filtered(imagetyp="dome flat")) != 0:
    
    flats_cl = bsub_cl.filter(imagetyp="dome flat")
    
    gratings = list(set(list(flats_cl.summary['grating'])))
    
    for g in gratings:
        
        if g == DV1:
            grname = 'DV1'
        elif g == DV2:
            grname = 'DV2'
        elif g == DV3:
            grname = 'DV3'
        elif g == DV4:
            grname = 'DV4'
        elif g == DV5:
            grname = 'DV5'
        elif g == DV6:
            grname = 'DV6'
        elif g == DV7:
            grname = 'DV7'
        elif g == DV8:
            grname = 'DV8'
        elif g == DV9:
            grname = 'DV9'
        
        gr_cl = flats_cl.filter(grating=g)
        grangles = list(set(list(gr_cl.summary['grangle'])))
        
        for gra in grangles:
            
            gra_cl = gr_cl.filter(grangle=gra)
            filtrears = list(set(list(gra_cl.summary['filtrear'])))
        
            for filt in filtrears:
            
                filt_cl = gra_cl.filter(filtrear=filt)
                lamps = list(set(list(filt_cl.summary['comment'])))

                print(lamps)
                
                for lamp in lamps:

                    lamp_cl = filt_cl.filter(comment=lamp)

                    print(lamp[0:3])
                    
                    if lamp[0:3] == 'Top':
                        lname = 'TRING'
                    else:
                        lname = 'FLOOD'
                    print(lname)
                        
                    flatname = "{}_{}_{}_{}".format(grname, gra,
                                                    filt, lname)
                    print(flatname)
                    
                    combined_flat = ccdp.combine(lamp_cl.files,
                                                 method='median',
                                                 sigma_clip=True,
                                                 sigma_clip_low_thresh=5,
                                                 sigma_clip_high_thresh=5,
                                                 sigma_clip_func=np.ma.median,
                                                 sigma_clip_dev_func=mad_std,
                                                 mem_limit=4e9)
                    flat_fn = 'Flat'+flatname+'.fits'
                    print(f'Saving combined flat as {flat_fn}')
                    combined_flat.write(flat_fn, overwrite=True)
                    for fn in lamp_cl.files:
                        os.remove(fn)

print("All flats combined.")

### After all that jazz...
flat_cl = ccdp.ImageFileCollection('.', glob_include='Flat*.fits')

for ccd, file_name in flat_cl.ccds(return_fname=True):
    
    
    ## We're going to collapse the flatfield image and fit a cubic spline to it
    nrows = ccd.shape[0]
    spec = np.sum(ccd, axis=0) / nrows
    
    pixnum = range(ccd.shape[1])  # Running pixel number
    order = 4
    knots = (range(order)/order)[1:]*ccd.shape[0]
    print(ccd.shape[0])
    print(knots)
    
    # Fit a natural spline with knots at ages 30, 50 and 70
    x_natural = dmatrix('cr(x, knots=(30, 50, 70))', {'x': pixnum})
    fit_natural = sm.GLM(spec, x_natural).fit()
    
    # Create spline lines for 50 evenly spaced values of age
    flat_fit = fit_natural.predict(dmatrix('cr(pixnum, knots=(30, 50, 70))', {'pixnum': pixnum}))

    
    
    
    ## Figure!
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(pixnum,spec)
    #ax.plot(pixnum,flat_fit,'r-')
    ax.set_ylim(ymin=0)
    plt.title(file_name)
    plt.show()
    
