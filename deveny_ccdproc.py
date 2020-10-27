### Import the necessary packages

## Built-in Libraries
from pathlib import Path
import shutil
import warnings
import os

## Numpy
import numpy as np

## Astropy
from astropy.nddata import CCDData
from astropy import units as u
from astropy.stats import mad_std
from astropy.utils.exceptions import AstropyWarning

## CCDPROC
import ccdproc as ccdp

## Local libraries
from trim_oscan import *


###========================================================================
### Define named things
BIN11 = '1 1'
ZEROFN = 'Zero.fits'
PREFIX = '20201003'

BIASSEC='[2101:2144,5:512]'
TRIMSEC='[54:  2096,5:512]'

## Suppress the warning:
##    WARNING: FITSFixedWarning: RADECSYS= 'FK5 ' / Astrometric System
##    the RADECSYS keyword is deprecated, use RADESYSa. [astropy.wcs.wcs]
warnings.simplefilter('ignore', AstropyWarning)
warnings.simplefilter('ignore', UserWarning)


###========================================================================
### Begin processessing files in this directory


### Load a list of (raw) images in this directory
files = ccdp.ImageFileCollection('.', glob_include=PREFIX+'.*.fits')

## If a directory containing the raw data is not extant, create it and copy
##   all FITS files there as a backup.
raw_data = Path('.', 'raw')
if not raw_data.exists():
    raw_data.mkdir(exist_ok=True)
    # Copy files to raw_data
    for img in files.files_filtered():
        print(f'Copying {img} to {raw_data}...')
        shutil.copy2(img,raw_data)
else:
    pass



### Before we begin, print a summary table of the files in the directory
print(files.summary['file', 'imagetyp', 'filtrear', 'grating',
                    'exptime', 'objname','naxis1','naxis2'])




### Find the biases, trim them, and combine
print("Collecting bias frames, trim, and combine...")

# Loop through the 'bias' frames to subtract overscan and trim
for ccd, file_name in files.ccds(ccdsum=BIN11, bitpix=16, imagetyp='bias',
                                 return_fname=True):

    trim_name = '{0}t{1}'.format(file_name[:-5],file_name[-5:])
    
    # Fit the overscan section, subrtract it, then trim the image
    ccd = trim_oscan(ccd, BIASSEC, TRIMSEC)
    
    # Save the result
    ccd.write(trim_name, overwrite=True)

    # Delete the input file to save space
    os.remove(file_name)

# Collect the trimmed biases
t_bias_cl = ccdp.ImageFileCollection('.', glob_include='*t.fits')

# If we have a fresh list of trimmed biases to work with...
if len(t_bias_cl.files) != 0:
    trimmed_biases = t_bias_cl.files_filtered(ccdsum=BIN11, imagetyp='bias',
                                              include_path=True)
    
    print("Next, combining 1x1-binned bias frames...")
    combined_bias = ccdp.combine(trimmed_biases, method='median',
                                 sigma_clip=True, sigma_clip_low_thresh=5,
                                 sigma_clip_high_thresh=5,
                                 sigma_clip_func=np.ma.median,
                                 sigma_clip_dev_func=mad_std,
                                 mem_limit=4e9)
    
    combined_bias.meta['combined'] = True
    combined_bias.meta['n_comb'] = len(t_bias_cl.files)

    for i,bias_in in enumerate(trimmed_biases):
        combined_bias.meta['comb'+'{:04}'.format(i+1)] = bias_in



    combined_bias.write(ZEROFN, overwrite=True)
    # Delete input fles to save space
    for f in trimmed_biases:
        os.remove(f)

    print("Done creating master 1x1-binned bias image.")

# But, if we ran this part already, just read in the file from disk
else:
    print("Trimmed biases already combined.")
    combined_bias = CCDData.read(ZEROFN)



### With master bias in hand, overscan correct, trim, and bias-subtract all
###  other framses
print("Trim and bias-subtract all other frames (just 1x1 right now)...")

# Reload the list of raw data frames
files = ccdp.ImageFileCollection('.', glob_include=PREFIX+'.*.fits')
for ccd, file_name in files.ccds(ccdsum=BIN11, bitpix=16,
                                 return_fname=True):

    print(file_name, ccd.header['NAXIS2'],
          ccd.header['NAXIS1'], ccd.header['OBJECT'])
    biased_name = '{0}b{1}'.format(file_name[:-5],file_name[-5:])
    if os.path.isfile(biased_name):
        print("Bias-subtracted image already exists!")
        continue

    # Fit the overscan section, subrtract it, then trim the image
    ccd = trim_oscan(ccd, BIASSEC, TRIMSEC)
    
    ccd = ccdp.subtract_bias(ccd, combined_bias)


    # Edit FILTREAR to possibly make it behave better...
    if ccd.header['filtrear'] == 'Clear (1)':
        ccd.header['filtrear'] = 'Clear'
    if ccd.header['filtrear'] == 'GG420 (2)':
        ccd.header['filtrear'] = 'GG420'
    if ccd.header['filtrear'] == 'GG495 (3)':
        ccd.header['filtrear'] = 'GG495'
    if ccd.header['filtrear'] == 'GG570 (4)':
        ccd.header['filtrear'] = 'GG570'
        
    # Save the result
    ccd.write(biased_name, overwrite=True)

    # Delete the input file to save space
    os.remove(file_name)

print("Done bias-subtracting and trimming images.")



# ### Now to gather the flats and combnine by filter
# print("Okay, let's look at the flats...")

# # Load the list of bias-subtracted data frames
# bsub_cl = ccdp.ImageFileCollection('.', glob_include='lmi.*b.fits')


# print("Normalizing flatfield images...")
# # Flatfield images will be normalized by the mean first, then combined by
# #   filter.

# # Once for the sky flats
# for ccd, file_name in bsub_cl.ccds(ccdsum=BIN11, imagetyp='sky flat',
#                                     return_fname=True):

#     # Divide each flat by its mean
#     ccd = ccd.divide(np.mean(ccd), handle_meta='first_found')
    
#     # Write out the file to a '*n.fits'
#     norm_name = '{0}n{1}'.format(file_name[:-6],file_name[-5:])
#     ccd.write(norm_name, overwrite=True)
#     os.remove(file_name)

# # Once for the dome flats
# for ccd, file_name in bsub_cl.ccds(ccdsum=BIN11, imagetyp='dome flat',
#                                     return_fname=True):

#     # Divide each flat by its mean
#     ccd = ccd.divide(np.mean(ccd), handle_meta='first_found')
    
#     # Write out the file to a '*n.fits'
#     norm_name = '{0}n{1}'.format(file_name[:-6],file_name[-5:])
#     ccd.write(norm_name, overwrite=True)
#     os.remove(file_name)

# print("Flats normalized.")


# print("Combining the flats for each of the filters used...")
# normed_cl = ccdp.ImageFileCollection('.', glob_include='*n.fits')
# if len(normed_cl.files) != 0:
    
#     # List the collection of filters found in this set
#     filters = list(normed_cl.summary['filtcomb'])
        
#     # Determine unique combinations, and produce a list to iterate on
#     unique_filts = list(set(filters))
        
#     # Also make short-list
#     for filt in unique_filts: 
        
#         f1, f2 = tuple(filt.split('-'))
#         flats = normed_cl.files_filtered(filter1=f1, filter2=f2,
#                                          include_path=True)
        
#         print(f"Combining flats for filter combination: {f1}-{f2}...")
#         combined_flat= ccdp.combine(flats, method='median',
#                                     sigma_clip=True, sigma_clip_low_thresh=5,
#                                     sigma_clip_high_thresh=5,
#                                     sigma_clip_func=np.ma.median,
#                                     sigma_clip_dev_func=mad_std,
#                                     mem_limit=4e9)
#         # Make a filename to save
#         if f1 == 'OPEN':
#             f1 = ''
#         if f2 == 'OPEN':
#             f2 = ''
#         flat_fn = 'Flat'+f1+f2+'.fits'
#         print(f'Saving combined flat as {flat_fn}')
#         combined_flat.write(flat_fn, overwrite=True)
#         for fn in flats:
#             os.remove(fn)
            
# else:
#     print("Flats already combined!")





# ### Next, apply the flatfield correction to all images of the same filter type
# print("Dividing science frames by the appropriate master flat...")
# flats_cl = ccdp.ImageFileCollection('.', glob_include='Flat*.fits')
# science_cl = ccdp.ImageFileCollection('.', glob_include='*b.fits')

# for filter in list(flats_cl.summary['filtcomb']):
    
#     print(filter)
#     flat_fn = (flats_cl.files_filtered(ccdsum=BIN11, filtcomb=filter))[0]
#     print(f'Master Flat: {flat_fn}')
#     master_flat = CCDData.read(flat_fn)
    
#     for ccd, file_name in science_cl.ccds(ccdsum=BIN11, filtcomb=filter,
#                                           return_fname=True):
#         ccdp.flat_correct(ccd, master_flat)
#         print(file_name)
        
#         # Write out the file to a '*n.fits'
#         flattened = '{0}f{1}'.format(file_name[:-6],file_name[-5:])
#         ccd.write(flattened, overwrite=True)
#         os.remove(file_name)
        
