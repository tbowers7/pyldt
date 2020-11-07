### Import the necessary packages

## Built-in Libraries
from pathlib import Path
import shutil
import warnings
import os,sys,glob

## Numpy
import numpy as np


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


DEL_INPUT = True

## Suppress the warning:
##    WARNING: FITSFixedWarning: RADECSYS= 'FK5 ' / Astrometric System
##    the RADECSYS keyword is deprecated, use RADESYSa. [astropy.wcs.wcs]
warnings.simplefilter('ignore', AstropyWarning)
warnings.simplefilter('ignore', UserWarning)




# #if __name__ == "__main__":
# print(f"Arguments count: {len(sys.argv)}")
# for i, arg in enumerate(sys.argv):
#     print(f"Argument {i:>6}: {arg}")
    
# Test that arguments 1 & 2 are integers
if len(sys.argv) < 3:
    print("Script requires start and stop image #s to combine.")
    sys.exit()

try:
    img_start = int(sys.argv[1])
    img_end   = int(sys.argv[2])
except:
    print("Start & stop image numbers must be integers.")
    sys.exit()


group = [] 
for fn in range(img_start, img_end+1):
    
    seq = '{:04}'.format(fn)

    group.append(glob.glob(PREFIX+'.'+seq+'*.fits').pop())
    
# print(type(group[0]))

file_cl = ccdp.ImageFileCollection(filenames=group, glob_exclude="*comb.fits")

print(file_cl.summary['objname','grating'])

## Combine these files!

for img,fn in file_cl.ccds(return_fname=True):
    mini,maxi,mean,stdv = mmms(img)
    print(fn+' Min: {:.2f} Max: {:.2f} Mean: {:.2f} Stddev: {:.2f}'.format(
        mini,maxi,mean,stdv))

comb_img = ccdp.combine(file_cl.files, method='median', sigma_clip=True,
                        sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                        sigma_clip_func=np.ma.median,
                        sigma_clip_dev_func=mad_std, mem_limit=4e9)

comb_fn = '{0}_comb{1}'.format(group[0][:-5],group[0][-5:])

comb_img.meta['combined']=True
comb_img.meta['n_comb'] = len(file_cl.files)
for i,fn in enumerate(file_cl.files):
    comb_img.meta['comb'+'{:04}'.format(i+1)] = fn

comb_img.write(comb_fn, overwrite=True)

if DEL_INPUT:
    #Delete input files to save space
    for f in file_cl.files:
        os.remove(f)
        
print("Done combining images.")
