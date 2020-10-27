### Import the necessary packages

## Built-in Libraries
from pathlib import Path
import shutil
import warnings
import os,sys,glob

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
from astropy.io import fits


## CCDPROC
import ccdproc as ccdp

## Local libraries
from trim_oscan import *


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


if len(sys.argv) < 2:
    print("Script requires a filename to extract.")
    sys.exit()

file_name = sys.argv[1]
    
if not os.path.isfile(file_name):
    print("File either does not exist or cannot be read.")
    sys.exit()

ccd = CCDData.read(file_name)

spec = ccdp.block_average(ccd[200:350], [150, 1])
print(spec.ndim)


print(spec.shape, ccd.shape)

pixnum = np.asarray(range(ccd.shape[1])).flatten()  # Running pixel number

print(type(spec),type(pixnum))
print(spec.shape, pixnum.shape)


## Figure!
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(pixnum,np.transpose(spec))
#ax.plot(pixnum,flat_fit,'r-')
ax.set_ylim(ymin=0)
plt.title(file_name)
plt.show()


oned_fn = '{0}_1d{1}'.format(file_name[:-5],file_name[-5:])

spec.write(oned_fn, overwrite=True)
os.remove(file_name)
