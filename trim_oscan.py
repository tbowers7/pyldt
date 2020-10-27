
## Astropy
from astropy.modeling import models

## CCDPROC
import ccdproc as ccdp
from ccdproc.utils.slices import slice_from_string



def trim_oscan(ccd, biassec, trimsec, model=None):
    
    # Convert the FITS bias & trim sectyions into slice classes for use
    yb, xb = slice_from_string(biassec, fits_convention=True)
    yt, xt = slice_from_string(trimsec, fits_convention=True)
    
    ## Because we are trimming off rows at the top & bottom, we must do that
    #  BEFORE modeling & subtracting the overscan, else python pukes because
    #  the trim section is smaller in BOTH axes.
    
    # First trim off the top & bottom rows
    ccd = ccdp.trim_image(ccd[yt.start:yt.stop, :])
    
    # Model & Subtract the overscan
    chebyshev_model = models.Chebyshev1D(1)   # Chebyshev 1st order function
    ccd = ccdp.subtract_overscan(ccd, overscan=ccd[:, xb.start:xb.stop],
                                 median=True, model=chebyshev_model)
    
    # Trim the overscan
    ccd = ccdp.trim_image(ccd[:, xt.start:xt.stop])
    
    return ccd
