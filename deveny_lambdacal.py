import numpy as np
from scipy.signal import find_peaks
from astropy.io import fits

from rascal.calibrator import Calibrator
from rascal.util import load_calibration_lines
import matplotlib.pyplot as plt

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
