cleaned_spectra.npy is a numpy array of the A9 Film data after some preprocessing to make the spectra
smoothed at the transmission normalized.

Specifically, both LHP and RHP intensities have been normalized to the greater raw intensity of the two such that
the maximimum intensity is treated as 100% tranmission--from calculations, this is roughly at ~500 nm.

This is, of course, an approximation as there is likely a finite absorbance here. For fitting, this is justifiable.


Without normalization, absorbance gets an arbitrary baseline offset such to around -10, which is clearly 
nonphysical.

If you normalize each separately, you get a spurious CD baseline difference due to the offset.

Absorbances defined as CD = (A_L-A_R)/2 and A = (A_L+A_R)/2 with A_x = -ln(T_x). These are not normalized again-
only the initial raw transmissions.

Transmissions smoothed over a buffer of 16 indices (approx 5 nm), with a padding constant buffer at the ends
of the initial and final data point.

Data arranged in three rows:
Wavelength (nm)
Absorbance (A_L+A_R)/2 (dim)
Circular Dichroism (A_L-A_R)/2 (dim)