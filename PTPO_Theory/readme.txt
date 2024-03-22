In this folder is a collection of numpy arrays that can be loaded using numpy.load(filename).

Transmission and reflection data is expressed as 2D np.ndarrays over energy/wavelength and angular dispersion,
with their linear spaces defined in the files "theory_ptpo_back_eV_array.npy" and "theory_ptpo_back_eV_array.npy"

Raw transmissions and refltions follow the naming convention
"theory_ptpo_" : initial identifier
"back_" or "front_" : which orientation of the ptpo
matrix_identifier : "trans_" indicates transmission, "refl_" reflection, lhp/rhp left and right-handed polarization
respectively

For direct comparison to experimental data, use the linear spaces as defined and the matrix "g_abs_matrix,"
where g_abs matrix is defined as 2 (-log(t_l)+log(t_r))/(-log(t_l)-log(t_r)), defined as such to be in most
direct comparison to experiment. Feel free to calculate different forms of the g factor from the raw data, 
though this runs into issues for raw differences of 2*(t_l-t_r)/(t_l+t_r) due to small absolute values  

