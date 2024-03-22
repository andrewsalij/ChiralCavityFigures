import matplotlib.pyplot as plt
import numpy as np
import berreman_mueller as bm
import dielectric_tensor as dt
import ptpo_params
import dbr_params
import dielectric_bridge as db
import pandas as pd
import os
def get_cd_dbr_stack(wl_nm_array,angle_set,quat = None):
    input_array = np.load("best_fit_input_params_v6.npy")
    input_array[(2*7):(3*7)] = -input_array[(2*7):(3*7)]+60*np.pi/180
    thickness_nm = 300
    dipole_set_params = db.inputs_to_dt_params(input_array, num_energies_size=7, length=thickness_nm / 197.3,input_style="v2")
    ptpo_dict = {"length_nm": thickness_nm}
    spec, e_array, dipoles_mu_debye_offset, theta_array, elec_e_array, dielectric_params, hr_array, dipole_mat, dielectric_tensor = \
        ptpo_params.get_ptpo_params_from_dipole_set(dipole_set_params, ptpo_dict)

    if (quat is not None):
        dielectric_tensor = dt.rotate_2D_tensor(quat,dielectric_tensor,rot_type="quat")
    ptpo_eps = dielectric_tensor
    wl_eps_set = dt.eV_to_nm(spec)
    ptpo_thick = thickness_nm

    sample_eps = [ptpo_eps]
    sample_thick = [ptpo_thick]
    wl_nm_list = [wl_eps_set]

    r_matrix_set,t_matrix_set = bm.general_berreman_angle_sweep(sample_eps,sample_thick,wl_nm_list,wl_nm_array,angle_set,circ=True,matrix_type="rl")

    r_l,t_l = bm.reflection_transmission_from_amplitude_matrices(r_matrix_set,t_matrix_set,np.array([1,0]))
    r_r,t_r = bm.reflection_transmission_from_amplitude_matrices(r_matrix_set,t_matrix_set,np.array([0,1]))

    return t_l,t_r,r_r,r_l

wl_nm_array = np.linspace(360,520,51)
angle_set = np.linspace(-40, 40, 11) * np.pi / 180  # in radians
angle_set_degrees = angle_set * 180 / np.pi

axis_angle_array = np.array([1,0,0,1e-5])
quaternion = dt.axis_angle_to_quaternion(axis_angle_array)

t_l,t_r,r_r,r_l  = get_cd_dbr_stack(wl_nm_array,angle_set,quat=quaternion)

wl_to_check = 454
arg_to_check = np.argmin(wl_nm_array-wl_to_check)
avg_r = (r_r+r_l)/2
avg_t = (t_r+t_l)/2
target_arbsorptance = 1-avg_r[arg_to_check,5]-avg_t[arg_to_check,5]

mirror_refl = .854
adj_finesse = dbr_params.get_finesse_from_reflectivities(mirror_refl,mirror_refl*(1-target_arbsorptance))
print("Adjusted Finesse:"+str(adj_finesse))

g = 2*(t_l-t_r)/(t_l+t_r)

cd = 32980*(np.log10(t_r/t_l))

plt.rcParams.update({'font.size': 7,'axes.unicode_minus':False})

filename = "ptpo_cd_300_nm.pdf"
bm.plot_heatmap_spectra(filename=filename,x_array=angle_set_degrees,y_array=wl_nm_array,z_matrix=cd,
                        **{'x_label':"Angle (deg.)",'y_label':"Wavelength (nm)",'cbar_label':r'CD (mdeg)',
                           'cmap':'seismic','title':'',
                           'vmin':-2000,'vmax':2000,'dpi':500})

make_source_files = True
if make_source_files:
    source_path = os.sep.join((os.getcwd(), "Source_Files"))
    source_dir = os.sep.join((source_path, "Fig_S10"))
    os.makedirs(source_dir, exist_ok=True)

    cur_df = pd.DataFrame(cd,columns = angle_set_degrees,index = wl_nm_array)
    cur_df.to_excel(os.sep.join((source_dir,"cd_300_nm.xlsx")),index = True,header = True)

