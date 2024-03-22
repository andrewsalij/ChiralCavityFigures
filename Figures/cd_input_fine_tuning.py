import numpy as np
import matplotlib.pyplot as plt
import pandas
from pandas import DataFrame
import python_util
import berreman_mueller as bm
import dielectric_tensor as dt
import dielectric_bridge as db
import ptpo_params
import os
'''
Final tuning of input parameters 
USE THIS FOR FIGURES IN SI 
'''

plt.rcParams.update({'font.size': 7,'axes.unicode_minus':False})

plot_goldsmith = False
create_source_files = True
source_path = os.sep.join((os.getcwd(),"Source_Files"))

input_array = np.load("best_fit_input_params_v6.npy")
input_array = db.order_inputs(input_array,num_dipoles= 7)
init_input_array = np.copy(input_array)
n= 7
energy_array = np.copy(init_input_array[:n])

init_angle_array = np.copy(init_input_array[2*n:3*n])
init_angle_array = init_angle_array
angle_deg_array = init_angle_array*180/np.pi
init_angle_deg_array = np.copy(angle_deg_array)

new_angle_array = -1*(angle_deg_array)*np.pi/180

input_array = db.update_inputs(input_array,energy_array= energy_array,angle_array=new_angle_array)

experimental_data_raw = ptpo_params.load_experimental_data()
experimental_data_raw_goldsmith = ptpo_params.load_experimental_data(dataset= "Goldsmith")

lin_abs_exp_data= experimental_data_raw[:,3]
solution_exp_data = ptpo_params.load_experimental_data(sheet="Solution")

thickness_nm = 300

dipole_set_params = db.inputs_to_dt_params(input_array, num_energies_size=7, length=thickness_nm / 197.3,
                                           input_style="v2")

ptpo_dict = {"length_nm": thickness_nm}

spec, e_array, dipoles_mu_debye_offset, theta_array, elec_e_array, dielectric_params, hr_array, dipole_mat, dielectric_tensor = \
        ptpo_params.get_ptpo_params_from_dipole_set(dipole_set_params, ptpo_dict)

solution_e_array = ptpo_params.get_ptpo_energies() #solution fit
solution_e_array = np.sort(solution_e_array)
solution_dipole_set_params = db.inputs_to_dt_params(db.update_inputs(input_array,energy_array =solution_e_array),num_energies_size=7,length = thickness_nm/197.3,input_style = "v2")
spec_sol, e_array_sol, dipoles_mu_debye_offset_sol, theta_array_sol, elec_e_array_sol, dielectric_params_sol, hr_array_sol, dipole_mat_sol, dielectric_tensor_sol = \
        ptpo_params.get_ptpo_params_from_dipole_set(solution_dipole_set_params, ptpo_dict)

linear_optics_pert =  dt.get_linear_optics_pert(dielectric_params,dipole_mat,e_array,spec)
linear_optics_pert_sol = dt.get_linear_optics_pert(dielectric_params_sol,dipole_mat_sol,e_array_sol,spec_sol)
r, t = bm.characterize_solo_sample_ps(dielectric_tensor, dt.eV_to_nm(spec), spec, thickness_nm, talk=False)
mm, t_set, t_l, t_r,r_l,r_r = bm.mueller_matrix_suite_from_refl_trans_amplitude_matrices_basis_ps(r, t)
absorbance_smm,absorbance_smm_circ,trans_lr_diff_avg = bm.get_absorbance_sets(t_l,t_r,t_set)
normed_mm, log_mm, smm_factored_mm, smm_factored_mm_circ, smm_factored_mm_xy, factored_log_mm = \
            bm.extract_mueller_matrix_factors(mm,t_set,absorbance_smm,absorbance_smm_circ)

cd_mdeg_factor = 32980/np.log(10)*2
exp_cd= experimental_data_raw[:,1]/cd_mdeg_factor

exp_cd_goldsmith = experimental_data_raw_goldsmith[2,:]

#Note: Negatives are to handle sign conventions and to "flip" the film over

fig, ax = plt.subplots(figsize= (3.46,3))
ax.set_ylabel("CD (mdeg)",fontsize = 7 )
ax.set_xlabel("Wavelength (nm) ",fontsize =7)
ax.plot(experimental_data_raw[:,0],exp_cd*cd_mdeg_factor,label = "Exp.",color ="black")

if (plot_goldsmith):
    ax.plot(experimental_data_raw_goldsmith[0,:], -exp_cd_goldsmith, label="Exp. GS", color="red")
y1 = -factored_log_mm[0,3,:]*cd_mdeg_factor
y2 = -np.log10(t_r/t_l)*32980
y3 = -linear_optics_pert.ldlb()*cd_mdeg_factor
x= dt.nm_to_eV(spec)
ax.plot(x,y1 ,label = r"SMM $e^{Az}M_{03}$",color = 'black',linestyle= "dotted")
ax.plot(x,y2, label = "SMM ($\log_{10}(T_R/T_L)$)",color = 'black',linestyle = "-.")
ax.plot(x,y3,label = "LDLB",color  ='black',linestyle = "dashed")
ax.set_xlim(270,520)    
fig.legend(fontsize = 7,loc=  "lower right",bbox_to_anchor= (.95,.15))
plt.tight_layout()
fig.savefig("ptpo_cd_tuning.pdf",dpi = 500)
fig.show()

if (create_source_files):
    source_dir = os.sep.join((source_path,"Fig_S9"))
    os.makedirs(source_dir,exist_ok = True)
    exp_data = np.column_stack((experimental_data_raw[:,0],exp_cd*cd_mdeg_factor))
    exp_df = DataFrame(exp_data,columns = ["Wavelength (nm)","Exp. CD"])
    exp_df.to_csv(os.sep.join((source_dir,"fig_S9_exp_data_x_y")),columns = exp_df.columns,index = False)
    theory_data = np.column_stack((x,y1,y2,y3))
    theory_df = DataFrame(theory_data,columns = ["Wavelength (nm)","SMM $e^{Az}M_{03}$","SMM ($\log_{10}(T_R/T_L)$)","LDLB"])
    theory_df.to_csv(os.sep.join((source_dir,"fig_S9_theory_data_x_y123")),columns = theory_df.columns,index = False)
np.save("new_fit_input_params",input_array)
show_lin_abs=  True
if (show_lin_abs):
    fig, ax = plt.subplots(figsize=  (3.46,3))
    ax.set_ylabel("Absorbance (normed)", fontsize=7)
    ax.set_xlabel("Wavelength (nm)", fontsize=7)
    ax.plot(experimental_data_raw[:,0],python_util.norm_array(experimental_data_raw[:,3]),label = "Exp.",color = 'black')
    ax.plot(dt.nm_to_eV(spec),python_util.norm_array(absorbance_smm_circ),label = "Calc.",color = "black",linestyle = "dotted")
    if (plot_goldsmith):
        ax.plot(experimental_data_raw_goldsmith[0,:],python_util.norm_array(experimental_data_raw_goldsmith[1,:]),label = "Exp. GS",color = 'red')
    ax.plot(dt.nm_to_eV(spec),python_util.norm_array(linear_optics_pert.absorbance),label = "FGR",color= "black",linestyle = "dashed")
    plt.tight_layout()
    ax.set_xlim(270, 520)
    fig.legend(fontsize=7,loc = "upper right",bbox_to_anchor = (.98,.98,0,0))
    fig_title = "final_fit_film_v3_lin_abs"
    fig.savefig(fig_title+".pdf",dpi = 500)
    fig.show()

    if create_source_files:
        source_dir = os.sep.join((source_path, "Fig_S8"))
        os.makedirs(source_dir, exist_ok=True)
        data = np.column_stack((dt.nm_to_eV(spec),python_util.norm_array(absorbance_smm_circ),python_util.norm_array(linear_optics_pert.absorbance)))
        dataframe = pandas.DataFrame(data,columns = ["Wavelength (nm)","SMM","FGR"])
        dataframe.to_csv("fig_S8_data_x_y12.csv",columns = dataframe.columns,index = False)
        exp_data = np.column_stack((experimental_data_raw[:,0],python_util.norm_array(experimental_data_raw[:,3])))
        exp_df = DataFrame(exp_data,columns = ["Wavelength (nm)","Exp."])
        exp_df.to_csv(os.sep.join((source_dir,"fig_S8_exp_data_x_y.csv")),columns = exp_df.columns,index = False)

    fig, ax = plt.subplots(figsize=  (3.46,3))
    ax.set_ylabel("Absorbance (normed)", fontsize=7)
    ax.set_xlabel("Wavelength (nm)", fontsize=7)
    ax.plot(solution_exp_data[:,0],python_util.norm_array(solution_exp_data[:,1]),label = "Exp.",color = "black")
    ax.plot(dt.nm_to_eV(spec),python_util.norm_array(linear_optics_pert_sol.absorbance),label = "Calc.",color= "black",linestyle = "dotted")
    ax.set_xlim(270, 520)
    fig.legend(fontsize=7,loc = "upper right",bbox_to_anchor = (.98,.98,0,0))
    plt.tight_layout()
    fig_title = "final_fit_solution_v2_lin_abs"
    fig.savefig(fig_title+".pdf",dpi = 500)
    fig.show()

    if create_source_files:
        source_dir = os.sep.join((source_path, "Fig_S7"))
        os.makedirs(source_dir, exist_ok=True)
        data = np.column_stack((dt.nm_to_eV(spec),python_util.norm_array(linear_optics_pert_sol.absorbance)))
        dataframe = pandas.DataFrame(data,columns = ["Wavelength (nm)","Calc."])
        dataframe.to_csv("fig_S7_theory_data_x_y.csv",columns = dataframe.columns,index = False)
        exp_df = pandas.DataFrame(solution_exp_data[:,:2],columns = ["Wavelength (nm)","Exp."])
        exp_df.to_csv(os.sep.join((source_dir,"fig_S7_exp_data_x_y.csv")),columns = exp_df.columns,index = False)
make_ldlb_only_figure = False
if make_ldlb_only_figure:
    linewidth = 4
    fig, ax = plt.subplots(figsize=(3.46, 4))
    ax.set_ylabel("CD (mdeg)", fontsize=7)
    ax.set_xlabel("Wavelength (nm) ", fontsize=7)
    ax.plot(experimental_data_raw[:, 0], exp_cd * cd_mdeg_factor, label="Exp.", color="black",linewidth = linewidth)
    ax.plot(dt.nm_to_eV(spec), -linear_optics_pert.ldlb() * cd_mdeg_factor, label="LDLB", color='black',
            linestyle="dotted",linewidth =   linewidth)
    ax.set_xlim(270, 520)
    fig.legend(fontsize=7, loc="lower right", bbox_to_anchor=(.95, .15))
    plt.tight_layout()
    fig.savefig("ptpo_ldlb_only"+".pdf", dpi=500)
    fig.show()







