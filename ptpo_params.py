import copy
import pandas as pd
import numpy as np
import dielectric_tensor as dt
import gradient_descent as gd
import os
import python_util

'''
Paramaterization of PTPO, a molecule demonstrating ACD 
Synthesis reported in Zinna, F., Albano, G., Taddeucci, A., Colli, T., Aronica, L. A., Pescitelli, G., & Di Bari, L. (2020). 
Emergent nonreciprocal circularly polarized emission from an organic thin film. Advanced Materials, 32(37), 2002575. 
 https://doi.org/10.1002/adma.202002575
'''

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def init_dp(e_inf = 6.4,vol = 10/1973*(10/1973)*(15/1973),
            gamma= .09,length_nm= 100,gamma_type = "linear",gam_offset= None,vol_offset = None,e_inf_offset= None):
    if (gam_offset is not None):
        gamma = gd.random_offset_to_scalar(gamma,gam_offset)
    if (vol_offset is not None):
        vol = gd.random_offset_to_scalar(vol,vol_offset)
    if (e_inf_offset is not None):
        e_inf = gd.random_offset_to_scalar(e_inf,e_inf_offset)
    return dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const=e_inf,volume_cell=vol,damping_factor=gamma,
                                length = length_nm/197.3,gamma_type=gamma_type)

def get_ptpo_params_offset(quaternion = None,length_nm = 160,save_dipoles = False,theta_offset = None,energy_offset = None,hr_array = None,e_vib= None,
                           mag_debye_offset= None,energy_rebase = None,theta_rebase = None,mag_debye_rebase= None,versionNo = 4,
                           gam_offset = None,e_vib_offset = None,hr_offset= None,vol_offset= None,e_inf_offset = None,e_inf = 6.4,
                           vol =  10/1973*(10/1973)*(15/1973),gamma = 0.09,energy_offset_type = "random",hr_offset_type = "fixed",theta_offset_type = "random",mag_offset_type = "scale",
                           bare_results = None,select_set = None):
    '''Produces PTPO parameters with arbitary rebasing and offsetting of a variety of styles
        As all arguments are keyworded, can notably take a dictionary as input.
        Offset types:
            "random": Each value in an array is offset absolutely by a random value within offset bounds
            "fixed": Each value in an array is offset absolutely by the same random vlaue within offset bounds
            "scale": Same as random but the offsets are now relative (percentage) offsets
    '''
    dp =init_dp(vol=  vol,gamma = gamma,e_inf = e_inf,
                length_nm=length_nm,gam_offset = gam_offset,e_inf_offset= e_inf_offset,vol_offset=vol_offset)
    unit_defs = dt.unit_defs_base #c = \hbar = 1
    if (bare_results is None):
        # internal orientation, vacuum
        print("Using PTPO energies from in vacuo DFT calculations")
        select_set = np.array([6, 13, 24, 26, 27, 36, 38])
        bare_results = python_util.open_object_file(os.sep.join((BASE_DIR, r"DFT\ptpo_tddft_bare_results_internal.pkl")))
    trunc_results = bare_results.truncate_selection(select_set)
    if (energy_rebase is not None):
        print("Rebasing energies to fit values ...")
        trunc_results.energies = energy_rebase
    trunc_results_offset = copy.deepcopy(trunc_results)
    if (energy_offset is not None):
        print("Offsetting energies ...")
        if (energy_offset_type=="fixed"):
            trunc_results_offset.energies = gd.random_offset_to_array(trunc_results.energies,energy_offset)
        elif(energy_offset_type=="random"):
            trunc_results_offset.energies = gd.random_offset_array_to_array(trunc_results.energies,energy_offset)
        else: raise ValueError("Invalid energy_offset_type")
    elec_e_array = trunc_results_offset.energies

    if (hr_array is not None):huang_rhys = hr_array
    else:huang_rhys =  np.ones(np.size(elec_e_array))*.5
    if (hr_offset is not None):
        if (hr_offset_type =="fixed"):
            huang_rhys = gd.random_offset_to_array(huang_rhys,hr_offset)
        elif (hr_offset_type=='random'):
            huang_rhys = gd.random_offset_array_to_array(huang_rhys,hr_offset)
        else: raise ValueError("Invalid hr_offset_type")
    if (e_vib):delta_vib = e_vib
    else: delta_vib = .16
    if (e_vib_offset is not None):
        print("Adding vibronics...")
        delta_vib = gd.random_offset_to_scalar(delta_vib,e_vib_offset)

    vib_modes = np.arange(5)
    dipoles_mu_debye = np.linalg.norm(trunc_results_offset.dip_mat[:,:2]/.393456,axis =1)

    if (mag_debye_rebase is not None):
        print("Rebasing dipole magnitudes...")
        dipoles_mu_debye = mag_debye_rebase
    if (mag_debye_offset is not None):
        print("Offsetting dipole magnitudes...")
        if (mag_offset_type == "fixed"):
            dipoles_mu_debye = gd.random_offset_to_array(dipoles_mu_debye,mag_debye_offset)
        elif (mag_offset_type == 'random'):
            dipoles_mu_debye = gd.random_offset_array_to_array(dipoles_mu_debye,mag_debye_offset)
        elif (mag_offset_type == 'scale'):
            dipoles_mu_debye = gd.random_offset_array_to_array(dipoles_mu_debye,mag_debye_offset,type = "scale")
        else:
            raise ValueError("Invalid hr_offset_type")
    else:
        dipoles_mu_debye = dipoles_mu_debye
    if (save_dipoles):
        np.save("dipoles_ptpo.npy",dipoles_mu_debye)

    dipoles_mu_au = dipoles_mu_debye*.393456
    theta_array_base = np.arctan2(trunc_results_offset.dip_mat[:,1],trunc_results_offset.dip_mat[:,0])

    theta_array_base = (theta_array_base-theta_array_base[0])%np.pi

    if (theta_rebase is not None):
        print("Rebasing angles ...")
        theta_array_base = theta_rebase
    if (theta_offset is not None):
        print("Offsetting angles ...")
        if (theta_offset_type == "fixed"):
            theta_array = gd.random_offset_to_array(theta_array_base,theta_offset)
        elif(theta_offset_type == "random"):
            theta_array = gd.random_offset_array_to_array(theta_array_base,theta_offset)
        else: raise ValueError("Invalid theta_offset_type")
    else:
        theta_array = theta_array_base


    dipole_mat_3d = dt.create_dipole_matrix_polar_3D(dipoles_mu_au,theta_array)

    trunc_results_offset.dip_mat[:,:2] = dipole_mat_3d[:,:2]#overwriting the x and y coordinates
    vib_bare_results = trunc_results_offset.vibronic_dressing(np.arange(np.size(select_set)), delta_vib * np.ones(np.size(select_set)),
                                                       huang_rhys, vib_modes=vib_modes,
                                                      to_zero_unselected=True)
    e_array = vib_bare_results.energies
    vib_dip_vec = vib_bare_results.dip_mat

    if (quaternion is not None):
        rot_matrix = dt.quaternion_rotation_matrix(quaternion=quaternion)
        vib_dip_vec = np.einsum("ij,kj->ki",rot_matrix,vib_dip_vec)

    vib_dip_vec_eV = 0.00026811961479979727 * vib_dip_vec
    spec_its = 401
    spec = dt.nm_to_eV(np.linspace(200, 600, spec_its))

    dipole_mat = vib_dip_vec_eV

    dielectric_tensor = dt.create_dielectric_tensor(dp, vib_dip_vec_eV, vib_bare_results.energies, spec, unit_defs,
                                                 **{"dimension": 3})
    return spec,e_array,dipoles_mu_debye,theta_array,elec_e_array, dp,huang_rhys, dipole_mat,dielectric_tensor


def get_ptpo_params_from_dipole_set(dipole_set,params_dict):
    '''
    Takes dipole set parameters and produces full vibrational dressing parameters
    :param dipole_set: DIPOLE_SET_PARAMS()
    :param params_dict: dict
    :return: spec,e_array,dipoles_mu_debye,theta_array,elec_e_array, dp,huang_rhys, dipole_mat,dielectric_tensor
    '''
    dielectric_params= dipole_set.dielectric_params
    additional_dictionary = {'theta_rebase':dipole_set.angle_array,
                            'energy_rebase':dipole_set.energies,
                             'mag_debye_rebase':dipole_set.dip_mags,
                             'hr_array':dipole_set.huang_rhys,
                             'e_inf':dielectric_params.epsilon_inf,
                             'vol':dielectric_params.v,
                             'gamma':dielectric_params.gamma,
                             'e_vib':dipole_set.e_vib}
    params_dict.update(additional_dictionary)
    return get_ptpo_params_offset(**params_dict)

def load_experimental_data(dataset = "di Bari",sheet = "Annealed film"):
    '''Loads PTPO spectral data from the di Bari group corresponding to https://doi.org/10.1002/adma.202002575'''
    if (dataset == "di Bari"):# https://doi.org/10.1002/adma.202002575
        filepath = os.sep.join((BASE_DIR,"PTPO_DATA\(S)-PTPO data.xlsx"))
        dataframe = pd.read_excel(filepath,sheet_name = sheet)
        array = dataframe.to_numpy()
    elif (dataset == "Goldsmith"): #this paper
        filepath = os.sep.join((BASE_DIR,"Goldsmith_PTPO\A_9_Film\cleaned_spectra.npy"))
        array = np.load(filepath) #see cleaned_spectra_readme.txt for details on this
    return array


'''What follows are parameterizations separated from logic, which is a much better
way of structuring things than was done initially  
'''

def get_ptpo_energies():
    # solution fit
    print("Using PTPO energies from fitting to absorption in solution")
    elec_e_array = np.array([2.95662452, 3.74864818, 4.41768995, 4.57882775, 4.68436758,
       5.47811937, 5.60956698])
    return elec_e_array

