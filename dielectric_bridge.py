import numpy as np
import dielectric_tensor as dt
'''
Bridging functions between arrays containing input parameters for dipoles and functions
that directly are used modelling the dielectric function 

Many of these functions are not designed for general modelling but rather as rough data transport
operations
'''
def update_inputs(input_array,num_dipoles = 7,energy_array = None,mu_debye_array = None,angle_array = None, vol = None,
                  gamma= None, eps_inf=  None, e_vib = None, huang_rhys = None):
    '''
    Updates
    Only supported for input_style v2 (which is the full inputs)
    :return: np.ndarray
    '''
    n = num_dipoles
    return_array = np.copy(input_array)
    if (energy_array is not None):return_array[:n] = energy_array
    if (mu_debye_array is not None):return_array[n:(2*n)] = mu_debye_array
    if (angle_array is not None):return_array[(2*n):(3*n)] =angle_array
    if (vol is not None): return_array[3*n] = vol
    if (gamma is not None):return_array[3*n+1] = gamma
    if (eps_inf is not None):return_array[3*n+2] = eps_inf
    if (e_vib is not None):return_array[3*n+3] = e_vib
    if (huang_rhys is not None):return_array[3*n+4] = huang_rhys
    return return_array

def order_inputs(input_array,num_dipoles = 7):
    n = num_dipoles
    sorted_idx = np.argsort(input_array[0:n])
    return update_inputs(input_array,num_dipoles=num_dipoles,energy_array = input_array[0:n][sorted_idx],
                         mu_debye_array= input_array[n:2*n][sorted_idx],angle_array=input_array[2*n:3*n][sorted_idx])
def inputs_to_dt_params(input_array, num_energies_size,length, input_style="v1"):
    '''Converts array of dipole parameters inputs to DIPOLE_SET_PARAMS() for dielectric calcs'''
    n = num_energies_size
    energy_array, mu_array, angle_array, vol, gamma, eps_inf, e_vib, huang_rhys = extract_inputs(input_array,n,input_style=input_style)
    angle_array = angle_array % np.pi
    hr_array = np.ones(n)*huang_rhys
    dielectric_params = dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const=eps_inf,volume_cell=vol,damping_factor=gamma,length =length )
    dipole_set_params = dt.DIPOLE_SET_PARAMS(dielectric_params,energy_array,mu_array,angle_array,hr_array,e_vib)
    return dipole_set_params

def extract_inputs(input_array,n,input_style):
    '''Parses dipole parameter input arrays according to a variety of styles'''
    if (input_style == "v1"):
        #form E , theta ,  [vol,gamma,eps_inf,e_vib,huang_rhys]
        energy_array = input_array[:n]
        angle_array = input_array[n:(2 * n)]
        vol, gamma, eps_inf, e_vib, huang_rhys = tuple(input_array[(2 * n):])
        mu_array = None
    elif (input_style =="v2"):
            #form E ,mu , theta , vol,gamma,eps_inf,e_vib,huang_rhys
            energy_array = input_array[:n]
            mu_array = input_array[n:(2 * n)]
            angle_array = input_array[(2 * n):(3 * n)]
            vol, gamma, eps_inf, e_vib, huang_rhys = tuple(input_array[(3 * n):])
    elif (input_style =="v3"):
        #from E, mu,gamma,e_vib,huang_rhys
        #used for some solvent fittings
        energy_array = input_array[:n]
        mu_array = input_array[n:(2*n)]
        angle_array = None
        gamma,e_vib,huang_rhys = tuple(input_array[(2*n):])
        vol, eps_inf = None,None
    else:
        raise ValueError("Invalid input style string")
    return energy_array,mu_array,angle_array,vol, gamma, eps_inf, e_vib, huang_rhys











