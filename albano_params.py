import numpy as np
import dielectric_tensor as dt
import python_util
import gradient_descent as gd
'''
Backend for handling TDDFT results, which is used in other places such as ptpo_params.py
'''
cd_per_mdeg_to_cd_factor = 3.491e-5
def tddft_results_dummy(size):
    return TDDFT_RESULTS(np.ones(size),np.ones(shape = (size,3)),np.ones(size),"opt_dummy","tddft_dummy")
class TDDFT_RESULTS():
    def __init__(self,energies,dipole_matrix,osc_strength_array,optimization_str,tddft_str):
        '''

        :param energies:
        :param dipole_matrix:
        :param osc_strength_array: \propto \mu^2 \omega
        :param optimization_str:
        :param tddft_str:
        '''
        self.energies = energies.flatten()
        self.dip_mat = dipole_matrix
        self.osc_array = osc_strength_array.flatten()
        self.opt = optimization_str
        self.tddft = tddft_str
    def print_info(self):
        print("Energies (eV):"+str(self.energies))
        print("Dipoles (a.u.):"+str(self.dip_mat))
        print("Oscillator strengths (a.u.):"+str(self.osc_array))
        print("Optimization Method:"+self.opt)
        print("TDDFT Method:"+self.tddft)
    def vibronic_dressing(self,vib_index_array,vib_dist_array,huang_rhys_array,vib_modes = np.arange(4),to_zero_unselected = False):
        peaks_per_vib_dressing = np.size(vib_modes)
        total_peaks = peaks_per_vib_dressing*np.size(vib_index_array)+np.size(self.energies)-np.size(vib_index_array)
        new_energies = np.zeros(total_peaks)
        new_osc_str = np.zeros(total_peaks)
        new_dip_mat = np.zeros((total_peaks,3))
        start_index = 0
        end_index = 0
        for i in range(0,np.size(self.energies)):
            start_index = end_index
            if (i in vib_index_array):
                end_index = end_index+np.size(vib_modes)
                #print("start:end"+str(start_index)+":"+str(end_index))
                cur_index = np.argwhere(vib_index_array==i)
                vib_osc = python_util.remove_unnecessary_indices(dt.vib_spec_osc_str(self.osc_array[i], vib_modes, huang_rhys_array[cur_index]))
                energies_to_add = vib_modes * vib_dist_array[cur_index] + self.energies[i]
                #vib_osc_ratio_array = vib_osc / self.osc_array[i]
                if (self.dip_mat is not None):
                    init_dipole = python_util.remove_unnecessary_indices(self.dip_mat[i,:])
                    new_dipoles = dt.vib_spec_dip_mat(init_dipole,vib_modes,huang_rhys_array[cur_index])
                    new_dip_mat[start_index:end_index,:] = new_dipoles
                new_energies[start_index:end_index] = energies_to_add
                new_osc_str[start_index:end_index] = vib_osc
            else:
                end_index = end_index+1
                if (to_zero_unselected):nonselected_factor = 0
                else: nonselected_factor = 1
                if (self.dip_mat is not None):
                    new_dip_mat[start_index:end_index, :] = self.dip_mat[i,:]*nonselected_factor
                new_energies[start_index:end_index] = self.energies[i]*nonselected_factor
                new_osc_str[start_index:end_index] = self.osc_array[i]*nonselected_factor
        return TDDFT_RESULTS(new_energies,new_dip_mat,new_osc_str,self.opt,self.tddft)
    def linear_lorentzian_spec(self,spectrum,gamma_array):
        params = tuple(python_util.interweave_arrays(self.osc_array/self.energies,self.energies))
        return gd.lorenztian_dielectric_multi(spectrum,gamma_array,params)
    def linear_lorentzian_spec_set_amp(self,spectrum,gamma_array,amp_array):
        params = tuple(python_util.interweave_arrays(amp_array, self.energies))
        return gd.lorenztian_dielectric_multi(spectrum, gamma_array, params)
    def truncate_selection(self,select_array):
        truncated_results = TDDFT_RESULTS(self.energies[select_array],self.dip_mat[select_array,:],self.osc_array[select_array],
                                          self.opt,self.tddft)
        return truncated_results

class VIB_DRESSING():
    def __init__(self,vib_index_array,vib_dist_array,huang_rhys_array):
        self.vib_index = vib_index_array
        self.vib_dist = vib_dist_array
        self.huang_rhys = huang_rhys_array

