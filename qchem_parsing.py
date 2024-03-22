import numpy as np
import pymatgen.io.qchem.outputs as out
import pymatgen.io.qchem.utils as utils
import albano_params as ap
import python_util

'''
Parsing for Q-Chem (TM) output files. Uses Pymatgen (https://doi.org/10.1016/j.commatsci.2012.10.028) for the bulk of parsing,
though specific regex handling has been added for items of interest. 
'''

class TRANSITION_ENERGIES():
    '''
    Data transfer object for set of electronic transitions
    '''
    def __init__(self,excite_energies,total_energies,multiplicity,trans_moments,osc_strengths):
        '''
        Information regarding n electronic transition dipoles
        :param excite_energies: np.ndarray (n)
        :param total_energies: np.ndarray
        Total energy over time of convergence
        :param multiplicity: int
        :param trans_moments: np.ndarray (n,3)
        :param osc_strengths: np.ndarray (n)
        '''
        self.excite_energies = excite_energies.astype(np.double)
        self.total_energies = total_energies.astype(np.double)
        self.multiplicity = multiplicity
        self.trans_moments = trans_moments.astype(np.double)
        self.osc_strengths = osc_strengths.astype(np.double)
    #truncates components to allowed transitions
    def print_info(self):
        print("Energies (eV): "+str(self.excite_energies))
        print("Total Energies (eV): "+str(self.total_energies))
        print("Multiplicity: "+str(self.multiplicity))
        print("Dipole moments (a.u.):"+str(self.trans_moments))
        print("Oscillator strengths (a.u.):"+str(self.osc_strengths))
    def truncate(self,lower_bound = 0.01):
        '''truncates object to only include indices which are above some bound (default 0.01 e*a_0)'''
        indices_to_keep = np.argwhere(self.osc_strengths>lower_bound)
        self.excite_energies = self.excite_energies[indices_to_keep]
        self.total_energies = self.total_energies[indices_to_keep]
        self.trans_moments = self.trans_moments[indices_to_keep,:]
        self.osc_strengths = self.osc_strengths[indices_to_keep]

    def to_TDDFT_RESULTS(self,opt_str,tddft_str):
        return ap.TDDFT_RESULTS(self.excite_energies,self.trans_moments,self.osc_strengths,opt_str,tddft_str)


def parse_tddft_output(filename,solo= False):
    '''Parses TDDFT ouput from Q-Chem and returns a TRANSITION_ENERGIES object'''
    text_to_read = get_qchem_text(filename,solo)
    transition_energies_object = read_transition_energies_fast(text_to_read)
    return transition_energies_object

#assumes the calc was optimization, tddft
def get_qchem_text(filename,solo = False):
    '''Returns text from Q-Chem file'''
    if (solo):
        output = out.QCOutput.multiple_outputs_from_file(filename)[0]
    else:
        output = out.QCOutput.multiple_outputs_from_file(filename)[1]
    return output.text

#pyqchem parsing for quick implementation
def read_transition_energies_fast(text_to_read):
    '''
    Reads transition energies from text (see get_qchem_text()) and returns a TRANSITION_ENERGIES object.
    Note: Much of this functionality was not in Pymatgen parsing
    :param text_to_read:
    :return:
    '''
    exited_state_energies = utils.read_pattern(text_to_read, {
        "key": r" Excited state  *\d*: excitation energy \(eV\) =    ([\d,.]*)"}).get("key")
    excited_state_energies_array = np.array(exited_state_energies).flatten()
    excited_state_total_energies = utils.read_pattern(text_to_read,
                                                      {"key": r"Total energy for state  *\d*:[ ]* ([-,.\d]*)"}).get("key")
    excited_state_total_energies_array = np.array(excited_state_total_energies).flatten()
    multiplicity = utils.read_pattern(text_to_read,
                                                      {"key": r"Multiplicity: ([a-z,A-Z]*)"}).get("key")

    transition_moments = utils.read_pattern(text_to_read, {
        "key": r"Trans. Mom.:[ ]*([-.\d]*)[ ]*X[ ]*([-.\d]*) Y[ ]*([-.\d]*) Z"}).get("key")
    transition_moments_array = np.array(transition_moments)
    transition_moments_array = python_util.remove_unnecessary_indices(transition_moments_array)
    osc_strength = utils.read_pattern(text_to_read, {"key": r"Strength[ ]*:[ ]*([-.\d]*)"}).get("key")
    osc_strength_array= np.array(osc_strength).flatten()
    transition_energies_object = TRANSITION_ENERGIES(excited_state_energies_array,excited_state_total_energies_array,multiplicity,transition_moments_array,osc_strength_array)
    return transition_energies_object

class VIBRATIONAL_MODE():
    '''Data transfer class for vibrational information'''
    def __init__(self,index,frequency,force_const,reduced_mass,ir_active,ir_intensity,raman_active):
        self.index = index
        self.freq = frequency
        self.k = force_const
        self.mu = reduced_mass # in amu
        self.ir_active = bool(ir_active)
        self.ir_intensity = ir_intensity
        self.raman_active = bool(raman_active)

def str_list_to_bool_list(str_list):
    bool_list = []
    for str in str_list:
        clean_str = str.replace(" ", "").upper()
        if (clean_str == "YES" or clean_str == "TRUE"):
            bool_list.append(True)
        elif (clean_str == "NO" or clean_str == "FALSE"):
            bool_list.append(False)
        else:
            bool_list.append(None)
    return bool_list
#Assumes single vibronic region in document--for multijob outputs, break up outputs
def read_vibronics(text_to_read):
    '''Parses vibronic data from text string. See pymatgen.io.qchem.utils for more info'''
    vibronic_subtext = utils.read_pattern(text_to_read,{
        "key": r"VIBRATIONAL ANALYSIS ([\S\n\t\v ]*)Archival summary"}).get("key")
    mode = utils.read_pattern(vibronic_subtext,{
        "key": r"Mode:.*(\d)"}).get("key")
    freq = utils.read_pattern(vibronic_subtext,{
        "key":r"Frequency:.*([-.\d]*)"}).get("key")
    force_const = utils.read_pattern(vibronic_subtext,{
        "key":r"Force Cnst:.*([-.\d]*)"}).get("key")
    red_mass = utils.read_pattern(vibronic_subtext,{
        "key":r"Red. Mass:.*([-.\d]*)"}).get("key")
    ir_active = utils.read_pattern(vibronic_subtext,{
        "key":r"IR Active:[ ]*([A-Za-z]*)"}).get("key")
    ir_active = str_list_to_bool_list(ir_active)
    raman_active = utils.read_pattern(vibronic_subtext,{
        "key":r"IR Active:[ ]*([A-Za-z]*)"}).get("key")
    raman_active = str_list_to_bool_list(raman_active)
    ir_intensity = utils.read_pattern(vibronic_subtext,{
        "key":r"IR Intens:[ ]*([-.\d]*)"}).get("key")
    return VIBRATIONAL_MODE(mode,freq,force_const,red_mass,ir_active,ir_intensity,raman_active)
