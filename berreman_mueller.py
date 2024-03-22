import copy
import numpy as np
from numpy.linalg import inv
from numpy.linalg import multi_dot
import cholesteric
import mueller
import pyllama
import dielectric_tensor as dt
import warnings

'''Handling of Jones vectors and Mueller matrices, particularly those coming from
numeric calculations

Mostly an extension of mueller.py with specific focus on transfer/scattering matrix calcs

Also includes functions for the construction of cavity systems for transfer/scattering matrix
calcs 

Convention: (1,1j) is LHP, keeping pyllama's convention (most of the rest of the codebase reverts this)
'''

def get_intensity_from_jones_vector(jones_vector):
    '''Jones vector conversion to intensity. Polarization axis must be 0'''
    return np.abs(jones_vector[0,...])**2+np.abs(jones_vector[1,...])**2

def berre_dim_handling(berre_dim,reflection_matrix,transmission_matrix,input_vector):
    '''Converts jones reflection and transmission matrices of dimension 2-4 to transfer
    vectors for arbitrary polarization input'''
    if (berre_dim>8):
        raise ValueError("Berreman matrix dimension is too high")
    einsum_str_list = ['ij,j->i','ijk,j->ik','ijkl,j->ikl','ijkla,j->ikla','ijklab,j->iklab',
                       'ijklabc,j->iklabc','ijklabcd,j->iklabcd'] #supports up to dimension 8, which should be sufficient for nearly all cases
    r_vec = np.einsum(einsum_str_list[int(berre_dim-2)],reflection_matrix,input_vector)
    t_vec = np.einsum(einsum_str_list[int(berre_dim-2)],transmission_matrix,input_vector)
    return r_vec,t_vec

def reflection_transmission_from_amplitude_matrices(reflection_matrix,transmission_matrix,input_vector):
    '''
    From Jones T and R matrices, provides normalized reflected and transmitted intensities
    :param reflection_matrix:
    :param transmission_matrix:
    :param input_vector:
    :return: (np.ndarray, np.ndarray)
    '''
    berre_dim = np.ndim(reflection_matrix)
    inc_intensity = get_intensity_from_jones_vector(input_vector)
    r_vec,t_vec = berre_dim_handling(berre_dim,reflection_matrix,transmission_matrix,input_vector)
    r_intensity = get_intensity_from_jones_vector(r_vec)
    t_intensity = get_intensity_from_jones_vector(t_vec)
    #abolute values to avoid negative numerical answers
    return np.abs(r_intensity/inc_intensity), np.abs(t_intensity/inc_intensity)

def amplitude_matrix_ps_to_rl(r_matrix,t_matrix):
    '''
    converts reflection and jones matrices to rl basis
    note that rl here is in terms of receiver, which is pyllama convention but opposite of how most
    of the codebase works
    :param r_matrix: np.ndarray np.shape =(2,2,...)
    :param t_matrix: np.ndarray np.shape = (2,2,...)
    :return:
    '''
    A = np.array([ [1, 1],[-1j, 1j]])
    B = np.array([[1, 1],[1j, -1j]])
    mat_dim = np.ndim(r_matrix)
    if (mat_dim == 2):
        r_mat_rl = multi_dot(inv(A),r_matrix,B)
        t_mat_rl = multi_dot(inv(B),t_matrix,B)
    elif (mat_dim ==3 ):
        r_mat_rl = np.einsum("ij,jkl->ikl",inv(A),np.einsum("ijl,jk->ikl",
                                                            r_matrix,B))
        t_mat_rl = np.einsum("ij,jkl->ikl",inv(B),np.einsum("ijl,jk->ikl",
                                                            t_matrix,B))
    elif (mat_dim == 4):
        r_mat_rl = np.einsum("ij,jklx->iklx",inv(A),np.einsum("ijlx,jk->iklx",
                                                            r_matrix,B))
        t_mat_rl = np.einsum("ij,jklx->iklx",inv(B),np.einsum("ijlx,jk->iklx",
                                                            t_matrix,B))

    return r_mat_rl,t_mat_rl

def reflection_transmission_set(reflection_matrix,transmission_matrix,style = "amplitude",pol_style = "default"):
    '''From R and T matrices (either Jones or Intensity), provides characterization of various polarizations'''
    if (pol_style == "imag_y"):
        input_n, input_a, input_b = np.array([1, 1]) / np.sqrt(2), np.array([1, 1j])/np.sqrt(2), np.array([1, -1j])/np.sqrt(2)
    else: input_n, input_a, input_b=  np.array([1,1])/np.sqrt(2),np.array([1,0]),np.array([0,1])
    input_all = np.vstack([input_n,input_a,input_b])
    trans_all = np.zeros((3,np.size(transmission_matrix,axis = -1)))
    refl_all = np.zeros((3,np.size(reflection_matrix,axis = -1)))
    for i in np.arange(np.size(input_all,axis =0)):
        if (style == "amplitude"):
            refl_all[i,:], trans_all[i,:] = reflection_transmission_from_amplitude_matrices(reflection_matrix,transmission_matrix,input_all[i,:])
        elif (style == "intensity"):
            refl_all[i, :], trans_all[i, :] = reflection_transmission_from_intensity_matrices(reflection_matrix,transmission_matrix,input_all[i, :])
        else:
            raise ValueError("Invalid style")
    return refl_all, trans_all

def reflection_transmission_from_intensity_matrices(reflection_matrix,transmission_matrix,input_vector):
    '''Provides reflected and transmiteed intensities for arbitary matrices'''
    input_vector = input_vector/np.linalg.norm(input_vector)
    input_vector = input_vector**2 #conversion to intensity
    berre_dim = np.ndim(reflection_matrix)
    r_vec, t_vec = berre_dim_handling(berre_dim, reflection_matrix, transmission_matrix, input_vector)
    r_intensity = r_vec[0,...]+r_vec[1,...]
    t_intensity = t_vec[0,...]+t_vec[1,...]
    return r_intensity, t_intensity

def kron_vectorized(a,b):
    '''
    kronecker product implemented to take advantage of numpy
    vectorization when calculating a stack of products
    Much faster than a for loop calculating each stack
    :param a: np.ndarray (shape (N,N,X))
    :param b:  np.ndarray (shape (N,N,X))
    :return:
    '''
    i,j,x = a.shape
    k,l,y = b.shape
    return np.einsum('ijx,klx->ikjlx',a,b).reshape(i*j,k*l,x)

def stokes_from_jones_vector(jones_vector):
    '''
    Conversion of Jones vector to corresponding Stokes vector
    Eq. 3.30a in Ossikovski and Perez 2nd Ed, Polarized Light and the Mueller Matrix Approach
    Jones vector must be in the xy basis
    '''
    a_mat = 1 * \
            np.array([[1, 0, 0, 1],
                      [1, 0, 0, -1],
                      [0, 1, 1, 0],
                      [0, 1j, -1j, 0]])
    j_dim = np.ndim(jones_vector)
    if (j_dim == 1):
        kron_jones = np.kron(jones_vector, np.conjugate(jones_vector))
        coherency_vector= np.reshape(kron_jones,(4))
        stokes_vector = np.dot(a_mat,coherency_vector)
    elif (j_dim == 2):
        raise ValueError("2 Dim unsupported atm")
    elif (j_dim == 3):
        raise ValueError("3 Dim unsupported atm")
    return stokes_vector

def mueller_from_jones_matrix(jones_matrix):
    '''
    Conversion of Jones transmission matrix to corresponding Mueller-Jones matrix
    Eq. 3.32 in Ossikovski and Perez 2nd Ed
    Jones matrix must be in the xy basis
    '''
    a_mat = 1/np.sqrt(2)*\
            np.array([[1,0,0,1],
                      [1,0,0,-1],
                      [0,1,1,0],
                      [0,1j,-1j,0]])
    a_mat_inv = inv(a_mat)
    j_dim = np.ndim(jones_matrix)
    if (j_dim == 2):
        kron_jones = np.kron(jones_matrix,np.conjugate(jones_matrix))
        mueller_mat = np.dot(a_mat,np.dot(kron_jones,a_mat_inv))
    elif (j_dim == 3):
        kron_jones = kron_vectorized(jones_matrix,np.conjugate(jones_matrix))
        mueller_mat = np.einsum('ij,jkl->ikl',a_mat,np.einsum('ijl,jk->ikl',kron_jones,a_mat_inv))
    elif (j_dim>3 and j_dim<7):
        # this is nonideal but much easier to read than a truly general solution
        a_mat_einsum_list = ['ij,jklx->iklx','ij,jklxy->iklxy','ij,jklxyz->iklxyz']
        a_mat_inv_einsum_list = ['ijlx,jk->iklx','ijlxy,jk->iklxy','ijlxyz,jk->iklxyz']
        f_size = np.prod(np.array(jones_matrix.shape)[2:])#product of all elements except first two
        jones_reshaped  =np.reshape(jones_matrix,(2,2,f_size))
        kron_jones = kron_vectorized(jones_reshaped, np.conjugate(jones_reshaped))
        new_shape = (4,4)+jones_matrix.shape[2:]
        kron_jones = np.reshape(kron_jones,new_shape)
        list_idx = int(j_dim-4)
        mueller_mat = np.einsum(a_mat_einsum_list[list_idx], a_mat,
                                np.einsum(a_mat_inv_einsum_list[list_idx], kron_jones, a_mat_inv))

    return mueller_mat
#kept to maintain some compatiblity--use mueller.py in future
def norm_mueller_matrix_stack(mueller_matrix):
    warnings.warn("Use mueller.norm_mueller_matrix_stack() instead",DeprecationWarning)
    return mueller.norm_mueller_matrix_stack(mueller_matrix)
def factor_mean_abs_mueller_matrix_stack(mueller_matrix,mean_abs,length):
    warnings.warn("Use mueller.factor_mean_abs_mueller_matrix_stack() instead", DeprecationWarning)
    return mueller.factor_mean_abs_mueller_matrix_stack(mueller_matrix,mean_abs,length)
def logm_mueller_matrix_stack(mueller_matrix_stack):
    warnings.warn("Use mueller.logm_mueller_matrix_stack()  instead", DeprecationWarning)
    return mueller.logm_mueller_matrix_stack(mueller_matrix_stack)

class DBR_Params_Berre():
    '''
    Cavity parameters for Berreman simulation. Note that only mirrors are defined here, 
    not spacing (which is in sample parameters)
    '''
    def __init__(self,dbr_period,dbr_target_wavelength,dbr_eps_list,dbr_wl_nm_set_list = None):
        self.per = dbr_period
        self.lambda_t = dbr_target_wavelength
        self.eps_list = dbr_eps_list #full wl set
        self.wl_nm_set_list = dbr_wl_nm_set_list
        if (len(dbr_eps_list)>1):
            if (np.ndim(dbr_eps_list[0])==2):
                self.eps1, self.eps2 = np.real(dbr_eps_list[0][0,0]),np.real(dbr_eps_list[1][0,0])
            elif (np.ndim(dbr_eps_list[0])==3):
                if (dbr_wl_nm_set_list is None): dbr_wl_nm_set_list = [None,None]
                self.eps1 = np.real(extract_tensor_wl_interpolated(dbr_eps_list[0],dbr_wl_nm_set_list[0],self.lambda_t))
                self.eps2 = np.real(extract_tensor_wl_interpolated(dbr_eps_list[1],dbr_wl_nm_set_list[1],self.lambda_t))
            else: raise ValueError("Invalid eps tensor dimension")
            thickness = 1/np.sqrt(np.array([self.eps1,self.eps2]))*self.lambda_t/4
            self.thickness_list = thickness.tolist()
        else:
            raise ValueError("DBR must be initialized with two dielectric tensors")
    def get_full_eps_list(self):
        return self.eps_list

def extract_tensor_wl_interpolated(eps_tensor,wl_array,target_wl,wl_interpolation_res = 101):
    eps_dim = np.ndim(eps_tensor)
    if (eps_dim == 2):
        return eps_tensor
    elif (eps_dim== 3):
        if (wl_array is None):
            return eps_tensor[0,0,0]
        else:
            target_idx = np.argmin(np.abs(wl_array-target_wl))
            eps_00_region = eps_tensor[0,0,target_idx-1:target_idx+2]
            wl_region = wl_array[target_idx-1:target_idx+2]
            high_res_wl_region = np.linspace(wl_region[0],wl_region[-1],wl_interpolation_res)
            eps_00_interpolated = np.interp(high_res_wl_region,wl_region,eps_00_region)
            new_target_idx = np.argmin(np.abs(high_res_wl_region-target_wl))
            return eps_00_interpolated[new_target_idx]

def extract_tensor_wl(tensor_set,wl_set,target_wl):
    idx = np.argmin(np.abs(wl_set-target_wl))
    return tensor_set[:,:,idx]



def create_DBR_cavity_params_list(dbr_params, sample_params,mirror_align=  True,second_dbr_params = None,inter_mirror_params = None,half_cavity = False,
                                  dbr_params_flipped = None,second_dbr_params_flipped = None):
    '''
    :param dbr_params: DBR_Params_Berre()
    :param sample_params: Sample_Params_Berre()
    :param mirror_align: Boolean: whether the dbrs orient with the same face into the cavity or not
    defaults to symmetric (i.e., True)
    :param second_dbr_params: DBR_Paramrs_Berre()
    For a double dbr mirror, produces a second dbr mirror exterally to the cavity if not None
    :return: list (np.ndarray), list (np.ndarray), list (np.ndarray)
    '''
    if (dbr_params_flipped is None):
        dbr_params_flipped = dbr_params
    if (second_dbr_params_flipped is None):
        second_dbr_params_flipped = second_dbr_params
    eps_total = []
    thick_total = []
    wl_nm_set= []
    second_dbr_eps_total = []
    second_dbr_thick_total = []
    second_dbr_wl_nm_set_total = []
    if (second_dbr_params is not None):
        dbr_eps_2 =second_dbr_params_flipped.get_full_eps_list()
        dbr_n_per_2 = second_dbr_params_flipped.per
        dbr_thick_2 = second_dbr_params_flipped.thickness_list
        for n in range(dbr_n_per_2):
            second_dbr_eps_total.extend(dbr_eps_2)
            second_dbr_thick_total.extend(dbr_thick_2)
            if (second_dbr_params.wl_nm_set_list):
                second_dbr_wl_nm_set_total.extend(second_dbr_params.wl_nm_set_list)
        if (not half_cavity):  # that is, if the full cavity is being modelled by default
            if (mirror_align == True):
                eps_total.extend(second_dbr_eps_total[::-1])
                thick_total.extend(second_dbr_thick_total[::-1])
                wl_nm_set.extend(second_dbr_wl_nm_set_total[::-1])
            else:
                eps_total.extend(second_dbr_eps_total)
                thick_total.extend(second_dbr_thick_total)
                wl_nm_set.extend(second_dbr_wl_nm_set_total)
    inter_wl_total = []
    if (inter_mirror_params is not None):
        if (not half_cavity):  # that is, if the full cavity is being modelled by default
            eps_total.extend(inter_mirror_params.eps_list)
            thick_total.extend(inter_mirror_params.thickness_list)
            if (inter_mirror_params.eps_wl_array):
                inter_wl_total = inter_mirror_params.eps_wl_array
            wl_nm_set.extend(inter_wl_total)
    dbr_eps = dbr_params_flipped.get_full_eps_list()
    dbr_n_per = dbr_params_flipped.per
    dbr_thick = dbr_params_flipped.thickness_list
    dbr_eps_total = []
    dbr_thick_total = []
    dbr_wl_nm_set_total = []
    for n in range(dbr_n_per):
        dbr_eps_total.extend(dbr_eps)
        dbr_thick_total.extend(dbr_thick)
        if (dbr_params.wl_nm_set_list):
            dbr_wl_nm_set_total.extend(dbr_params.wl_nm_set_list)
    if (not half_cavity):  # that is, if the full cavity is being modelled by default
        if (mirror_align == True):
            eps_total.extend(dbr_eps_total[::-1])
            thick_total.extend(dbr_thick_total[::-1])
            wl_nm_set.extend(dbr_wl_nm_set_total[::-1])
        else:
            eps_total.extend(dbr_eps_total)
            thick_total.extend(dbr_thick_total)
            wl_nm_set.extend(dbr_wl_nm_set_total)

    sample_eps = sample_params.eps_list
    sample_thick = sample_params.thickness_list
    eps_total.extend(sample_eps)
    thick_total.extend(sample_thick)
    if (len(wl_nm_set)>0):
        if (isinstance(sample_params.eps_wl_array,list)):
            wl_nm_set.extend(sample_params.eps_wl_array)
        elif (isinstance(sample_params.eps_wl_array,np.ndarray)):
            wl_nm_set.append([sample_params.eps_wl_array])
    else: wl_nm_set = sample_params.eps_wl_array

    dbr_eps = dbr_params.get_full_eps_list()
    dbr_n_per = dbr_params.per
    dbr_thick = dbr_params.thickness_list
    dbr_eps_total = []
    dbr_thick_total = []
    dbr_wl_nm_set_total = []
    for n in range(dbr_n_per):
        dbr_eps_total.extend(dbr_eps)
        dbr_thick_total.extend(dbr_thick)
        if (dbr_params.wl_nm_set_list):
            dbr_wl_nm_set_total.extend(dbr_params.wl_nm_set_list)
    eps_total.extend(dbr_eps_total)
    thick_total.extend(dbr_thick_total)
    wl_nm_set.extend(dbr_wl_nm_set_total)

    inter_wl_total = []
    if (inter_mirror_params is not None):
        eps_total.extend(inter_mirror_params.eps_list)
        thick_total.extend(inter_mirror_params.thickness_list)
        if (inter_mirror_params.eps_wl_array):
            inter_wl_total = inter_mirror_params.eps_wl_array
        wl_nm_set.extend(inter_wl_total)
    if (second_dbr_params is not None):
        second_dbr_eps_total = []
        second_dbr_thick_total = []
        second_dbr_wl_nm_set_total = []
        dbr_eps_2 = second_dbr_params.get_full_eps_list()
        dbr_n_per_2 = second_dbr_params.per
        dbr_thick_2 = second_dbr_params.thickness_list
        for n in range(dbr_n_per_2):
            second_dbr_eps_total.extend(dbr_eps_2)
            second_dbr_thick_total.extend(dbr_thick_2)
            if (second_dbr_params.wl_nm_set_list):
                second_dbr_wl_nm_set_total.extend(second_dbr_params.wl_nm_set_list)
        eps_total.extend(second_dbr_eps_total)
        thick_total.extend(second_dbr_thick_total)
        wl_nm_set.extend(second_dbr_wl_nm_set_total)
    return eps_total, thick_total, wl_nm_set

def DBR_simulation_lists_from_params(dbr_params,sample_params,dbr_style = "default",second_dbr_params = None,half_cavity = False,
                                     dbr_params_flipped = None,second_dbr_params_flipped= None):
    return create_DBR_cavity_params_list(dbr_params, sample_params,mirror_align=  True,
                                         second_dbr_params=second_dbr_params,half_cavity=half_cavity,
                                         dbr_params_flipped = dbr_params_flipped,second_dbr_params_flipped = second_dbr_params_flipped)


def create_dual_mirror_params_list(eps_high,eps_mid,thick_high,thick_mid,sample_params):
    '''
    Creates the phenemological system of a system list so
    |eps_high| sample |eps_mid|eps_high|
    :param eps_high: np.ndarray
    :param eps_mid: np.npdarray
    :param sample_params: Sample_Params()
    :return: list
    '''
    eps_total = [eps_high]
    thick_total = [thick_high]
    wl_nm_set= [0] #null index
    sample_eps = sample_params.eps_list
    sample_thick = sample_params.thickness_list
    eps_total.extend(sample_eps)
    thick_total.extend(sample_thick)
    if (isinstance(sample_params.eps_wl_array,list)):
        wl_nm_set.extend(sample_params.eps_wl_array)
    elif (isinstance(sample_params.eps_wl_array,np.ndarray)):
        wl_nm_set.append([sample_params.eps_wl_array])

    eps_total.extend([eps_mid,eps_high])
    thick_total.extend(([thick_mid,thick_high]))
    wl_nm_set.extend([0,0]) #null index
    return eps_total, thick_total, wl_nm_set

class Sample_Params_Berre():
    '''
    Sample parameters for Berreman simulation. Note that air spacers count as part of sample here
    '''
    def __init__(self,eps_list,thickness_list,eps_wl_array):
        self.eps_list = eps_list
        self.thickness_list = thickness_list
        self.eps_wl_array = eps_wl_array
        
def cavity_berreman_angle_sweep(dbr_params,sample_params,spec_nm_set,angle_set,circ = True,n_entry = 1,n_exit = 1,type = "amplitude",talk = False,dbr_style = "default",half_cavity = False,
                                dbr_params_flipped = None):
    '''
    :param dbr_params: either a DBR_Params_Berre() or list of DBR_Params_Berre()
    :param sample_params: Sample_Params_Berre()
    :param spec_nm_set: np.ndarray
    :param angle_set: np.ndarray
    :param circ: bool
    :param n_entry: scalar
    :param n_exit: scalar
    :param type: str
    :param talk: bool : Indicates whether to print spectral characterization progress
    :param dbr_style: str
    :return:
    '''
    dbr_params_obj, second_dbr_params_obj, dbr_params_obj_flipped, second_dbr_params_obj_flipped, matrix_type = dbr_params_handling(dbr_params,dbr_params_flipped,circ)
    eps_total_list, thick_total_list,wl_nm_list = DBR_simulation_lists_from_params(dbr_params_obj,sample_params,dbr_style= dbr_style,second_dbr_params=second_dbr_params_obj,half_cavity=half_cavity,
                                                                                   dbr_params_flipped = dbr_params_obj_flipped,second_dbr_params_flipped = second_dbr_params_obj_flipped)
    return general_berreman_angle_sweep(eps_total_list,thick_total_list,wl_nm_list,spec_nm_set,angle_set,circ,n_entry= n_entry,n_exit = n_exit,type = type,talk = talk,matrix_type=matrix_type)

def dbr_params_handling(dbr_params,dbr_params_flipped,circ):
    '''
    :param dbr_params:
    :param dbr_params_flipped:
    :param circ:
    :return:
    '''
    if (isinstance(dbr_params,list)):
        dbr_params_obj, second_dbr_params_obj = dbr_params[0],dbr_params[1]
    else:dbr_params_obj,second_dbr_params_obj = dbr_params, None
    if (dbr_params_flipped is None):
        dbr_params_obj_flipped,second_dbr_params_obj_flipped = None,None
    else:
        if (isinstance(dbr_params_flipped, list)):
            dbr_params_obj_flipped, second_dbr_params_obj_flipped = dbr_params_flipped[0], dbr_params_flipped[1]
        else:
            dbr_params_obj_flipped, second_dbr_params_obj_flipped = dbr_params_flipped, None
    if (circ):matrix_type = "rl"
    else: matrix_type = "ps"
    return dbr_params_obj,second_dbr_params_obj,dbr_params_obj_flipped,second_dbr_params_obj_flipped,matrix_type
def strained_dielectric_isotropic(eps_constant,stress_tensor,elastic_material):
    '''
    Provides strained dielectric tensor
    :param eps_constant:
    :param stress_tensor:
    :param elastic_material: dt.Elastic_Material
    :return:
    '''
    strain_tensor = dt.strain_tensor_hookes_law_isotropic_from_stress(stress_tensor,elastic_material.get_stiffness_matrix())
    electrostriction_params = dt.get_electrostriction_parameters(eps_constant,material_type="isotropic")
    eps_tensor = dt.strain_dielectric_tensor(eps_constant, strain_tensor, electrostriction_params)
    return eps_tensor

def create_pyllama_spectrum_dict(cur_angle,model_args,**other_args):
    model_args = {**model_args,**other_args}
    model_args.update({"theta_in_rad": cur_angle})
    return model_args

def general_berreman_angle_sweep(eps_total_list,thick_total_list,wl_nm_list,spec_nm_set,angle_set,circ,n_entry = 1,n_exit =1,type="amplitude",talk = False,matrix_type = "ps",
                                 model_type = "StackModel",**other_args):
    '''
    For some dielectric system, calculates a spectrume over a range of angles of incidence
    :param eps_total_list:
    :param thick_total_list:
    :param wl_nm_list:
    :param spec_nm_set:
    :param angle_set:
    :param circ:
    :param n_entry:
    :param n_exit:
    :param type:
    :param talk:
    :param matrix_type:
    :param model_type:
    :param other_args:
    :return:
    '''
    r_matrix_set  = np.zeros((2,2,np.size(spec_nm_set),np.size(angle_set)),dtype = np.cdouble)
    t_matrix_set = np.zeros((2,2,np.size(spec_nm_set),np.size(angle_set)),dtype = np.cdouble)
    for i in range(np.size(angle_set)):
        cur_angle = angle_set[i]
        model_args = create_pyllama_spectrum_dict(cur_angle,dict(eps_set = eps_total_list,
                        wl_eps_set = wl_nm_list,eps_list =eps_total_list,thickness_nm_list = thick_total_list,
                        n_entry = n_entry,n_exit = n_exit,rotangle_rad= 0,rotaxis = "z"),**other_args)
        cur_spectrum = pyllama.Spectrum(spec_nm_set,model_type,
                                model_args)
        cur_r,cur_t = get_refl_trans_matrix_spectra(cur_spectrum,coef_type=type,circ=  circ,talk= talk,method = "SM")
        r_matrix_set[:,:,:,i],t_matrix_set[:,:,:,i] = cur_r, cur_t
    return r_matrix_set,t_matrix_set
def rotate_rank2_tensor_stack(rotation_matrix,tensor):
    '''
    :param rotation_matrix:
    :param tensor:
    :return:
    '''
    return np.einsum("ij,jkl->ikl", rotation_matrix,np.einsum("ijl,jk->ikl", tensor, np.transpose(rotation_matrix)))

def rotate_rank2_tensor(rotation_matrix,tensor):
    '''
    :param rotation_matrix:
    :param tensor:
    :return:
    '''
    return np.einsum("ij,jk->ik", rotation_matrix,np.einsum("ij,jk->ik", tensor, np.transpose(rotation_matrix)))

def rotate_rank2_tensor_general(rotation_matrix,tensor):
    '''
    :param rotation_matrix:
    :param tensor:
    :return:
    '''
    if (np.ndim(tensor) ==2):return rotate_rank2_tensor(rotation_matrix,tensor)
    elif (np.ndim(tensor)==3): return rotate_rank2_tensor_stack(rotation_matrix,tensor)
    else: raise ValueError("Invalid tensor dimension")

def rotate_tensor_list(rotation_matrix,tensor_list):
    rotated_tensor_list = []
    for i in range(len(tensor_list)):
        cur_tensor = tensor_list[i]
        rotated_tensor = rotate_rank2_tensor_general(rotation_matrix, cur_tensor)
        rotated_tensor_list.append(rotated_tensor)
    return rotated_tensor_list

class MATERIAL_STACK():
    def __init__(self,eps_list,thick_list,wl_nm_list,rho_list = None,rhop_list = None,mu_list = None):
        '''
        Note: lists can have only a single element, in which this represents a single slab of material
        :param eps_list:
        :param thick_list:
        :param wl_nm_list:
        :param rho_list:
        :param rhop_list:
        :param mu_list:
        '''
        self.eps_list = eps_list
        self.thick_list = thick_list
        self.wl_nm_list = wl_nm_list
        self.rho_list = rho_list
        self.rhop_list = rhop_list
        self.mu_list = mu_list
    def get_rotated_tensors(self,rotation_matrix):
        '''
        Provides tuple of the lists of rotated tensors. Does not mutate object
        :param rotation_matrix:
        :return: tuple
        '''
        rotated_eps_list = rotate_tensor_list(rotation_matrix,self.eps_list)
        rotated_tuple_to_return = (rotated_eps_list,)
        if (self.rho_list is not None):
            rotated_rho_list = rotate_tensor_list(rotation_matrix,self.rho_list)
            rotated_tuple_to_return = rotated_tuple_to_return+(rotated_rho_list,)
        if (self.rhop_list is not None):
            rotated_rhop_list = rotate_tensor_list(rotation_matrix,self.rhop_list)
            rotated_tuple_to_return = rotated_tuple_to_return+(rotated_rhop_list,)
        if (self.mu_list is not None):
            rotated_mu_list = rotate_tensor_list(rotation_matrix,self.mu_list)
            rotated_tuple_to_return = rotated_tuple_to_return+(rotated_mu_list,)
        return rotated_tuple_to_return
    def get_rotated_object(self,rotation_matrix):
        rotated_material_stack = copy.deepcopy(self)
        rotated_material_stack.rotate_tensors(rotation_matrix)
        return rotated_material_stack
    def rotate_tensors(self,rotation_matrix):
        '''
        Mutates object by rotating composite tensors
        :param rotation_matrix:
        :return:
        '''
        self.eps_list= rotate_tensor_list(rotation_matrix,self.eps_list)
        if (self.rho_list is not None):
            self.rho_list = rotate_tensor_list(rotation_matrix,self.rho_list)
        if (self.rhop_list is not None):
            self.rhop_list = rotate_tensor_list(rotation_matrix,self.rhop_list)
        if (self.mu_list is not None):
            self.mu_list = rotate_tensor_list(rotation_matrix,self.mu_list)

def get_refl_trans_matrix_spectra(cur_spectrum,coef_type,circ=False,talk = False,method = "SM"):
    if (circ ==True):matrix_type = "rl"
    else: matrix_type = "ps"
    if (coef_type == "amplitude"):
        cur_spectrum.calculate_refl_trans_coefs(circ=circ, method=method, talk=talk)
        cur_r, cur_t = cur_spectrum.export_r_t_matrices(type="amplitude", matrix_type=matrix_type)
    elif (coef_type == "intensity"):
        cur_spectrum.calculate_refl_trans(circ=circ, method=method, talk=talk)
        cur_r, cur_t = cur_spectrum.export_r_t_matrices(type="intensity", matrix_type=matrix_type)
    return cur_r,cur_t


def angular_charactetrization_both_sides(material_stack_object,azimuthal_set,polar_angle_set,spec_nm_set,circ,n_entry = 1,n_exit =1,coef_type="amplitude",talk = False,
                                 model_type = "StackModel",method = "SM",chole = cholesteric.Cholesteric(),**other_args):
    '''
    Characterizes two sides of some sample.
    Currently only supports "StackModel" which for lists of length 1 is the same as the "SlabModel",
    and "CholestericModel"
    :param azimuthal_array:
    :param polar_angle_array:
    :param spec_nm_set:
    :param circ:
    :param n_entry:
    :param n_exit:
    :param type:
    :param talk:
    :param matrix_type:
    :param model_type:
    :param other_args:
    :return:
    '''
    #indices are (2,2) Jones matrix, measured spectrum,azimuthal angle, polar angle, side (top/bottom)
    r_matrix_set = np.zeros((2, 2, np.size(spec_nm_set), np.size(azimuthal_set),np.size(polar_angle_set),2), dtype=np.cdouble)
    t_matrix_set = np.zeros((2, 2, np.size(spec_nm_set), np.size(azimuthal_set),np.size(polar_angle_set),2), dtype=np.cdouble)
    if (material_stack_object.rho_list is not None or material_stack_object.rhop_list is not None or material_stack_object.mu_list is not None):
        tensor_handling = "full"
    else: tensor_handling = "simple"
    for a in range(2):
        if (a==1):
            x_flip_rotation_matrix = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) #flips object about its x axis
            material_stack_object.rotate_tensors(rotation_matrix=x_flip_rotation_matrix)
        for i in range(np.size(azimuthal_set)):
            cur_azimuthal_angle = azimuthal_set[i]
            for j in range(np.size(polar_angle_set)):
                cur_polar_angle = polar_angle_set[j]
                rotation_matrix = pyllama.euler_rotation_matrix(0,0,cur_polar_angle)
                rotated_material_stack = material_stack_object.get_rotated_object(rotation_matrix)
                if (model_type == "CholestericModel"):
                    other_args.update({'chole':chole,'eps':rotated_material_stack.eps_list[0],
                                       'wl_eps_set': rotated_material_stack.wl_nm_list[0],
                                       'eps_set': rotated_material_stack.eps_list[0]})
                if (tensor_handling=="full"):
                    if (model_type == "StackModel"):
                        other_args.update({'mag_elec_list':rotated_material_stack.rho_list,'mag_elec_set':rotated_material_stack.rho_list,
                                       'elec_mag_list':rotated_material_stack.rhop_list,'elec_mag_set':rotated_material_stack.rhop_list,
                                       'mu_list':rotated_material_stack.mu_list,'mu_set':rotated_material_stack.mu_list})
                    elif (model_type == "CholestericModel"):
                        #converts material_stack to appropriate first slice of cholesteric liquid crystal
                        other_args.update({'mag_elec':rotated_material_stack.rho_list[0],'mag_elec_set':rotated_material_stack.rho_list[0],
                                       'elec_mag':rotated_material_stack.rhop_list[0],'elec_mag_set':rotated_material_stack.rhop_list[0],
                                       'mu':rotated_material_stack.mu_list[0],'mu_set':rotated_material_stack.mu_list[0]})
                    else:raise ValueError("Unsupported model type")
                model_args = create_pyllama_spectrum_dict(cur_azimuthal_angle, dict(eps_set=rotated_material_stack.eps_list,
                                                                          wl_eps_set=rotated_material_stack.wl_nm_list, eps_list=rotated_material_stack.eps_list,
                                                                          thickness_nm_list=rotated_material_stack.thick_list,
                                                                          n_entry=n_entry, n_exit=n_exit, rotangle_rad=0,
                                                                          rotaxis="z"), **other_args) #other_args overwrites the first dictionary by design
                cur_spectrum = pyllama.Spectrum(spec_nm_set, model_type,
                                                model_args)
                cur_r,cur_t = get_refl_trans_matrix_spectra(cur_spectrum,coef_type=coef_type,circ = circ,talk = talk,method=method)
                r_matrix_set[:, :, :, i,j,a], t_matrix_set[:, :, :, i,j,a] = cur_r, cur_t
    return r_matrix_set,t_matrix_set

def plot_heatmap_spectra(x_array,y_array,z_matrix,filename = "",figure = None, axis = None,style = "pcolormesh",**kwargs):
    '''kept to avoid deprecation of berreman_mueller.plot_heatmap_spectra_calls
    Deprecated: use mueller.plot_heatmap_spectra_directly
    '''
    mueller.plot_heatmap_spectra(x_array,y_array,z_matrix,filename = filename,figure = figure, axis = axis,style = style,**kwargs)


class DBR_Results():
    def __init__(self,data,spec_nm,set_indicators):
        self.data = data
        self.spec_nm = spec_nm
        self.indicators = set_indicators



def characterize_solo_sample_ps(dielectric_tensor,wl_nm_array,spec,thickness_nm,theta_in_rad = 0,talk = "True"):
    eps_tensor_set = dielectric_tensor
    eps_tensor = dielectric_tensor[:, :, 0]
    wl_eps_set = dt.nm_to_eV(spec)
    n_entry = 1
    n_exit = 1
    spectrum = pyllama.Spectrum(wl_nm_array, "SlabModel",
                                dict(eps_set=eps_tensor_set,
                                     wl_eps_set=wl_eps_set,
                                     eps=eps_tensor,
                                     thickness_nm=thickness_nm,
                                     n_entry=n_entry,
                                     n_exit=n_exit,
                                     theta_in_rad=theta_in_rad,
                                     rotangle_rad=0,
                                     rotaxis="z"))
    spectrum.calculate_refl_trans_coefs(circ=False, method="SM", talk=talk)
    r, t = spectrum.export_r_t_matrices(type="amplitude", matrix_type='ps')
    return r,t

def characterize_solo_sample_intensity_ps(dielectric_tensor,wl_nm_array,spec,thickness_nm,theta_in_rad = 0,talk = "True"):
    eps_tensor_set = dielectric_tensor
    eps_tensor = dielectric_tensor[:, :, 0]
    wl_eps_set = dt.nm_to_eV(spec)
    n_entry = 1
    n_exit = 1
    spectrum = pyllama.Spectrum(wl_nm_array, "SlabModel",
                                dict(eps_set=eps_tensor_set,
                                     wl_eps_set=wl_eps_set,
                                     eps=eps_tensor,
                                     thickness_nm=thickness_nm,
                                     n_entry=n_entry,
                                     n_exit=n_exit,
                                     theta_in_rad=theta_in_rad,
                                     rotangle_rad=0,
                                     rotaxis="x"))
    spectrum.calculate_refl_trans(circ=False, method="SM", talk=talk)
    r, t = spectrum.export_r_t_matrices(type="intensity", matrix_type='ps')
    return r,t

def characterize_solo_sample_rl(dielectric_tensor,wl_nm_array,spec,thickness_nm,theta_in_rad = 0,talk = "True"):
    '''
    Note that pyllama CP convention is different that ours
    :param dielectric_tensor:
    :param wl_nm_array:
    :param spec:
    :param thickness_nm:
    :param theta_in_rad:
    :param talk:
    :return:
    '''
    eps_tensor_set = dielectric_tensor
    eps_tensor = dielectric_tensor[:, :, 0]
    wl_eps_set = dt.nm_to_eV(spec)
    n_entry = 1
    n_exit = 1
    spectrum = pyllama.Spectrum(wl_nm_array, "SlabModel",
                                dict(eps_set=eps_tensor_set,
                                     wl_eps_set=wl_eps_set,
                                     eps=eps_tensor,
                                     thickness_nm=thickness_nm,
                                     n_entry=n_entry,
                                     n_exit=n_exit,
                                     theta_in_rad=theta_in_rad,
                                     rotangle_rad=0,
                                     rotaxis="x"))
    spectrum.calculate_refl_trans_coefs(circ=True, method="SM", talk=talk)
    r, t = spectrum.export_r_t_matrices(type="amplitude", matrix_type='rl')
    return r,t

def mueller_matrix_suite_from_refl_trans_amplitude_matrices_basis_ps(r,t):
    #assumes ps is xy basis
    #rhp/lhp in convention of view of receiver where (1,1j) is RHP
    r_l, t_l = reflection_transmission_from_amplitude_matrices(r, t, np.array([1, -1j]) / np.sqrt(2))
    r_r, t_r = reflection_transmission_from_amplitude_matrices(r, t, np.array([1, 1j]) / np.sqrt(2))
    r_set, t_set = reflection_transmission_set(r, t)
    mm = np.real(mueller_from_jones_matrix(t))
    return mm,t_set,t_l,t_r,r_l,r_r

def get_absorbance_sets(t_l,t_r,t_set_ps):
    #calculating absorbance as a absorbance of a mixed polarized wave
    #absorbance_smm = -np.log(t_set[0, :])
    #absobance as an average
    absorbance_smm = -1/2*(np.log((t_set_ps[1,:]))+np.log(t_set_ps[2,:]))
    absorbance_smm_circ = -1/2*(np.log(t_l)+np.log(t_r))
    trans_lr_diff_avg = (t_l - t_r) / 2
    return absorbance_smm,absorbance_smm_circ,trans_lr_diff_avg

def extract_mueller_matrix_factors(mueller_mat,t_set_ps,abs_smm,abs_smm_circ):
    '''Factors Mueller matrix in a variety of ways '''
    normed_mm = mueller.norm_mueller_matrix_stack(mueller_mat)
    smm_factored_mm = mueller.factor_mean_abs_mueller_matrix_stack(mueller_mat,abs_smm,1)
    smm_factored_mm_circ = mueller.factor_mean_abs_mueller_matrix_stack(mueller_mat,abs_smm_circ,1)
    smm_factored_mm_xy = mueller.factor_mean_abs_mueller_matrix_stack(mueller_mat,-np.log(t_set_ps[0,:]),1)
    log_mm = mueller.logm_mueller_matrix_stack(mueller_mat)
    abs_from_logm = -1/4*np.einsum("iik->k",log_mm)
    factored_log_mm = mueller.factor_mean_abs_mueller_matrix_stack(mueller_mat,abs_from_logm,1)
    return normed_mm,log_mm,smm_factored_mm,smm_factored_mm_circ,smm_factored_mm_xy,factored_log_mm



def get_m03_raw(t):
    ''',
    Provides m_03 from a transfer matrix in xy basis
    :param t:
    :return:
    '''
    t_xx, t_xy,t_yx,t_yy= t[0,0,:] ,t[0,1,:],t[1,0,:],t[1,1,:]
    return 1j*(np.conjugate(t_xx)*t_xy+np.conjugate(t_yx)*t_yy-np.conjugate(t_xy)*t_xx-np.conjugate(t_yy)*t_yx)

def extract_sweep_intensities_max(intensity_matrix,intensity_matrix_to_max,x_axis_array,y_axis_array,y_bounds = None):
    '''
    Finds and extracts intensities from 2D matrices where their average contribution is greatest within
    the bounds if provided. If bounds are 2D arrays, they indicate bounds that shift wrt the other axis
    to allow for variable objects such as bands
    :param intensity_matrix
    :params intensity_matrix_to_max
    :param x_axis_array: np.ndarray
    :param y_axis_array: np.ndarray
    :param x_bounds: np.ndarray or None
    :params y_bounds: np.ndarray or None
    :return:
    '''
    if (np.size(intensity_matrix,axis = 0) != np.size(x_axis_array) and np.size(intensity_matrix,axis = 0) == np.size(y_axis_array)):
        #alligns things if there is a transpose error in input
        intensity_matrix = intensity_matrix.T
        intensity_matrix_to_max = intensity_matrix_to_max.T
    if (np.size(intensity_matrix,axis = 0) != np.size(x_axis_array)):
        raise ValueError("Intensity matrix x axis must be same size as x axis array")
    if (np.size(intensity_matrix,axis = 1)!= np.size(y_axis_array)):
        raise ValueError("Intensity matrix y axaix must be same size as y_axis array")
    mask_matrix = np.ones(np.shape(intensity_matrix),dtype = bool) #begin by masking everything
    if (y_bounds is None):
        mask_matrix[:,:] = False #no masking
    elif (np.ndim(y_bounds) ==1 and np.size(y_bounds)==2):
        y_lb_idx = np.argmin(np.abs(y_axis_array-y_bounds[0]))
        y_ub_idx = np.argmin(np.abs(y_axis_array-y_bounds[1]))
        mask_matrix[:,y_lb_idx:y_ub_idx] = False
    elif (np.ndim(y_bounds) ==2):
        if (np.size(y_bounds,axis= 0) != np.size(x_axis_array)):
            raise ValueError("2 dimensional y bounds must have same number of bounds as x axis size")
        y_size = np.size(y_axis_array)
        x_size = np.size(x_axis_array)
        y_axis_tiled = np.tile(y_axis_array,(x_size,1))
        y_lb_tiled = np.tile(y_bounds[:,0],(y_size,1)).T
        y_ub_tiled = np.tile(y_bounds[:,1],(y_size,1)).T
        y_lb_idx_array = np.argmin(np.abs(y_axis_tiled-y_lb_tiled),axis = 1)
        y_ub__idx_array = np.argmin(np.abs(y_axis_tiled-y_ub_tiled),axis = 1)
        for i in range(0,x_size): #probably should vectorize this but couldn't find a simple way
            mask_matrix[i,y_lb_idx_array[i]:y_ub__idx_array[i]] = False
    matrix_masked = np.ma.masked_array(intensity_matrix_to_max,mask_matrix)
    selected_sum_masked_idx_array = np.argmax(matrix_masked,axis = 1) #finds y value of maximum signal
    intensity_selected_array = np.zeros(np.size(x_axis_array))
    for i in range(np.size(x_axis_array)):
        intensity_selected_array[i] = intensity_matrix[i,selected_sum_masked_idx_array[i]]
    return intensity_selected_array


def create_band_bounds(wl_nm_normal,angle_array,effective_n_cavity= 1.5,percent_offset = .025):
    '''
    :param wl_nm_normal:
    :param angle_array:
    :param dispersion_factor:
    :param percent_offset:
    :return: np.ndarray
    '''
    band = wl_nm_normal*np.sqrt(1-np.sin(angle_array)**2/effective_n_cavity**2)
    band_bounds  = np.zeros((np.size(angle_array),2))
    band_bounds[:,0] = band*(1-percent_offset)
    band_bounds[:,1] = band*(1+percent_offset)
    return band_bounds

class Intensity_Results():
    def __init__(self,intensity_stack,x_array,y_array,labels,mu_1 = 0,rotation_matrix = np.identity(3),dipole_params =None,description = None):
        '''
        :param intensity_stack: np.ndarray (x,y,pol)
        :param x_array: np.ndarray (x)
        :param y_array: np.ndarray (y)
        :param labels: np.ndarray (pol)
        :param mu_1: np.ndarray (2) or (3)
        Dipole of first dipole w/o vibronic progression
        :param rotation_matrix: np.ndarray(3,3)
        Rotation matrix necessary from initial parameters to this map
        :param description: str
        :param dipole_params: dt.DIPOLE_PARAMS
        '''
        self.labels = labels
        self.intensity_stack = intensity_stack
        self.x_array = x_array
        self.y_array = y_array
        self.mu_1 = mu_1
        self.rotation_matrix = rotation_matrix
        self.description = description #possible string that describes saved data
        self.dipole_params = dipole_params
    def get_meshes(self):
        return np.meshgrid(self.x_array,self.y_array)
    def get_stack_slice(self,index):
        if isinstance(index,str):#converting string to matching integer index
            index = index.lower()
            labels = map(lambda x: x.lower(),self.labels)
            index_list = [i for i, x in enumerate(labels) if x == index]
            index = int(index_list[0])
        return self.intensity_stack[:,:,index]
    def get_stack_tuple(self):
        stack_reshaped = np.moveaxis(self.intensity_stack,-1,0)
        return tuple(stack_reshaped)

def slice_max_abs_val(array,axis =-1):
    '''
    Gets an array of 1 dimension smaller than initial array with maximum absolute values along
    the truncated dimension
    :param array: np.ndarray
    :param axis: int
    axis to truncate
    :return:
    '''
    array_shape = np.shape(array)
    array_shape_list = list(array_shape)
    sliced_array_shape = tuple(array_shape_list)[:2]
    argmax_array = np.argmax(np.abs(array),axis=axis).flatten() #converting to 1D for iteration over unknown number of dimensions
    max_abs_array = np.zeros(sliced_array_shape).flatten()
    array_sliced_axis_last = np.swapaxes(array,axis,-1)
    array_sliced_axis_last = array_sliced_axis_last.reshape((np.size(max_abs_array),np.size(array_sliced_axis_last,-1)))
    for i in range(np.size(max_abs_array)):
        max_abs_array[i] = array_sliced_axis_last[i,argmax_array[i]]
    return max_abs_array.reshape(sliced_array_shape)
