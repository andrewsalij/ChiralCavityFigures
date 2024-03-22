import numpy as np
from scipy.special import factorial
from scipy.signal import find_peaks
import scipy.spatial as scp_spa
import python_util as pu
from scipy.spatial.transform import Rotation
from warnings import warn

'''
Handles the dielectric function in 3D for sets of Lorentz oscillators.
Also handles the core of LDLB theory (https://doi.org/10.1021/jacs.1c06752)


Incorporates some details from Laurence Barron's Molecular Theory of Solids for 
general response functions for magneto-electric couplings, magnetic couplings, and
stress/strain/dielectric relationships from electrostriction theory. 
Note that Barron's definitions of dichroism and birefrigence are half mine, so they are
altered by a factor of 2 in here 
 
Written by Andrew Salij
'''

class DIPOLE_SET_PARAMS():
    '''Highest level function for storing all dipole-related parameters'''
    def __init__(self,dielectric_params,energies,dip_mags,angle_array,huang_rhys,e_vib,
                 dipole_matrix = None):
        self.dielectric_params = dielectric_params
        self.energies = energies
        self.dip_mags=  dip_mags
        self.angle_array = angle_array
        self.huang_rhys = huang_rhys
        self.e_vib = e_vib
        if (dipole_matrix is None and self.dip_mags is not None and self.angle_array is not None):
            self.dipole_matrix = create_dipole_matrix_polar_3D(self.dip_mags,self.angle_array)
        else: self.dipole_matrix  = dipole_matrix

class DIELECTRIC_PARAMS():
    '''
    For encapsulation of parameters specific to dielectric of interest
    Mostly for hiding and shuttling data but also has some functionality
    with regards to the many (oh so many) ways that damping can be handled
    in a Lorentz oscillator model.
    '''
    def __init__(self, iso_hf_dielectric_const,volume_cell,damping_factor,length,gamma_type = "linear"):
        self.epsilon_inf = iso_hf_dielectric_const
        self.v = volume_cell
        self.gamma = damping_factor
        self.length = length
        self.gamma_type = gamma_type
    #gives damping function as array values at specific energy indices
    def damping_array(self,energy_array = np.ones(1)):
        arr_size = np.size(energy_array)
        if (self.gamma_type == "constant"):
            return np.ones(arr_size)*self.gamma
        elif (self.gamma_type == "linear"):
            return energy_array*self.gamma
        else:
            return np.array([self.gamma])
    #gives damping function for entire spectrum according to set rules
    def damping_spec(self,spectrum_array,special_dispersion_function = None):
        if (self.gamma_type == "arbitrary"):
            return self.gamma
        elif (self.gamma_type == "constant"):
            return np.ones(np.size(spectrum_array))*self.gamma
        elif (self.gamma_type == "linear"):
            return spectrum_array*self.gamma
        elif (special_dispersion_function):
            return special_dispersion_function(spectrum_array,self.gamma)
        #treats as constant as default
        else:
            return np.ones(np.size(spectrum_array))*self.gamma

class LINEAR_OPTICS():
    '''
    Data transfer class for dichroism and biregringence in xy and x'y' basis
    '''
    def __init__(self,linear_dichroism,linear_dichroism_prime, linear_birefringence,linear_birefringence_prime,absorbance = np.zeros(0)):
        self.ld = linear_dichroism
        self.ldp = linear_dichroism_prime
        self.lb = linear_birefringence
        self.lbp = linear_birefringence_prime
        self.absorbance = absorbance

    def ldlb(self):
        '''Returns LDLB :1/2(LD'*LB-LD*LB')'''
        ldlb = 0.5 * (self.ld * self.lbp - self.ldp * self.lb)
        return ldlb
    def select_by_index(self,idx):
        return LINEAR_OPTICS(self.ld[idx],self.ldp[idx],self.lb[idx],self.lbp[idx],self.absorbance[idx])

class UNIT_DEFINITIONS():
    '''Class to have consistent unit definitions between c, \hbar, \eps_0,mu_0'''
    def __init__(self,c,hbar,epsilon_0):
        self.c = c
        self.hbar = hbar
        self.e0 = epsilon_0
        self.eps0 = self.e0 #alternate name
        self.mu0 = 1/(self.c**2*self.e0)
unit_defs_base = UNIT_DEFINITIONS(1,1,1/(4*np.pi*0.007297))
#from e**2/(2*fine_structure*2pi*hbar*c) where e=1,hbar=1,c=1

def create_lineshape(a,b,gamma):
    '''
    Lineshape of a Lorentz oscillator with gamma damping.
    b is angular frequency, and a is oscillator angular frequency
    '''
    warn("create_lineshape() is deprecated. Use create_molecular_property_lineshape() instead")
    return b*create_molecular_property_lineshape(a,b,gamma)

def create_molecular_property_lineshape(spec,omega_n,gamma_n):
    '''
    :param spec:
    :param omega_n:
    :param gamma_n:
    :return:
    '''
    lineshape = 1/(omega_n**2-spec**2-1j*gamma_n*spec)
    return lineshape

def create_dipole_matrix_polar_2D(magnitude_array,theta_array):
    '''
    Produces 2D np.ndarray corresponding to a set of dipoles in the xy-plane
    :param magnitude_array: np.ndarray
    :param theta_array: np.ndarray
    :return: np.ndarray
    '''
    x_values = magnitude_array*np.cos(theta_array)
    y_values = magnitude_array*np.sin(theta_array)
    return np.vstack((x_values,y_values)).T

def create_dipole_matrix_polar_3D(magnitude_array,theta_array):
    '''
    Produces 3D np.ndarray corresponding to a set of dipoles in the xy-plane
    :param magnitude_array: np.ndarray
    :param theta_array: np.ndarray
    :return: np.ndarray
    '''
    x_values = magnitude_array*np.cos(theta_array)
    y_values = magnitude_array*np.sin(theta_array)
    z_values = np.zeros(np.size(magnitude_array))
    return np.vstack((x_values,y_values,z_values)).T

def dipole_matrix_to_params(matrix):
    '''
    Inverse transform to create_dipole_matrix_2D() or create_dipole_matrix_3D()
    :param matrix: np.ndarray (axis 1 must be x,y,...)
    :return: (np.ndarray,np.ndarray)
    '''
    dip_mags = np.sqrt(np.sum(matrix**2,axis = 1))
    angles = np.arctan(matrix[:,1]/matrix[:,0])
    angles = np.nan_to_num(angles)
    return dip_mags,angles

def center_angle_array(angle_array,index = 0,angle = 0.0):
    '''
    Takes an array of angles and shifts all elements so that the chosen index
    is set to the chosen angle. Does not perform modulo operations, which must
    be handled separately
    :param angle_array: np.ndarray
    :param index: np.int (default 0)
    :param angle: np.float (default 0.0)
    :return:
    '''
    offset = angle_array[index]-angle
    new_array = angle_array-offset
    return new_array

def get_molecular_property_tensor(excitonic_energy_array,energy_spectrum,dipole_matrix,gamma_array,unit_defs= unit_defs_base,
                                  tensor_type = "elec",magnetic_dipole_matrix = None):
    '''
    For a given set of Loretnz oscillators, provides \chi, the electronic suspeptibility
    :param dielectric_params: DIELECTRIC_PARAMS()
    :param excitonic_energy_array: np.ndarray (1D)
    :param energy_spectrum: np.ndarray (1D)
    :param dipole_matrix: np.ndarray (2D)
    :param gamma_array: np.ndarray (1D)
    :param unit_defs: UNIT_DEFS()
    :return: np.ndarray
    '''
    num_omeg=  np.size(energy_spectrum)
    prefactor = 2 /(unit_defs.hbar)
    tot_peak_shape = np.zeros((np.size(excitonic_energy_array), num_omeg), dtype=np.cdouble)
    tot_peak_shape_prime = np.zeros((np.size(excitonic_energy_array), num_omeg), dtype=np.cdouble)
    for n in range(0, np.size(excitonic_energy_array)):
        current_energy = pu.array_and_scalar_selector(excitonic_energy_array, n)
        current_gamma = pu.array_and_scalar_selector(gamma_array, n)
        peak_shape = create_molecular_property_lineshape(
            energy_spectrum, current_energy, current_gamma)
        tot_peak_shape[n, :] = peak_shape*current_energy
        tot_peak_shape_prime[n, :] = peak_shape*energy_spectrum
    if (tensor_type=="elec"):
        trans_cross = np.einsum("ki,kj->ijk", np.conjugate(dipole_matrix), dipole_matrix)
    elif(tensor_type =="mag_elec" and magnetic_dipole_matrix is not None):
        trans_cross = np.einsum("ki,kj->ijk",np.conjugate(dipole_matrix), magnetic_dipole_matrix)
    elif(tensor_type == "mag" and magnetic_dipole_matrix is not None):
        trans_cross = np.einsum("ki,kj->ijk", np.conjugate(magnetic_dipole_matrix), magnetic_dipole_matrix)
    tensor_contribiutions = np.einsum('ijk,kl->ijl', np.real(trans_cross), tot_peak_shape)+ \
                    np.einsum('ijk,kl->ijl', 1j*np.imag(trans_cross), tot_peak_shape_prime)
    return prefactor*tensor_contribiutions



def create_response_tensor(dielectric_params,dipole_matrix,excitonic_energy_array,energy_spectrum,tensor_type = "elec",unit_defs = unit_defs_base,magnetic_dipole_matrix = None,
                           **kwargs):
    '''
    Provides response tensor for electric, magneto-electro, or magnetic coupling
    Units are dimensionless, scaled by permittivities and permeabilities of free space for relevant quantities
    :param dielectric_params: DIELECTRIC_PARAMS()
    :param dipole_matrix: np.ndarray
    :param excitonic_energy_array: np.ndarray
    :param energy_spectrum: np.ndarray
    :param tensor_type: str
    :param unit_defs: UNIT_DEFS()
    :param magnetic_dipole_matrix: np.ndarray
    :param kwargs: dict
    :return: np.ndarray
    '''
    dim = np.size(dipole_matrix,axis = -1)
    if (dipole_matrix.ndim ==1):
        dipole_matrix = np.array([dipole_matrix])
    discrete_damping = True
    for key,value in kwargs.items():
        if key == "dimension":
            dim = value
        if key == "continuous_damping":
            discrete_damping = ~value
    excitonic_energy_array = pu.remove_unnecessary_indices(excitonic_energy_array)
    num_omeg = np.size(energy_spectrum)
    if (np.size(dielectric_params.gamma) == 1):
        if (discrete_damping):
            gamma_array = dielectric_params.damping_array(excitonic_energy_array)
        else:
            gamma_array = dielectric_params.damping_spectrum(energy_spectrum)
    else:
        gamma_array = dielectric_params.gamma
    #ensuring that data types are arrays
    if (tensor_type=="elec"):
        diag_tensor = np.dstack([dielectric_params.epsilon_inf*np.diag(np.ones(dim,dtype=np.csingle))]*num_omeg)
        scaling_factor = 1 / (unit_defs.e0 * dielectric_params.v)
    elif (tensor_type=="mag"):
        diag_tensor = np.dstack([np.diag(np.ones(dim,dtype=np.csingle))]*num_omeg)
        scaling_factor = unit_defs.mu0/dielectric_params.v
    elif (tensor_type=="mag_elec"):
        diag_tensor = 0
        scaling_factor = 1/(unit_defs.e0*unit_defs.c*dielectric_params.v)
    else:
        diag_tensor = 0
    polarizability = get_molecular_property_tensor(excitonic_energy_array,energy_spectrum,dipole_matrix,gamma_array,
                                                   tensor_type=tensor_type,magnetic_dipole_matrix=magnetic_dipole_matrix)

    response_function = diag_tensor+polarizability*scaling_factor
    return response_function

def create_dielectric_tensor(dielectric_params,dipole_matrix,excitonic_energy_array,energy_spectrum,unit_defs = unit_defs_base,**kwargs):
    return create_response_tensor(dielectric_params,dipole_matrix,excitonic_energy_array,energy_spectrum,tensor_type = "elec",unit_defs = unit_defs,**kwargs)

def create_all_response_tensors(dielectric_params,dipole_matrix,magnetic_dipole_matrix,excitonic_energy_array,energy_spectrum,unit_defs = unit_defs_base,**kwargs):
    '''
    :param dielectric_params: 
    :param dipole_matrix: 
    :param magnetic_dipole_matrix: 
    :param excitonic_energy_array: 
    :param energy_spectrum: 
    :param unit_defs: 
    :param kwargs: 
    :return: 
    '''
    dielectric_tensor = create_response_tensor(dielectric_params,dipole_matrix,excitonic_energy_array,energy_spectrum,
                                               tensor_type = "elec",unit_defs = unit_defs,**kwargs)
    g_tensor = create_response_tensor(dielectric_params,dipole_matrix,excitonic_energy_array,energy_spectrum,
                                      tensor_type = "mag_elec",unit_defs = unit_defs,magnetic_dipole_matrix = magnetic_dipole_matrix,**kwargs)
    g_tensor_transpose = np.swapaxes(g_tensor,0,1)
    mag_elec_tensor = 0.5*(g_tensor_transpose+g_tensor)
    elec_mag_tensor = -1*np.swapaxes(mag_elec_tensor,0,1)    #assuming reciprocity
    mag_tensor = create_response_tensor(dielectric_params,dipole_matrix,excitonic_energy_array,energy_spectrum,
                                               tensor_type = "mag",unit_defs = unit_defs,magnetic_dipole_matrix=magnetic_dipole_matrix,**kwargs)
    return dielectric_tensor,mag_elec_tensor,elec_mag_tensor,mag_tensor

def create_refractive_index_tensor_pert(dielectric_params,dipole_matrix,excitonic_energy_array,energy_spectrum):
    '''

    :param dielectric_params:
    :param dipole_matrix:
    :param excitonic_energy_array:
    :param energy_spectrum:
    :return:
    '''
    excitonic_energy_array = pu.remove_unnecessary_indices(excitonic_energy_array)
    gamma_array = dielectric_params.damping_array(excitonic_energy_array)
    n_tensor = np.dstack([np.sqrt(dielectric_params.epsilon_inf) * np.diag(np.ones(3, dtype=np.csingle))] * np.size(energy_spectrum))
    susceptibility =1/(unit_defs_base.e0*dielectric_params.v)*\
                    get_molecular_property_tensor(excitonic_energy_array, energy_spectrum, dipole_matrix,
                                        gamma_array)
    n_tensor = n_tensor+susceptibility/(2*np.sqrt(dielectric_params.epsilon_inf))
    return n_tensor

def get_linear_optics_pert(dielectric_params,dipole_matrix,excitonic_energy_array,energy_spectrum):
    if (np.size(dipole_matrix,axis= 1)==2):
        dipole_matrix = np.hstack([dipole_matrix,np.array([np.zeros(np.size(dipole_matrix,axis=1))]).T])
    length_over_c = dielectric_params.length/unit_defs_base.c
    rot_array = np.array([0,0,-np.pi/4]) #dipoles rotated to primed axis x'y'
    dipole_matrix_rotated = rotate_vector(rot_array, dipole_matrix, transpose=True)
    n_tensor_pert = create_refractive_index_tensor_pert(dielectric_params,dipole_matrix,excitonic_energy_array,energy_spectrum)
    n_tensor_pert_prime = create_refractive_index_tensor_pert(dielectric_params,dipole_matrix_rotated,excitonic_energy_array,energy_spectrum)
    linear_dichroism, linear_dichroism_prime, linear_birefringence, linear_birefringence_prime = get_ldlb_params_from_n_tensor(n_tensor_pert,n_tensor_pert_prime,energy_spectrum,length_over_c)
    absorbance_v1 = (n_tensor_pert_prime[1, 1, :].imag + n_tensor_pert_prime[0, 0, :].imag)*energy_spectrum*length_over_c
    absorbance_v2 = (n_tensor_pert[1, 1, :].imag + n_tensor_pert[0, 0, :].imag)*energy_spectrum*length_over_c
    absorbance_net = np.maximum(absorbance_v1,absorbance_v2)
    return LINEAR_OPTICS(linear_dichroism,linear_dichroism_prime,linear_birefringence,linear_birefringence_prime,absorbance_net)

def euler_rotation_matrix(r_x,r_y,r_z): #note that the transformation is z, then y, then x
    '''Deprecated. Use quaternions or axis angles instead '''
    r_matrix_x = np.array([[1,0,0],[0,np.cos(r_x),-np.sin(r_x)],[0,np.sin(r_x),np.cos(r_x)]])
    r_matrix_y = np.array([[np.cos(r_y),0,np.sin(r_y)],[0,1,0],[-np.sin(r_y),0,np.cos(r_y)]])
    r_matrix_z = np.array([[np.cos(r_z),-np.sin(r_z),0],[np.sin(r_z),np.cos(r_z),0],[0,0,1]])
    rotation_matrix = np.dot(r_matrix_x,np.dot(r_matrix_y,r_matrix_z))
    return rotation_matrix

def axis_angle_to_quaternion(axis_angle_array):
    '''
    :param axis_angle_array: ([{x,y,z},theta])
    :return: np.ndarray
    '''
    theta = axis_angle_array[3]
    unit_vec = axis_angle_array[:3]/np.linalg.norm(axis_angle_array[:3])
    quat_xyz=  np.sin(theta/2)*unit_vec
    quat_w = np.cos(theta/2)
    return np.append(quat_xyz,quat_w)

def multiply_quaternion(quat_b,quat_a):
    '''
    Multiplies two quaternions of the form (x,y,z,w) by definition
    Rotates by quaternion b, then quaternion a
    :param quat_b: np.ndarray or tuple
    :param quat_a: np.ndarray or tuple
    :return: np.ndarray
    '''
    x_a,y_a,z_a,w_a = quat_a
    x_b,y_b,z_b,w_b = quat_b
    return np.array([w_a*x_b+x_a*w_b-y_a*z_b-z_a*y_b,w_a*y_b-x_a*z_b+y_a*w_a+z_a*x_b,w_a*z_b+x_a*y_b-y_a*x_b*z_a*w_b,
                     w_a*w_b-x_a*x_b-y_a*y_b-z_a*z_b])


def quaternion_rotation_matrix(quat):
    '''
    Provides 3D rotation matrix corresponding to given quaternion
    :param quat: np.ndarray (1D) 
    :return: np.narray (2D)
    '''
    rotation = Rotation.from_quat(quat)
    return rotation.as_matrix()

def rotate_2D_tensor(angle_rotation_array,initial_tensor,rot_type= "euler"):
    '''
    Rotates a 2D tensor.
    Convention chosen such that angle rotation array is that of counter-clockwise rotation of object,
    or a clockwise rotation in the case of changing coordinate system.
    Hence there is essentially a double transpose going on so that the rotation matrix will
    be of the familiar [[cos(theta),-sin(theta)],[sin(theta),cos(theta)]] form
    rotation along z axis of -45 deg. (-pi/4 radians) goes from xy to x'y'
    :param angle_rotation_array: 
    :param initial_tensor: np.ndarray (2D or 3D)
    :param rot_type: 
    :return: 
    '''
    if (rot_type =='euler'):
        r_x, r_y, r_z = angle_rotation_array[0],angle_rotation_array[1],angle_rotation_array[2]
        rotation_matrix = euler_rotation_matrix(r_x, r_y, r_z)
    elif (rot_type =='quat'):
        rotation_matrix = quaternion_rotation_matrix(angle_rotation_array)
    elif (rot_type == 'axis_angle'):
        angle_rotation_quat= axis_angle_to_quaternion(angle_rotation_array)
        rotation_matrix = quaternion_rotation_matrix(angle_rotation_quat)
    else:
        raise ValueError("Rotation Type Invalid")
    if (initial_tensor.ndim == 2):
        rotated_matrix = np.einsum("ij,jk->ik",\
                                   rotation_matrix,np.einsum("ij,jk->ik",initial_tensor,np.transpose(rotation_matrix)))
    elif (initial_tensor.ndim == 3):
        rotated_matrix = np.einsum("ij,jkl->ikl",\
                                   rotation_matrix,np.einsum("ijl,jk->ikl",initial_tensor,np.transpose(rotation_matrix)))
    else:raise ValueError("Invalid tensor size")
    return rotated_matrix


def rotate_vector(angle_rotation_array,initial_vector,transpose = False):
    '''
    Rotates a vector or set of vectors
    :param angle_rotation_array: np.ndarray (1D, shape = 3)
    :param initial_vector: np.ndarray (1D or 2D)
    :param transpose: bool , whether to transpose the initial_vector array or not 
    :return: np.ndarray (shape = np.shape(inital_vector))
    '''
    r_x, r_y, r_z = angle_rotation_array[0], angle_rotation_array[1], angle_rotation_array[2]
    rotation_matrix = euler_rotation_matrix(r_x, r_y, r_z)
    if (initial_vector.ndim == 1):
        rotated_vector = np.einsum("ij,j->i",rotation_matrix,initial_vector)
    if (initial_vector.ndim == 2):
        if (transpose):
            initial_vector = initial_vector.T
        rotated_vector = np.einsum("ij,jl->il", rotation_matrix, initial_vector)
        if (transpose):
            rotated_vector = rotated_vector.T
    return rotated_vector

def order_2D_basis(orthogonal_2D_basis):
    '''
    Provides an ordered definition of two vectors x,y for an arbitrary orthogonal 2D basis
    :param orthogonal_2D_basis: np.ndarray 
    :return:(np.ndarray,np.ndarray) 
    '''
    vec_1 = orthogonal_2D_basis[:,0]
    vec_2 = orthogonal_2D_basis[:,1]
    error = 0.01
    rotation_matrix = np.array([[0,-1],[1,0]])
    rotated_vec_1 = np.einsum("ij,j->i",rotation_matrix,vec_1)
    if  (np.less_equal(vec_2-error,rotated_vec_1).all() & np.greater_equal(vec_2+error,rotated_vec_1).all()):
        x = vec_1
        y = vec_2
    else:
        x = vec_2
        y = vec_1
    return x,y

def order_3D_basis(orthonormal_basis):
    '''
    Provides an ordered definition of three vectors x,y,z for an arbitrary orthogonal 3D basis
    :param orthogonal_3D_basis: np.ndarray
    :return:(np.ndarray,np.ndarray,np.ndarray)
    '''
    cross_product_matrix = np.zeros((3,3,3),dtype=np.csingle)
    error = 0.01
    ordered_sets = np.zeros((3,2))
    ordered_basis=  np.zeros((3,3),dtype = np.csingle)
    for i in range(0,3):
        for j in range(0,3):
            cross_product_matrix[i,j,:] = np.cross(orthonormal_basis[:,i],orthonormal_basis[:,j])
    for i in range(0,3):
        for j in range(0, 3):
            for k in range(0,3):
                if (np.less_equal(orthonormal_basis[:,k]-error,cross_product_matrix[i,j,:]).all() & np.greater_equal(orthonormal_basis[:,k]+error,cross_product_matrix[i,j,:]).all()):
                    ordered_sets[i,:] = np.array([i,j])
    x_index = np.array(np.where(orthonormal_basis[0,:] == np.max(orthonormal_basis[0,:])))[0]
    y_index = np.int(ordered_sets[x_index,1])
    z_index = np.int(ordered_sets[y_index,1])
    indices = np.array([x_index,y_index,z_index]).astype(int)
    for i in range(0,3):
        ordered_basis[:,i] = orthonormal_basis[:,np.int(indices[i])]
    return indices, ordered_basis

def solve_tensor(tensor):
    '''
    Decomposes a tensor or tensor stack into eigenvalues and eigenvectors and returns
    a diagonalized tensor or tensor stack with eigenvectors
    :param tensor: np.ndarray (2D or 3D)
    :return: (np.ndarray (2D), np.ndarray (shape = np.shape(tensor))
    '''
    new_tensor = np.zeros(np.shape(tensor),dtype = np.csingle)
    axes_dim = np.size(tensor,axis= 0)
    vecs_tot = np.zeros(np.shape(tensor),dtype = np.csingle)
    if (axes_dim== 3):
        for i in range(np.size(tensor,2)):
            vals, vecs = np.linalg.eigh(tensor[:,:,i])
            vals_real, vecs_real = np.linalg.eigh(np.real(tensor[:, :, i]))
            id, vecs = order_3D_basis(vecs_real)
            vals = vals[id]
            new_tensor[:,:,i] = np.diag(vals)
            vecs_tot[:,:,i] = vecs
    if (axes_dim ==2):
        for i in range(np.size(tensor,2)):
            vals, vecs = np.linalg.eigh(tensor[:, :, i])
            new_tensor[:,:,i] = np.diag(vals)
            vecs_tot[:,:,i] = vecs
    return vecs_tot, new_tensor


def get_refractive_index(symmetric_dielectric_tensor):
    '''
    Provides refractive indices from a dielectric tensor
    Somewhat deprecated, use get_refractive_index_tensor() instead
    :param symmetric_dielectric_tensor: np.ndarray (3D)
    :return: (np.ndarray (1D) ,np.ndarray (1D),np.ndarray (1D))
    '''
    n_x = np.sqrt(symmetric_dielectric_tensor[0,0,:])
    n_y = np.sqrt(symmetric_dielectric_tensor[1,1,:])
    n_z = np.sqrt(symmetric_dielectric_tensor[2,2,:])
    return n_x, n_y, n_z

def get_refractive_index_2d(dielectric_tensor_2d):
    '''
    Identical to get_refractive_index() but only returns n_x,n_y
    :param dielectric_tensor_2d: np.ndarray (3D)
    :return: (np.ndarray (1D)),np.ndarray (1D))
    '''
    n_x = np.sqrt(dielectric_tensor_2d[0,0,:])
    n_y = np.sqrt(dielectric_tensor_2d[1,1,:])
    return n_x,n_y

def get_refractive_index_tensor(dielectric_tensor):
    '''
    Provides complex refractive index tensor from dielectric tensor
    :param dielectric_tensor: np.ndarray
    :return: np.ndarray
    '''
    n_complex = np.sqrt(dielectric_tensor)
    return n_complex



def lorentzian(spectrum,freq,damping_factor):
    '''Provides Lorentzian lineshape'''
    return spectrum/(freq**2-spectrum**2-1j*damping_factor*spectrum)


def eigen_portion(matrix,axis_set):
    '''Extracts submatrix from matrix'''
    rows, columns = axis_set, axis_set
    submatrix = matrix[rows,columns]
    return submatrix

#should be faster than the other implementation-assumes matrix is xyz
def eigen_portion_xy(matrix):
    "Extracts eigenvalues and rotated submatrix from matrix"
    submatrix = matrix[:2,:2]
    vals, vecs = np.linalg.eig(submatrix)
    total_rotation_matrix = np.zeros((3,3))
    total_rotation_matrix[:2,:2] = vecs
    total_rotation_matrix[2,2] = 1
    return vals, total_rotation_matrix

def diagonalize_matrix_stack(matrix_stack,dtype = np.float64,axis_set = None,to_reorder = True):
    '''
    Diagonalizes and returns a stack of matrices
    :param matrix_stack: np.ndarray (3D)
    :param dtype: dtype of matrix to return
    :param axis_set: np.ndarray
    :param to_reorder: bool
    :return: (np.ndarray (3D),np.ndarray (3D))
    '''
    diag_matrix_stack = np.zeros(np.shape(matrix_stack),dtype = dtype)
    rotation_matrix_stack = np.zeros(np.shape(matrix_stack),dtype = dtype)
    for i in range(0,np.size(matrix_stack,axis =2)):
        if (axis_set is None):
            vals, vecs= np.linalg.eig(matrix_stack[:,:,i])
        else:
            vals, vecs= eigen_portion_xy(matrix_stack[:,:,i])
            vals = np.hstack((vals,matrix_stack[2,2,i]))
        if (to_reorder):
            idx = vals.argsort()[::-1]
            vals = vals[idx]
            vecs = vecs[:, idx]
        diag_matrix_stack[:, :, i] = np.diag(vals)
        rotation_matrix_stack[:, :, i] = vecs
    return diag_matrix_stack, rotation_matrix_stack

#NOTE THAT the rotation element [1,0] = sin(\theta)--this is for e' = R^T e R, where
#e' is the diag matrix, then R e' R^t = e (e in Cartesian basis)
def rotation_matrix_2D_to_angles(rotation_matrix):
    '''
    Converts rotation matrix to angle array
    :param rotation_matrix: np.ndarray (2D or 3D)
    :return: np.ndarray (1D)
    '''
    dimensions = rotation_matrix.ndim
    if (dimensions ==2):
        return np.arctan2(rotation_matrix[1,0],rotation_matrix[0,0])
    elif(dimensions == 3):
        return np.arctan2(rotation_matrix[1,0,:],rotation_matrix[0,0,:])
    else:
        raise ValueError("Invalid rotation_matrix dimension")

def rotation_matrix_3D_to_angles(rotation_matrix):
    '''
    Converts rotation matrix to angle array in terms of euler angles (xyz ordering)
    :param rotation_matrix: np.ndarray (2D or 3D)
    :return: np.ndarray (1D)
    '''
    rotation_matrix_reordered = np.moveaxis(rotation_matrix,[0,1],[1,2])
    return scp_spa.transform.Rotation.from_matrix(rotation_matrix_reordered).as_euler("xyz")

def diagonal_n_tensor(dielectric_params,transition_dipole_matrix,transition_energies,spectrum,unit_defs = UNIT_DEFINITIONS(1,1,1/(4*np.pi*0.007297))):
    '''
    Creates diagonalized n_tensor (real and complex) as well as rotations necessary to diagonalize it
    :param dielectric_params: DIELECTRIC_PARAMS()
    :param transition_dipole_matrix: np.ndarray (2D)
    :param transition_energies: np.ndarray (1D)
    :param spectrum: np.ndarray (1D)
    :param unit_defs: UNIT_DEFS()
    :return: (np.ndarray,np.ndarray,np.ndarray,np.ndarray)
    '''
    dielectric_tensor= create_dielectric_tensor(dielectric_params, transition_dipole_matrix, transition_energies,
                                                      spectrum, unit_defs,
                                                      **{"dimension": 2})
    n_tensor = get_refractive_index_tensor(dielectric_tensor)
    n_real = n_tensor.real
    n_imag=  n_tensor.imag
    diag_n, rot_n = diagonalize_matrix_stack(n_real)
    diag_k, rot_k = diagonalize_matrix_stack(n_imag)
    return diag_n, rot_n, diag_k, rot_k

def diagonal_dielectric_tensor(dielectric_params,transition_dipole_matrix,transition_energies,spectrum,unit_defs = UNIT_DEFINITIONS(1,1,1/(0.007297*4*np.pi))):
    '''
    Creates diagonalized dielectric tensor (real and complex) as well as rotations necessary to diagonalize it
    :param dielectric_params: DIELECTRIC_PARAMS()
    :param transition_dipole_matrix: np.ndarray (2D)
    :param transition_energies: np.ndarray (1D)
    :param spectrum: np.ndarray (1D)
    :param unit_defs: UNIT_DEFS()
    :return: (np.ndarray,np.ndarray,np.ndarray,np.ndarray)
    '''
    dielectric_tensor= create_dielectric_tensor(dielectric_params, transition_dipole_matrix, transition_energies,
                                                      spectrum, unit_defs,
                                                      **{"dimension": 2})
    dielectric_real = dielectric_tensor.real
    dielectric_imag =  dielectric_tensor.imag
    diag_real, rot_real = diagonalize_matrix_stack(dielectric_real)
    diag_imag, rot_imag = diagonalize_matrix_stack(dielectric_imag)
    return diag_real,rot_real,diag_imag,rot_imag

def get_ldlb_2D(dielectric_params,transition_dipole_matrix,transition_energies,spectrum,unit_defs = unit_defs_base,rotation_array = None,**kwargs):
    '''
    Produces full linear and ldlb characterization of a Lorentzian dipole system in xy-plane
    :param dielectric_params: DIELECTRIC_PARAMS()
    :param transition_dipole_matrix: np.ndarray (2D)
    :param transition_energies: np.ndarray (1D)
    :param spectrum: np.ndarray (1D)
    :param unit_defs: UNIT_DEFS()
    :param rotation_array: np.ndarray(1D)
    :param kwargs: dict
    :return: ld_net, ldp_net, lb_net, lbp_net, ldlb_net,n_x,n_y (all np.ndarray)
    '''
    length = dielectric_params.length
    fixed_angle = False
    if (rotation_array is not None):
        rot_size=  np.size(rotation_array)
    else:
        rot_size = 100
    for key,value in kwargs.items():
        if key == "fixed_angle":
            rot_size = 1
            fixed_angle = True
            angle_to_set = value
    dielectric_tensor_init = create_dielectric_tensor(dielectric_params, transition_dipole_matrix, transition_energies,
                                                      spectrum, unit_defs,
                                                      **{"dimension": 2})
    #axes, dielectric_tensor = solve_tensor(dielectric_tensor_init)
    dielectric_tensor = dielectric_tensor_init

    ld_net = np.zeros((np.size(spectrum),rot_size),dtype = np.csingle)
    ldp_net =np.zeros((np.size(spectrum),rot_size),dtype = np.csingle)
    lb_net = np.zeros((np.size(spectrum),rot_size),dtype = np.csingle)
    lbp_net =np.zeros((np.size(spectrum),rot_size),dtype = np.csingle)
    ldlb_net = np.zeros((np.size(spectrum),rot_size),dtype = np.csingle)
    n_x = np.zeros((np.size(spectrum), rot_size), dtype=np.csingle)
    n_y = np.zeros((np.size(spectrum),rot_size),dtype = np.csingle)
    for i in range(0,rot_size):
        if (fixed_angle):
            theta = angle_to_set
        elif (rotation_array is not None):
            theta = rotation_array[i]
        else:
            theta = 2 * np.pi * i / 100
        init_rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        dielectric_tensor = np.einsum("ij,jkl->ikl",\
                                       init_rotation_matrix,np.einsum("ijl,jk->ikl",dielectric_tensor_init,np.transpose(init_rotation_matrix)))
        n_tensor = get_refractive_index_tensor(dielectric_tensor)
        linear_dichroism, linear_dichroism_prime, linear_birefringence, linear_birefringence_prime = ld_params_from_dielectric(dielectric_tensor,spectrum,length)
        ldlb = 0.5*(linear_dichroism*linear_birefringence_prime-linear_dichroism_prime*linear_birefringence)
        ld_net[:,i] = linear_dichroism
        ldp_net[:,i] = linear_dichroism_prime
        lb_net[:,i] = linear_birefringence
        lbp_net[:,i] = linear_birefringence_prime
        ldlb_net[:,i] = ldlb
        n_x[:,i] = n_tensor[0,0,:].real
        n_y[:,i] = n_tensor[1,1,:].real
    return ld_net, ldp_net, lb_net, lbp_net, ldlb_net,n_x,n_y

def ldlb_from_tuple(ldlb_tuple):
    '''Converts tuple of ldp,ld,lbp,lb (all np.ndarray) to ldlb np.ndarray'''
    a = ldlb_tuple
    return 0.5*(a[1]*a[2]-a[0]*a[3])

def ld_params_from_dielectric(dielectric_tensor,spectrum,length):
    '''
    Converts dielectric tensor over a spectrum to a tuple of linear dichroism and 
    linear birefringence spectra 
    :param dielectric_tensor: np.ndarray 
    :param spectrum: np.ndarray (1D)
    :param length: float or np.ndarray (size = np.size(spectrum))
    :return: (LD,LDP,LB,LBP)(all np.ndarray)
    '''
    prime_angle = -np.pi / 4
    dim = np.size(dielectric_tensor,axis = 0)
    if (dim == 2):
        rotation_matrix = np.array(
        [[np.cos(prime_angle), -np.sin(prime_angle)], [np.sin(prime_angle), np.cos(prime_angle)]])
    else:
        rotation_matrix = np.array(
            [[np.cos(prime_angle), -np.sin(prime_angle),0], [np.sin(prime_angle), np.cos(prime_angle),0],[0,0,1]])
    dt_prime = np.einsum("ij,jkl->ikl", \
                         rotation_matrix, np.einsum("ijl,jk->ikl", dielectric_tensor, np.transpose(rotation_matrix)))
    n_tensor = get_refractive_index_tensor(dielectric_tensor)
    n_prime = get_refractive_index_tensor(dt_prime)
    return get_ldlb_params_from_n_tensor(n_tensor,n_prime,spectrum,length)

def get_ldlb_params_from_n_tensor(n_tensor,n_prime,spectrum,length_over_c):
    '''
    Converts refractive index tensor over a spectrum to a tuple of linear dichroism and 
    linear birefringence spectra 
    :param n_tensor: np.ndarray 
    :param n_prime: np.ndarray (n_tensor rotated 45 deg. )
    :param spectrum: np.ndarray (1D)
    :param length: float or np.ndarray (size = np.size(spectrum))
    :return: (LD,LDP,LB,LBP)(all np.ndarray)
    '''
    linear_dichroism = -1 * (n_tensor[1, 1, :].imag - n_tensor[0, 0, :].imag) * spectrum * length_over_c
    linear_dichroism_prime = -1 * (n_prime[1, 1, :].imag - n_prime[0, 0, :].imag) * spectrum * length_over_c
    linear_birefringence = -1 * (n_tensor[1, 1, :].real - n_tensor[0, 0, :].real) * spectrum * length_over_c
    linear_birefringence_prime = -1 * (n_prime[1, 1, :].real - n_prime[0, 0, :].real) * spectrum * length_over_c
    return linear_dichroism, linear_dichroism_prime, linear_birefringence, linear_birefringence_prime

#the matrices are in energy along their 0 axis, rotation of measurement along axis 1
def get_ld_lb_axes(ld_matrix,lb_matrix,rotation_array):
    '''Extracts principal linear dichroism and linear birefringence axes'''
    ld_primary_axis =  rotation_array[np.argmax(ld_matrix,axis = 1)]
    lb_primary_axis = rotation_array[np.argmax(lb_matrix,axis = 1)]
    return ld_primary_axis,lb_primary_axis

class FILM_PARAMS():
    '''
    Container class for a thin film of Lorentzian oscillators 
    '''
    def __init__(self, dielectric_params, transition_dipole_matrix, transition_energy_array):
        self.dielectric_params = dielectric_params
        self.transition_dipole_matrix = transition_dipole_matrix
        self.transition_energies = transition_energy_array

def get_xy_rotation_matrix(axis_dimensions,angle):
    if (axis_dimensions== 2):
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    elif (axis_dimensions == 3):
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle),0], [np.sin(angle), np.cos(angle),0],[0,0,1]])
    return rotation_matrix


def linear_optics_from_dielectric_tensor(dielectric_tensor,spectrum,to_print = False, **kwargs):
    '''
    For a given dielectric tensor, returns all linear polarization spectra 
    :param dielectric_tensor: np.ndarray (2D or 3D)
    :param spectrum: np.ndarray (1D)
    :param to_print: bool
    :param kwargs: dict
    :return: LINEAR_OPTICS()
    '''
    length_over_c = 1
    for key, value in kwargs.items():
        if key == "length_over_c":
            length_over_c = value
    prime_angle = -np.pi/4 #such that the coordinates become x'y'
    #eps' = R(-45)eps R(45)
    axis_dimensions = np.size(dielectric_tensor,axis=0)
    rotation_matrix = get_xy_rotation_matrix(axis_dimensions,prime_angle)
    dt_prime = np.einsum("ij,jkl->ikl", \
                         rotation_matrix,
                         np.einsum("ijl,jk->ikl", dielectric_tensor, np.transpose(rotation_matrix)))
    n_tensor = get_refractive_index_tensor(dielectric_tensor)
    n_prime = get_refractive_index_tensor(dt_prime)
    linear_dichroism,linear_dichroism_prime,linear_birefringence,linear_birefringence_prime = get_ldlb_params_from_n_tensor(n_tensor,n_prime,spectrum,length_over_c)
    absorbance_v1 = (n_prime[1, 1, :].imag + n_prime[0, 0, :].imag)*spectrum*length_over_c
    absorbance_v2 = (n_tensor[1, 1, :].imag + n_tensor[0, 0, :].imag)*spectrum*length_over_c
    #Why do this? Basically, there are (very slight) numeric differences between each orientation, which
    #means that a resonance sometimes linear dichroism>absorbance if the lesser one is used, which is nonphysical and breaks
    #things in Mueller calculus at very long path lengths
    #this was not an issue in the first paper, but becomes later
    absorbance = np.maximum(absorbance_v1,absorbance_v2)
    optics_params = LINEAR_OPTICS(linear_dichroism,linear_dichroism_prime,linear_birefringence,linear_birefringence_prime,absorbance =absorbance )
    if (to_print == True):
        print("tensor" + str(n_tensor[:, :, 0]))
    return optics_params

def linear_optics_from_refractive_index_tensor(refractive_index_tensor,refractive_index_tensor_rotated,spectrum, **kwargs):
    '''
    For a given refractive index tensor, returns all linear polarization spectra
    :param refractive_index_tensor: np.ndarray (2D or 3D)
    :param spectrum: np.ndarray (1D)
    :param to_print: bool
    :param kwargs: dict
    :return: LINEAR_OPTICS()
    '''
    '''
    For a given dielectric tensor, returns all linear polarization spectra 
    :param dielectric_tensor: np.ndarray (2D or 3D)
    :param spectrum: np.ndarray (1D)
    :param to_print: bool
    :param kwargs: dict
    :return: LINEAR_OPTICS()
    '''
    length_over_c = 1
    for key, value in kwargs.items():
        if key == "length_over_c":
            length_over_c = value
    n_prime = refractive_index_tensor_rotated
    n_tensor = refractive_index_tensor
    linear_dichroism,linear_dichroism_prime,linear_birefringence,linear_birefringence_prime = get_ldlb_params_from_n_tensor(n_tensor,n_prime,spectrum,length_over_c)
    absorbance_v1 = (n_prime[1, 1, :].imag + n_prime[0, 0, :].imag)*spectrum*length_over_c
    absorbance_v2 = (n_tensor[1, 1, :].imag + n_tensor[0, 0, :].imag)*spectrum*length_over_c
    absorbance = np.maximum(absorbance_v1,absorbance_v2)
    optics_params = LINEAR_OPTICS(linear_dichroism,linear_dichroism_prime,linear_birefringence,linear_birefringence_prime,absorbance =absorbance )
    return optics_params

def absorbance_from_dielectric_tensor(dielectric_tensor,spectrum):
    '''Deprecated: use LINEAR_OPTICS() class instead'''
    n_tensor = get_refractive_index_tensor(dielectric_tensor)
    return (n_tensor[1, 1, :].imag + n_tensor[0, 0, :].imag) * spectrum


def get_ldlb_2D_two_films(film_a,film_b,spectrum,unit_defs = UNIT_DEFINITIONS(1,1,1/(4*np.pi*0.007297)),**kwargs):
    ''' For two ajacent films, provides linear optics for each film and LDLB for the
    entire system.
    :param film_a: THIN_FILM_PARAMS()
    :param film_b: THIN_FILM_PARAMS()
    :param spectrum: np.ndarray (1D)
    :param unit_defs: UNIT_DEFS()
    :param kwargs: dict
    :return: (LINEAR_OPTICS(),LINEAR_OPTICS(),np.ndarray (LDLB))
    '''
    rot_size = 100
    fixed_angle = False
    for key,value in kwargs.items():
        if key == "fixed_angle":
            rot_size = 1
            fixed_angle = True
            angle_to_set = value

    dielectric_tensor_init_a = create_dielectric_tensor(film_a.dielectric_params, film_a.transition_dipole_matrix, film_a.transition_energies,
                                                      spectrum, unit_defs,
                                                      **{"dimension": 2})
    dielectric_tensor_init_b = create_dielectric_tensor(film_b.dielectric_params, film_b.transition_dipole_matrix, film_b.transition_energies,
                                                      spectrum, unit_defs,
                                                      **{"dimension": 2})

    rotation_second_film_size = 100
    ldlb_net = np.zeros((np.size(spectrum),rot_size,rotation_second_film_size),dtype = np.csingle)
    optics_params_a_net = []
    optics_params_b_net = []
    for i in range(0,rot_size):
        if(fixed_angle):
            theta = angle_to_set
        else:
            theta = 2*np.pi*i/rot_size
        init_rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        dielectric_tensor_a = np.einsum("ij,jkl->ikl",\
                                       init_rotation_matrix,np.einsum("ijl,jk->ikl",dielectric_tensor_init_a,np.transpose(init_rotation_matrix)))
        optics_params_a = linear_optics_from_dielectric_tensor(dielectric_tensor_a,spectrum)
        optics_params_a_net.append(optics_params_a)
        for j in range(0,rotation_second_film_size):
            theta_prime = 2*np.pi*j/rotation_second_film_size
            rotation_matrix_second_film = np.array([[np.cos(theta+theta_prime), -np.sin(theta+theta_prime)], [np.sin(theta+theta_prime), np.cos(theta+theta_prime)]])
            dielectric_tensor_b = np.einsum("ij,jkl->ikl",\
                                      rotation_matrix_second_film,np.einsum("ijl,jk->ikl",dielectric_tensor_init_b,np.tranpose(rotation_matrix_second_film)))
            optics_params_b = linear_optics_from_dielectric_tensor(dielectric_tensor_b,spectrum)
            optics_params_b_net.append(optics_params_b)
            ldlb_signal = ldlb_two_film_from_params(optics_params_a,optics_params_b)
            ldlb_net[:,i,j] = ldlb_signal
    return optics_params_a_net, optics_params_b_net, ldlb_net

def ldlb_pert(dipole_mat,energy_array,spectrum,damping_factor,alpha):
    '''
    Gives ldlb lineshape, not quantitative value (works for two dipoles)
    :param dipole_mat: np.ndarray (2D)
    :param energy_array: np.ndarray (1D)
    :param spectrum:np.ndarray (1D)
    :param damping_factor: must be constant
    :param alpha: \theta_1 - \theta_2
    :return:
    '''
    i1 = np.sum(dipole_mat[0,:]**2)*energy_array[0]
    i2 = np.sum(dipole_mat[1,:]**2)*energy_array[1]
    w1 = energy_array[0]
    w2 = energy_array[1]
    alpha = np.array([alpha])
    if (alpha.ndim == 1):
        ldlb = np.sin(2*alpha)*i1*i2*damping_factor*spectrum*(w2**2-w1**2)\
            /(((w1**2-spectrum**2)**2+damping_factor**2*spectrum**2)*((w2**2-spectrum**2)**2+damping_factor**2*spectrum**2))
    elif(alpha.ndim == 2):
        ldlb_lack_angle = i1*i2*damping_factor*spectrum*(w2**2-w1**2)\
            /(((w1**2-spectrum**2)**2+damping_factor**2*spectrum**2)*((w2**2-spectrum**2)**2+damping_factor**2*spectrum**2))
        angle_contribs = np.sin(2*alpha)
        ldlb = np.outer(ldlb_lack_angle,angle_contribs)
    ldlb = ldlb*spectrum**2
    return ldlb

def pert_lineshape(w1,w2,spec,gamma):
    '''Perturbative LDLB lineshape for two energies'''
    num = (w2**2-w1**2)*spec**3
    denom = ((w1**2-spec**2)**2+gamma**2*spec**2)*((w2**2-spec**2)**2+gamma**2*spec**2)
    return num/denom

def func_ldlb(ld,ldp,lb,lbp):
    '''LDLB from linear optics spectra (all np.ndarray)'''
    ldlb = 0.5 * (ld* lbp - ldp * lb)
    return ldlb

def func_ldlb_two_films(optical_params_a,optical_params_b):
    return func_ldlb(optical_params_a.ld,optical_params_a.ldp,optical_params_b.lb,optical_params_b.lbp)

def ldlb_two_film_from_params(optical_params_a,optical_params_b):
    '''For two films LINEAR_OPTICS(), provides LDLB signal'''
    signal = 1/4*(2*func_ldlb_two_films(optical_params_a,optical_params_b)+func_ldlb_two_films(optical_params_a,optical_params_a)+func_ldlb_two_films(optical_params_b,optical_params_b))
    #the 1/4 is to account for the doubling of effective length due to the two film domain
    return signal

def ldlb_pert_factor(e_inf,unit_defs,volume):
    '''Perterbative LDLB proportionality factor'''
    init_factor = (2/(unit_defs.e0*unit_defs.hbar*volume))**2
    proportionality_factor = 1/2*init_factor*1/((2*np.sqrt(e_inf)))**2
    return proportionality_factor

def eV_to_nm(array):
    '''Converts eV (energy) to nm (wavelength)'''
    return 1e9*2.998e8*4.136e-15/array

def nm_to_eV(array):
    '''Converts nm (wavelength) to eV (energy)'''
    return (1e9 * 2.998e8 * 4.136e-15)/array

def dirac_delta(a,b,**kwargs):
    '''
    Provides Gaussian approximating Dirac delta function
    :param a: np.ndarray, x_array
    :param b: float, x_center
    :param kwargs: dict
    :return: np.ndarray (size = np.size(a))
    '''
    c = .01
    for key,value in kwargs.items():
        if key == "width":
            c = value/4.291
    delta = np.exp(-.5 * (a-b) ** 2 / c ** 2)
    return delta

def linear_spec(energy_spectrum,energy_array,osc_strength,linewidth):
    '''
    Provides linear absorption spectrum for a set of Loretnzian oscillators
    :param energy_spectrum: np.ndarray (1D)
    :param energy_array: np.ndarray (1D)
    :param osc_strength: np.ndarray (1D0
    :param linewidth: np.ndarray or float
    :return: np.ndarray (size = np.size(energy_spectrum)
    '''
    fsc = 0.007297
    num_states = np.size(energy_array)
    intensity_array = np.zeros(np.size(energy_spectrum))
    if (type(linewidth) is np.ndarray):
        linewidth_array = linewidth
    else:
        linewidth_array = np.ones(num_states)*linewidth
    for i in range(0,num_states):
        oscillator_strength = osc_strength[i]
        peak_shape = fsc/(energy_spectrum**2)*dirac_delta(energy_spectrum,energy_array[i],**{"width":linewidth_array[i]})
        intensity_array = intensity_array+oscillator_strength*peak_shape
    return intensity_array

def linear_spec_no_div(energy_spectrum,energy_array,osc_strength,linewidth):
    '''
    Provides linear absorption spectrum for a set of Loretnzian oscillators w/o
    dividing by \omega**2
    :param energy_spectrum: np.ndarray (1D)
    :param energy_array: np.ndarray (1D)
    :param osc_strength: np.ndarray (1D0
    :param linewidth: np.ndarray or float
    :return: np.ndarray (size = np.size(energy_spectrum)
    '''
    fsc = 0.007297
    num_states = np.size(energy_array)
    intensity_array = np.zeros(np.size(energy_spectrum))
    if (type(linewidth) is np.ndarray):
        linewidth_array = linewidth
    else:
        linewidth_array = np.ones(num_states)*linewidth
    for i in range(0,num_states):
        oscillator_strength = osc_strength[i]
        peak_shape = fsc*dirac_delta(energy_spectrum,energy_array[i],**{"width":linewidth_array[i]})
        intensity_array = intensity_array+oscillator_strength*peak_shape
    return intensity_array

def normalized_cost_function(normalized_training_data,test_data,bound = 1e-10):
    '''mean squared cost for unnormed data'''
    n = np.max(np.abs(test_data))
    if (n>bound):
        norm_test_data = test_data / n
        cost = np.sum((normalized_training_data- norm_test_data) ** 2)
    else:
        cost = 1e5
    return cost

def extract_cd_peaks(x_data,y_data):
    '''Extracts x and y coordinates of circular dichroism peaks from y_data'''
    max_val = np.max(np.abs(y_data))
    prominence_factor = 1/10
    max_peaks = find_peaks(y_data,height = 0, prominence= prominence_factor*max_val)
    min_peaks = find_peaks(-1*y_data,height= 0,prominence= prominence_factor*max_val)
    x_max = x_data[max_peaks[0]]
    x_min = x_data[min_peaks[0]]
    x_tot = np.hstack((x_max,x_min))
    y_tot = np.hstack((y_data[max_peaks[0]],y_data[min_peaks[0]]))
    x_ind_sorted  = x_tot.argsort()
    x_tot_sorted = x_tot[x_ind_sorted]
    y_tot_sorted = y_tot[x_ind_sorted]
    return x_tot_sorted, y_tot_sorted

#only works properly if there are equal numbers of peaks in both the fit and the data spectra, but it won't crash otherwise
def cost_peak_only_function(x_data,y_data,x_fit,y_fit,lower_bound = 0):
    '''Cost function from extracted peaks. Largely obsolete'''
    num_peaks_data = np.size(x_data)
    num_peaks_fit = np.size(x_fit)
    num_peaks = np.min(np.array([num_peaks_data,num_peaks_fit]))
    if (num_peaks ==0):
        cost  = 1e10
    else:
        y_data_max = np.max(np.abs(y_data))
        if (y_data_max <= lower_bound):
            cost = 1e10
        else:
            y_fit_max = np.max(np.abs(y_fit))
            #truncation steps to avoid errors
            x_data = x_data[:num_peaks]
            y_data = y_data[:num_peaks]/y_data_max
            x_fit = x_fit[:num_peaks]
            y_fit = y_fit[:num_peaks]/y_fit_max
            cost = (np.sum(((x_data-x_fit)**2+(y_data-y_fit)**2)))/num_peaks
    return cost

#assumes that sizes and spectra spanning is identical for all
def cost_semi_sum(semi_sum_data,semi_sum_fit,y_data,y_fit):
    '''mean square cost function. Deprecated, use cost_semi_sum_v2()'''
    mixed_data = semi_sum_data/np.max(np.abs(y_data))
    mixed_fit = semi_sum_fit/np.max(np.abs(y_fit))
    cost = (np.max(np.abs(mixed_fit))-np.max(np.abs(mixed_data)))**2
    return cost


#tries to get shape right via least squares regression
def cost_semi_sum_v2(semi_sum_data,semi_sum_fit):
    '''mean square cost function'''
    semi_sum_fit = pu.norm_array(semi_sum_fit)
    semi_sum_data=  pu.norm_array(semi_sum_data)
    return np.sum((semi_sum_fit-semi_sum_data)**2)/(np.sum(semi_sum_data**2))

def interpolate_raw_data(x_raw_data,y_raw_data,spectrum):
    '''Interpolation of data onto new spectrum'''
    interpolated_data = np.interp(spectrum,x_raw_data,y_raw_data)
    return interpolated_data

def ldlb_3d_from_params(dipole_mat,rot_array,e_array,spec,unit_defs,dielectric_params):
    '''
    Provides LDLB spectra from Lorentzian oscillator parameters
    :param dipole_mat: np.ndarray
    :param rot_array: np.ndarray
    :param e_array: np.ndarray
    :param spec: np.ndarray
    :param unit_defs: UNIT_DEFS()
    :param dielectric_params: DIELECTRIC_PARAMS
    :return: np.ndarray
    '''
    rotate_dip_mat = rotate_vector(rot_array, dipole_mat, transpose=True)
    rotated_dielectric = create_dielectric_tensor(dielectric_params, rotate_dip_mat, e_array, spec, unit_defs,
                                                     **{"dimension": 3})
    ldlb = linear_optics_from_dielectric_tensor(rotated_dielectric,spec).ldlb()
    return ldlb

#vib_split_array and indices_to_dress should be same size
def vibronic_dressing(energy_array_init,vib_split_array,indices_to_dress):
    '''
    Addes vibronic dressing to a inital electronic energy array
    :param energy_array_init: np.ndarray (1D)
    :param vib_split_array: np.ndarray (1D) (e_vib for chosen indices)
    :param indices_to_dress: np.ndarray (1D)
    :return: np.ndarray (1D)
    '''
    e_array = energy_array_init
    if (np.size(vib_split_array) != np.size(indices_to_dress)):
        print("Error: indices to dress and vib split array sizes must be identical")
        return energy_array_init
    new_size = np.int(np.size(e_array)+2*np.size(indices_to_dress))
    new_e_array = np.zeros(new_size)
    counter = 0
    for i in range(0,np.size(e_array)):
        if (not (np.any(indices_to_dress == i))):
            new_e_array[counter] = e_array[i]
            counter = counter+1
        else:
            vib_id = np.where(indices_to_dress == i)
            new_e_array[counter] = e_array[i]-vib_split_array[vib_id]
            counter = counter+1
            new_e_array[counter] = e_array[i]
            counter = counter+1
            new_e_array[counter] = e_array[i]+vib_split_array[vib_id]
            counter = counter+1
    return new_e_array

def fc_factor(huang_rhys,modes):
    '''Franck-Condon factor for array of vibronic modes and scalar Huang-Rhys factor. See (2.7) in Excitonic and exciton-phonon interactions in molecular aggregates, Roel Tempelaar'''
    return np.sqrt((huang_rhys**(2*modes)*np.exp(-huang_rhys**2))/(factorial(modes)))

def vib_spec_osc_str(electronic_osc_strength,modes_array,huang_rhys):
    '''Vibrational oscillator parameters for modes of a single transition'''
    fc_factor_array = fc_factor(huang_rhys,modes_array)**2
    return electronic_osc_strength*fc_factor_array

def vib_spec_dip_mat(dipole_mat,modes_array,huang_rhys):
    '''Vibrational dipole matrix from electronic dipole matrix '''
    fc_factor_array = fc_factor(huang_rhys, modes_array)
    if (dipole_mat.ndim == 1):
        dipole_mat = dipole_mat[:,np.newaxis]
    new_dipoles = np.einsum("ai,bj->ib", fc_factor_array, dipole_mat)
    return new_dipoles



def ldlb_prefactor_2(e_inf,volume,length,unit_defs = unit_defs_base):
    '''Prefactor v2 for LDLB'''
    return length/(unit_defs.c*unit_defs.hbar*unit_defs.e0*np.sqrt(e_inf)*volume)

def f_dielectric_real(resonant_energy,spectrum,damping_factor):
    '''
    Lineshape of real part of refractive index spectra
    :param resonant_energy: np.ndarray
    :param spectrum: np.ndarray
    :param damping_factor: np.ndarray
    :return: np.ndarray
    '''
    if (np.isscalar(resonant_energy)):
        return (resonant_energy ** 2 - spectrum ** 2) / ((resonant_energy ** 2 - spectrum ** 2) ** 2 + damping_factor ** 2 * spectrum ** 2)
    else:
        spec_size = np.size(spectrum)
        energy_size = np.size(resonant_energy)
        resonant_energy = np.tile(resonant_energy,(spec_size,1)).T
        if (np.size(damping_factor) == energy_size):
            damping_factor = np.tile(damping_factor,(spec_size,1)).T
        else:
            damping_factor =  np.tile(damping_factor,(energy_size,1))
        spectrum = np.tile(spectrum,(energy_size,1))
        return (resonant_energy ** 2 - spectrum ** 2)/((resonant_energy ** 2 - spectrum ** 2) ** 2 + damping_factor ** 2 * spectrum ** 2)


def f_dielectric_im(resonant_energy, spectrum, damping_factor):
    '''
    Lineshape of imaginary part of refractive index spectra
    :param resonant_energy: np.ndarray
    :param spectrum: np.ndarray
    :param damping_factor: np.ndarray
    :return: np.ndarray
    '''
    if (np.isscalar(resonant_energy)):
        return damping_factor * spectrum / (
                (resonant_energy ** 2 - spectrum ** 2) ** 2 + damping_factor ** 2 * spectrum ** 2)
    else:
        spec_size = np.size(spectrum)
        energy_size = np.size(resonant_energy)
        resonant_energy = np.tile(resonant_energy,(spec_size,1)).T
        if (np.size(damping_factor) == energy_size):
            damping_factor = np.tile(damping_factor, (spec_size, 1)).T
        else:
            damping_factor = np.tile(damping_factor, (energy_size, 1))
        spectrum = np.tile(spectrum,(energy_size,1))
        return damping_factor*spectrum/((resonant_energy**2-spectrum**2)**2+damping_factor**2*spectrum**2)

def ldlb_helical_perturbative(spectrum,dielectric_params,e_array,dip_mags,dip_angles,gamma_array,total_rotation):
    '''
    Perterbative analytic treatement of LDLB for a helix
    Minus sign at end is to account for the change in sign convention
    :param spectrum: np.ndarray (1D)
    :param dielectric_params: DIELECTRIC_PARAMS()
    :param e_array: np.ndarray (1D)
    :param dip_mags: np.ndarray (1D)
    :param dip_angles: np.ndarray (1D)
    :param gamma_array: np.ndarray (1D)
    :param total_rotation: np.float (radians)
    :return: np.array (LDLB)
    '''
    prefactor =  ldlb_prefactor_2(dielectric_params.epsilon_inf,dielectric_params.v,dielectric_params.length)
    #neg one due to sign error in deriv.
    num = np.size(e_array)
    b = total_rotation

    if b == 0:
        b = 1e-10 #to avoid divide by zero issues
    net_result = np.zeros(np.size(spectrum))
    dichroism_p = np.zeros(np.size(spectrum))
    for m in range(0,num): #ld terms
        i_m = e_array[m]*dip_mags[m]**2
        v_m = gamma_array[m]*spectrum/((e_array[m]**2-spectrum**2)**2+gamma_array[m]**2*spectrum**2)
        for n in range(0,num): #lb terms
            i_n = e_array[n]*dip_mags[n]**2
            w_n = (e_array[n]**2-spectrum**2)/((e_array[n]**2-spectrum**2)**2+gamma_array[n]**2*spectrum**2)
            angle_factor = np.sin(2*(dip_angles[m]-dip_angles[n]))*np.sin(b)**2/(2*b**2) \
                            -np.cos(2*(dip_angles[m]-dip_angles[n]))*(b-np.sin(b)*np.cos(b))/(2*b**2)
            net_result = net_result+i_n*i_m*w_n*v_m*angle_factor
    return -spectrum**2*prefactor**2*net_result

def extract_diagonal_tensor(tensor):
    return np.diagonal(tensor).T

def tensor_stack_coordinate_transform(tensor,coordinate_transform):
    transform_inverse =np.moveaxis(np.linalg.inv(np.moveaxis(coordinate_transform,-1,0)),0,-1)
    return np.einsum("ijl,jkl->ikl", coordinate_transform,
                                     np.einsum("ijl,jkl->ikl", tensor, transform_inverse))

def characteristic_aborption_lengths(dielectric_tensor,spectrum):
    '''
    Extracts absorption lengths--largely redundant
    :param dielectric_tensor: np.ndarray (1D)
    :param spectrum: np.ndarray (1D)
    :return: (abs,angles,n) (all np.ndarray)
    '''
    diagonalized_dielectric_real,rotation_matrix_real = diagonalize_matrix_stack(dielectric_tensor.real,axis_set = "xy",to_reorder = False)
    dielectric_rotated_no_birefringence = tensor_stack_coordinate_transform(dielectric_tensor,rotation_matrix_real)
    dielectric_im_rotated_no_biref = extract_diagonal_tensor(dielectric_rotated_no_birefringence.imag)
    refractive_index_no_biref = extract_diagonal_tensor(get_refractive_index_tensor(dielectric_rotated_no_birefringence).real)
    spectrum_tiled = np.tile(spectrum,(np.size(refractive_index_no_biref,axis =0),1))
    abs_no_biref = spectrum_tiled*dielectric_im_rotated_no_biref/refractive_index_no_biref
    angles = rotation_matrix_3D_to_angles(rotation_matrix_real)
    return abs_no_biref, angles, refractive_index_no_biref

def characteristic_aborption_lengths_no_rotation(dielectric_tensor,spectrum):
    '''
    Extracts abs w/o rotation--largely redundant
    :param dielectric_tensor: np.ndarray (1D)
    :param spectrum: np.ndarray (1D)
    :return: (abs,angles,n) (all np.ndarray)
    '''
    refractive_index_tensor = get_refractive_index_tensor(dielectric_tensor)
    refractive_index = extract_diagonal_tensor(refractive_index_tensor.real)
    dielectric_im = extract_diagonal_tensor(dielectric_tensor.imag)
    spectrum_tiled = np.tile(spectrum,(np.size(refractive_index,axis =0),1))
    abs = spectrum_tiled*dielectric_im/refractive_index.real
    return abs, refractive_index

'''
The below functions come form Molecular Light Scattering and Optical Activity 
by Laurence Barron, which understands chiroptical effects in terms of 
tensorial components for a system. To maintain consistency and for 
simplicity of construction, we borrow this method here for CB and CD

For symbols, Barron uses Δθ = ωl/2c (n_l-n_r) for optical rotation (circular birefrigence)
and η = ωl/2c (n'_l-n'_r) for ellipcitity (circular dichroism) where n' is the imaginary
part of the refractive index 

In this code base, terms are defined as CB (or β_3)  = ωl/2c (n_l-n_r)
and CD (or β]d_3)  = ωl/2c (n'_l-n'_r) in accordence with symbols for linear dichroism
(LD) and linear birefringence (LB) 
'''

def disp_f(omega_n,spec,gamma_n):
    '''
    Analogous to W_n in rest of code--rewritten for use with Barron's tensors
    :return: np.ndarray
    '''
    return (omega_n**2-spec**2)/((omega_n**2-spec**2)**2+spec**2*gamma_n**2)

def disp_g(omega_n,spec,gamma_n):
    '''
    Analogous to V_n in rest of code--rewritten for use with Barron's tensors
    :return: np.ndarray
    '''
    return (spec*gamma_n) / ((omega_n ** 2 - spec ** 2) ** 2 + spec ** 2 * gamma_n ** 2)

def get_alpha_tensor(omega_n,spec,mu_fi_vec,mu_if_vec,gamma_n):
    '''
    Modified for within absorption band
    :param omega_n:
    :param spec:
    :param mu_fi_vec:
    :param mu_if_vec:
    :param gamma_n:
    :return:
    '''
    pref = 2/unit_defs_base.hbar
    alpha_f = pref*np.einsum("l,jk->jkl",omega_n*disp_f(omega_n,spec,gamma_n),np.real(np.outer(mu_fi_vec,mu_if_vec)))
    alpha_g = pref*np.einsum("l,jk->jkl",omega_n*disp_f(omega_n,spec,gamma_n),np.real(np.outer(mu_fi_vec,mu_if_vec)))
    return alpha_f+1j*alpha_g


def get_alpha_prime_tensor(omega_n,spec,mu_fi_vec,mu_if_vec,gamma_n):
    '''
    Modified for within absorption band
    :param omega_n:
    :param spec:
    :param mu_fi_vec:
    :param mu_if_vec:
    :param gamma_n:
    :return:
    '''
    pref = 2/unit_defs_base.hbar
    alpha_f = pref*np.einsum("l,jk->jkl",spec*disp_f(omega_n,spec,gamma_n),
                             np.imag(np.outer(mu_fi_vec,mu_if_vec)))
    alpha_g = pref*np.einsum("l,jk->jkl",spec*disp_g(omega_n,spec,gamma_n),
                             np.imag(np.outer(mu_fi_vec,mu_if_vec)))
    return alpha_f+1j*alpha_g

def get_g_tensor(omega_n,spec,mu_fi_vec,m_if_vec,gamma_n):
    pref = 2/unit_defs_base.hbar
    g_f = pref*np.einsum("l,jk->jkl",omega_n*disp_f(omega_n,spec,gamma_n),
                         np.real(np.outer(mu_fi_vec,m_if_vec)))
    g_g = pref*np.einsum("l,jk->jkl",omega_n*disp_g(omega_n,spec,gamma_n),
                         np.real(np.outer(mu_fi_vec,m_if_vec)))
    return g_f+1j*g_g

def get_g_tensor_prime(omega_n,spec,mu_fi_vec,m_if_vec,gamma_n):
    pref = -2/unit_defs_base.hbar
    g_f = pref*np.einsum("l,jk->jkl",spec*disp_f(omega_n,spec,gamma_n),
                         np.imag(np.outer(mu_fi_vec,m_if_vec)))
    g_g = pref*np.einsum("l,jk->jkl",spec*disp_g(omega_n,spec,gamma_n),
                         np.imag(np.outer(mu_fi_vec,m_if_vec)))
    return g_f+1j*g_g

def get_a_tensor(omega_n,spec,mu_fi_vec,quad_if_tensor,gamma_n):
    pref = 2/unit_defs_base.hbar
    a_f = pref*np.einsum("l,jkg->jkgl",omega_n*disp_f(omega_n,spec,gamma_n),
                         np.real(np.einsum("a,bg->abg",mu_fi_vec,quad_if_tensor)))
    a_g = pref*np.einsum("l,jkg->jkgl",omega_n*disp_g(omega_n,spec,gamma_n),\
                         np.real(np.einsum("a,bg->abg",mu_fi_vec,quad_if_tensor)))
    return a_f+1j*a_g

def get_a_prime_tensor(omega_n,spec,mu_fi_vec,quad_if_tensor,gamma_n):
    pref = -2/unit_defs_base.hbar
    a_f = pref * np.einsum("l,jkg->jkgl", spec * disp_f(omega_n, spec, gamma_n),
                           np.imag(np.einsum("a,bg->abg", mu_fi_vec, quad_if_tensor)))
    a_g = pref * np.einsum("l,jkg->jkgl", spec * disp_g(omega_n, spec, gamma_n),
                           np.imag(np.einsum("a,bg->abg", mu_fi_vec, quad_if_tensor)))
    return a_f+1j*a_g

def get_d_tensor(omega_n,spec,m_fi_vec,quad_if_tensor,gamma_n):
    pref = 2/unit_defs_base.hbar
    d_f = pref*np.einsum("l,jkg->jkgl",omega_n*disp_f(omega_n,spec,gamma_n),
                         np.real(np.einsum("a,bg->abg",m_fi_vec,quad_if_tensor)))
    d_g = pref*np.einsum("l,jkg->jkgl",omega_n*disp_g(omega_n,spec,gamma_n),\
                         np.real(np.einsum("a,bg->abg",m_fi_vec,quad_if_tensor)))
    return d_f+1j*d_g


def get_d_tensor_prime(omega_n,spec,m_fi_vec,quad_if_tensor,gamma_n):
    pref = -2/unit_defs_base.hbar
    d_f = pref*np.einsum("l,jkg->jkgl",spec*disp_f(omega_n,spec,gamma_n),
                         np.imag(np.einsum("a,bg->abg",m_fi_vec,quad_if_tensor)))
    d_g = pref*np.einsum("l,jkg->jkgl",spec*disp_g(omega_n,spec,gamma_n),\
                         np.imag(np.einsum("a,bg->abg",m_fi_vec,quad_if_tensor)))
    return d_f+1j*d_g

def oriented_circular_birefringence(spec,a_tensor,g_prime_tensor,number_density = 1):
    '''
    #5.2.1a in Barron
    i,j,k are xyz indices
    :param spec: np.array (ω)
    :param a_tensor: np.ndarray (i,j,k,ω)
    :param g_prime_tensor: np.ndarray (i,j,ω)
    :param number_density: float (default 1.0)
    :return: np.ndarray (ω)
    '''

    mu0 = 1/(unit_defs_base.c**2*unit_defs_base.e0)
    pref=  -spec*mu0*number_density
    a_component = 1/3*spec*(np.real(a_tensor[0,1,2,:]-a_tensor[1,0,2,:]))
    #A_x,yz(f) - A_y,xz(f)--note that Re(A) = A(f) and that Im(A) = A(g)
    g_component = np.real(g_prime_tensor[0,0,:]+g_prime_tensor[1,1,:])
    return pref*(a_component+g_component)

def isotropic_circular_birefringence(spec,g_prime,number_density = 1):
    '''
    5.2.2b in Barron
    :param spec: np.array (ω)
    :param g_prime: np.ndarray (ω)
    g_prime is the g_prime tensor diagonal magnitude
    :param number_density: float (default 1.0)
    :return: np.ndarray (ω)
    '''
    #5.2.2a in Barron
    mu0 = 1/(unit_defs_base.c**2*unit_defs_base.e0)
    pref=  -2/3*spec*mu0*number_density
    g_component = np.real(g_prime)
    return pref*(g_component)


def oriented_circular_dichroism(spec, a_tensor, g_prime_tensor, number_density=1.0):
    '''
    5.2.1b in Barron
    i,j,k are xyz indices
    :param spec: np.array (ω)
    :param a_tensor: np.ndarray (i,j,k,ω)
    :param g_prime_tensor: np.ndarray (i,j,ω)
    :param number_density: float (default 1.0)
    :return: np.ndarray (ω)
    '''

    mu0 = 1 / (unit_defs_base.c ** 2 * unit_defs_base.e0)
    pref = -spec * mu0 * number_density
    a_component = 1 / 3 * spec * (np.imag(a_tensor[0, 1, 2, :] - a_tensor[1, 0, 2, :]))
    # A_x,yz(f) - A_y,xz(f)--note that Re(A) = A(f) and that Im(A) = A(g)
    g_component = np.imag(g_prime_tensor[0, 0, :] + g_prime_tensor[1, 1, :])
    return pref * (a_component + g_component)


def isotropic_circular_birefringence(spec, g_prime, number_density=1):
    '''
    5.2.2b in Barron
    :param spec: np.array (ω)
    :param g_prime: np.ndarray (ω)
    g_prime is the g_prime tensor diagonal magnitude
    :param number_density: float (default 1.0)
    :return: np.ndarray (ω)
    '''

    mu0 = 1 / (unit_defs_base.c ** 2 * unit_defs_base.e0)
    pref = -2 / 3 * spec * mu0 * number_density
    g_component = np.imag(g_prime)
    return pref * (g_component)

def get_core_tensors(spec,mu_vec_matrix,m_vec_matrix,quad_tensor_matrix,omega_n_matrix,gamma_n_matrix):
    '''

    :param spec: np.ndarray (ω)
    :param mu_vec_matrix: np.ndarray (n,n,3)
    :param m_vec_matrix: np.ndarray (n,n,3)
    :param quad_tensor_matrix: np.ndarray (n,n,3,3)
    :param omega_n_matrix: np.ndarray(n,n)
    :param gamma_n_matrix: np.ndarray(n,n)
    :return:
    '''
    alpha_tensor = np.zeros((3,3,np.size(spec)),dtype = np.cdouble)
    alpha_prime_tensor = np.zeros((3,3,np.size(spec)),dtype = np.cdouble)
    a_tensor = np.zeros((3, 3, np.size(spec)), dtype=np.cdouble)
    g_tensor = np.zeros((3, 3, np.size(spec)), dtype=np.cdouble)
    g_tensor_prime = np.zeros((3, 3, np.size(spec)), dtype=np.cdouble)
    for i in range(np.size(mu_vec_matrix,axis = 0)):
        for f in range(i+1,np.size(mu_vec_matrix,axis =0)): #only considering transitions from lower to higher states
            mu_fi_vec = mu_vec_matrix[f,i,:]
            mu_if_vec = mu_vec_matrix[i,f,:]
            quad_if_tensor = quad_tensor_matrix[i,f,:,:]
            m_if_vec = m_vec_matrix[i,f,:]
            omega_n = omega_n_matrix[f,i]
            gamma_n = gamma_n_matrix[f,i]
            alpha_tensor= alpha_tensor+get_alpha_tensor(omega_n,spec,mu_fi_vec,mu_if_vec,gamma_n)
            alpha_prime_tensor = alpha_prime_tensor+get_alpha_prime_tensor(omega_n,spec,mu_fi_vec,mu_if_vec,gamma_n)
            a_tensor = a_tensor+get_a_tensor(omega_n,spec,mu_fi_vec,quad_if_tensor,gamma_n)
            g_tensor = g_tensor+get_g_tensor(omega_n,spec,mu_fi_vec,m_if_vec,gamma_n)
            g_tensor_prime = g_tensor_prime+get_g_tensor_prime(omega_n,spec,mu_fi_vec,m_if_vec,gamma_n)
    return alpha_tensor, alpha_prime_tensor, a_tensor,g_tensor,g_tensor_prime

def get_alpha_tensors_simple(spec,mu_vec_matrix,omega_n_array,gamma_n_array):
    alpha_tensor = np.zeros((3, 3, np.size(spec)), dtype=np.cdouble)
    alpha_prime_tensor = np.zeros((3, 3, np.size(spec)), dtype=np.cdouble)
    num_transitions = np.size(omega_n_array)
    assert np.size(mu_vec_matrix,axis =1)==3,"dipoles must have 3 dimensions"
    for i in range(num_transitions):
        omega_n = omega_n_array[i]
        mu_fi_vec = mu_vec_matrix[i,:]
        mu_if_vec = np.conjugate(mu_fi_vec) #this should be true--check w/ Roel
        gamma_n = gamma_n_array[i]
        alpha_tensor = alpha_tensor + get_alpha_tensor(omega_n, spec, mu_fi_vec, mu_if_vec, gamma_n)
        alpha_prime_tensor = alpha_prime_tensor + get_alpha_prime_tensor(omega_n, spec, mu_fi_vec, mu_if_vec,
                                                                         gamma_n)
    return alpha_tensor, alpha_prime_tensor

def get_complex_alpha(spec,mu_vecs_matrix,omega_n_array,gamma_n_array):
    alpha_tensor, alpha_prime_tensor =get_alpha_tensors_simple(spec,mu_vecs_matrix,omega_n_array,gamma_n_array)
    return alpha_tensor-1j*alpha_prime_tensor

def refractive_index_from_complex_alpha_tensor(complex_alpha_tensor,number_density = 1.0):
    '''
    Eq. 3.4.7 in Barron
    :param complex_alpha_tensor:
    :param number_density: float (default 1.0)
    :return:
    '''
    pref = 1/(2*unit_defs_base.e0)*number_density
    refractive_index_tensor = 1+pref*complex_alpha_tensor
    return refractive_index_tensor

def linear_optics_from_complex_scattering_tensor(complex_scattering_tensor,number_density = 1.0):
    '''
    In dipole approximation complex scattering tensor and complex polarizability tensor (alpha) are the same
    :param complex_scattering_tensor:
    :param number_density:
    :return:
    '''
    ref_index_tensor = refractive_index_from_complex_alpha_tensor(complex_scattering_tensor,number_density)
    rotation_matrix = get_xy_rotation_matrix(axis_dimension =3,angle= -np.pi/4) #to x',y' basis
    complex_alpha_rotated = tensor_stack_coordinate_transform(complex_scattering_tensor,rotation_matrix)
    ref_index_prime = refractive_index_from_complex_alpha_tensor(complex_alpha_rotated,number_density)
    return linear_optics_from_refractive_index_tensor(ref_index_tensor,ref_index_prime)

def oriented_circular_dichroism_birefringence_simple(spec,omega_array,mu_fi_matrix,m_if_matrix,gamma_array,number_density = 1.0):
    '''
    Simplified method of determining oriented CD and CB that omits quadrupole contributions
    and, instead of directly dealing with the entire matrices of matrix operators,
    only considers n listed states with angular frequencies given by omega_array.
    What is returned is the differential behavior (in inverse length units). That is,
    we do not consider the pathlength here.
    :param spec: np.array (
    :param omega_array:
    :param mu_fi_matrix:
    :param m_if_matrix:
    :param gamma_array:
    :param number_density: float (default 1.0)
    :return:
    '''
    g_tensor_prime = np.zeros((3,3,np.size(spec)),dtype = np.csingle)
    for i in range(np.size(omega_array)):
        omega_n = omega_array[i]
        mu_fi_vec = mu_fi_matrix[i,:]
        m_if_vec = m_if_matrix[i,:]
        gamma_n = gamma_array[i]
        g_tensor_prime = g_tensor_prime+get_g_tensor_prime(omega_n,spec,mu_fi_vec,m_if_vec,gamma_n)
    pref = -1/2*spec*1/(unit_defs_base.e0*unit_defs_base.c**2)*number_density
    circular_dichroism = pref*np.imag(g_tensor_prime[0,0,:]+g_tensor_prime[1,1,:])
    circular_birefringence = pref*np.real(g_tensor_prime[0,0,:]+g_tensor_prime[1,1,:])
    return circular_dichroism,circular_birefringence

def wannier_stark_cd_cb_simple(spec,omega_array,wannier_shift,mu_fi_matrix,m_if_matrix,gamma_array,number_density = 1.0):
    cd_matrix = np.zeros((np.size(spec),np.size(wannier_shift,axis= -1)))
    cb_matrix = np.zeros((np.size(spec),np.size(wannier_shift,axis= -1)))
    for i in range(np.size(wannier_shift,axis =-1)):
        cd_matrix[:,i],cb_matrix[:,i] = oriented_circular_dichroism_birefringence_simple(spec,omega_array+wannier_shift[:,i],mu_fi_matrix,m_if_matrix,gamma_array,number_density)
    return cd_matrix,cb_matrix

def strain_dielectric_tensor(dielectric_tensor_init,strain_tensor,alpha_coefs):
    '''
    Takes according to Eq. 4 in https://aip.scitation.org/doi/pdf/10.1063/1.2073977
    Note that this is natively for isotropic dielectric functions, which should be
    fine for the DBRs, but might be not fully accurate for anisotropic films,
    which a check of Lifshitz and Landau should help .
    alpha_coefs are dielectric-specific coefficients that relate the strained dielectric
    to the incident strain, with the first being for the tensor and the tensor for the invariant
    :param dielectric_tensor_init: np.ndarray (2D or 3D) (3,3) or (3,3,omega)
    :param strain_tensor: np.ndarray (2D or 3D) (3,3) or (3,3,omega)
    :param alpha_coefs: np.array (2)
    :return: np.ndarray (2D or 3D) (3,3) or (3,3,omega)
    '''
    strain_dim = np.ndim(strain_tensor)
    eps_dim = np.ndim(dielectric_tensor_init)
    strain_invariant= np.trace(strain_tensor)
    anisotropic_dielectric_strain = alpha_coefs[0]*strain_tensor
    if (strain_dim ==2):
        isotropic_dielectric_strain = alpha_coefs[1]*strain_invariant*np.identity(3)
    elif(strain_dim==3):
        isotropic_dielectric_strain = alpha_coefs[1] * strain_invariant * np.dstack([np.identity*np.size(dielectric_tensor_init,axis = -1)])
    else:raise ValueError("Invalid strain dimension")
    total_strain = anisotropic_dielectric_strain+isotropic_dielectric_strain
    if (strain_dim != eps_dim):
        if (strain_dim > eps_dim): #this is unlikely to occur, but code written for coverage
            dielectric_tensor_init = np.reshape(np.repeat(dielectric_tensor_init,np.size(strain_tensor,axis= -1)),np.shape(strain_tensor))
        if (eps_dim> strain_dim):
            total_strain = np.reshape(np.repeat(total_strain,np.size(dielectric_tensor_init,axis= -1)),np.shape(dielectric_tensor_init))
    dielectric_strained_tensor = dielectric_tensor_init + total_strain
    return dielectric_strained_tensor

def get_stress_vector_from_tensor(stress_tensor):
    '''
    Extracts stress vector from stress tensor
    :param stress_tensor:
    :return:
    '''
    s = stress_tensor #for brevity
    stress_vector=  np.array([s[0,0],s[1,1],s[2,2],s[1,2],s[2,0],s[0,1]])
    return stress_vector

def get_strain_tensor_from_vector(strain_vector):
    '''
    Note that the elements 3-5 are 2X that of the strain tensor
    :param strain_vector:
    :return:
    '''
    s = strain_vector
    strain_tensor = np.array([[s[0],0.5*s[5],0.5*s[4]],
                              [0.5*s[5],s[1],0.5*s[3]],
                              [0.5*s[4],0.5*s[3],s[2]]])
    return strain_tensor

def strain_tensor_hookes_law_isotropic_from_stress(stress_tensor,stiffness_matrix):
    '''
    :param stress_tensor:
    :param stiffness_matrix
    :return:
    '''
    stress_vector = get_stress_vector_from_tensor(stress_tensor)
    strain_vector=  np.einsum("ij,j->i",stiffness_matrix,stress_vector)
    strain_tensor = get_strain_tensor_from_vector(strain_vector)
    return strain_tensor

def create_stiffness_matrix(youngs_modulus,poissons_ratio):
    '''
    Creates the stiffness matrix (4 dim tensor) that relates stress tensor to strain
    tensor.
    See Atanackovic, T. M., & Guran, A. (2000). Theory of elasticity for scientists and engineers. Springer Science & Business Media.
    :param youngs_modulus:
    :param poissons_ratio:
    :return:
    '''
    v = poissons_ratio #for brevity, used for symbolic similarity to ν
    poisson_ratio_matrix = np.array([[1,-v,-v,0,0,0],
                                     [-v,1,-v,0,0,0],
                                     [-v,-v,1,0,0,0],
                                     [0,0,0,2*(1+v),0,0],
                                     [0,0,0,0,2*(1+v),0],
                                     [0,0,0,0,0,2*(1+v)]])
    return 1/youngs_modulus*poisson_ratio_matrix

def get_poissons_ratio(material_name):
    if (material_name== "sio2"):
        poissons_ratio = .19 #on Si substrate
        #from https://link.springer.com/content/pdf/10.1007/s10853-009-3365-3.pdf
    if (material_name=="ta2o5"):
        poissons_ratio = .33 # on Si substrate
        # from https://link.springer.com/content/pdf/10.1007/s10853-009-3365-3.pdf
    return poissons_ratio

def get_youngs_modulus(material_name):
    '''
    Units in giga pascals
    :param material_name: str
    :return: np.float
    '''
    if (material_name== "sio2"):
        youngs_modulus = 91 #on Si substrate
        #from https://link.springer.com/content/pdf/10.1007/s10853-009-3365-3.pdf
    if (material_name=="ta2o5"):
        youngs_modulus = 138# on Si substrate
        # from https://link.springer.com/content/pdf/10.1007/s10853-009-3365-3.pdf
    return youngs_modulus

class Elastic_Material():
    '''
    Container class for (isotropic) elastic materials
    '''
    def __init__(self,material_name):
        self.name = material_name.lower()
        self.youngs_modulus = get_youngs_modulus(self.name)
        self.poissons_ratio = get_poissons_ratio(self.name)
    def get_stiffness_matrix(self):
        return create_stiffness_matrix(self.youngs_modulus,self.poissons_ratio)


def get_electrostriction_parameters(dielectric_constant,material_type = "isotropic"):
    '''
    Provides the model dielectrostriction coefficients for a variety of
    material types. Note that cubic materials also require a third paramter
    not given here.
    From Table 1 in https://sor.scitation.org/doi/pdf/10.1122/1.1835340
    :param dielectric_constant: float or np.array
    :param material_type: str
    "isotropic": For an isotropic solid, from (Shkel and Klingenberg 1998)
    "cubic": For a cubic solid (Anderson 1986)
    "newtonian": For a newtonian fluid (Stratton 1941)
    :return: list (2)
    '''
    material_type = pu.clean_up_string(material_type)
    eps = dielectric_constant
    if (material_type=="newtonian"):
        alpha_1 = 0*eps #multiplication to ensure type consistency
        alpha_2 = -1/3*(eps+1)*(eps+2)
    elif(material_type=="isotropic"):
        alpha_1 = -2/5*(eps-1)**2
        alpha_2 = -1/3*(eps-1)*(eps+2)+2/15*(eps-1)**2
    elif(material_type=="cubic"):
        alpha_1 = -1.515*(eps-1)**2
        alpha_2 = -1/3*(eps-1)*(eps+2)*0.505*(eps-1)**2
    else: raise ValueError("Invalid material_type")
    return [alpha_1,alpha_2]

