import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import python_util as pu
import scipy.linalg
import scipy
import collections


'''Numeric and symbolic handling of Mueller matrices with some plotting functions

Andrew Salij'''

def commute_matrix_stacks(matrix_a,matrix_b):
    '''
    Returns commutator of a matrix with a same shape stack of matrices
    Designed for matrix to be magnus term after integration and stakc to be S term
    :param matrix: np.ndarray (N,N) or (N,N,t)
    :param matrix_stack: np.ndarray (N,N) or (N,N,t)
    :return: np.ndarray (N,N,t)
    '''
    ab,ba = commute_handling(matrix_a,matrix_b)
    return np.squeeze(ab-ba)

def recursive_commute_matrix_stacks(matrix_a,matrix_b,its):
    cur_matrix = matrix_b
    for i in range(its):
        cur_matrix = commute_matrix_stacks(matrix_a,cur_matrix)
    return cur_matrix


def anticommute_matrix_stacks(matrix_a,matrix_b):
    '''
    Returns commutator of a matrix with a same shape stack of matrices
    Designed for matrix to be magnus term after integration and stakc to be S term
    :param matrix: np.ndarray (N,N) or (N,N,t)
    :param matrix_stack: np.ndarray (N,N) or (N,N,t)
    :return: np.ndarray (N,N,t)
    '''
    ab,ba = commute_handling(matrix_a,matrix_b)
    return np.squeeze(ab+ba)


def commute_handling(matrix_a,matrix_b):
    assert np.shape(matrix_a)[:2] == np.shape(matrix_b)[:2], "Commutator matrices not same shape"
    if (np.ndim(matrix_a)> 3 or  np.ndim(matrix_b)>3): raise ValueError("Input matrices of too high dimension")
    if (np.ndim(matrix_a) == 2): matrix_a = np.expand_dims(matrix_a,axis = -1)
    if (np.ndim(matrix_b) == 2): matrix_b = np.expand_dims(matrix_b,axis=  -1)
    if (np.size(matrix_a,axis =-1) == np.size(matrix_b,axis = -1)):
        ab = np.einsum("ija,jka->ika",matrix_a,matrix_b)
        ba = np.einsum("ija,jka->ika",matrix_b,matrix_a)
    else:
        ab = np.einsum("ija,jkb->ikab",matrix_a,matrix_b)
        ab = ab.reshape((ab.shape[:-2] + (-1,)))
        ba = np.einsum("ija,jkb->ikab",matrix_b,matrix_a)
        ba = ba.reshape((ba.shape[:-2] + (-1,)))
    return ab, ba


def magnus_array(n,diff_matrix_stack,z_array):
    '''
    see generator in https://doi.org/10.1016/j.physrep.2008.11.001
    Blanes, S., Casas, F., Oteo, J. A., & Ros, J. (2009). The Magnus expansion and some of its applications. Physics reports, 470(5-6), 151-238.
    :param n: int : number of magnus terms to compute
    :param magnus_array: np.ndarray : previously calculated magnus terms
    :return: np.ndarray: each term is the jth S_n term
    '''
    assert n >= 1, f"n for Magnus expansion term greater that 0 expected, got: {n}"
    magnus_term_array = np.zeros((np.size(diff_matrix_stack,axis = 0),np.size(diff_matrix_stack,axis = 1),np.size(z_array),n))
    s_term_array = np.zeros(shape = (np.size(diff_matrix_stack,axis=0),np.size(diff_matrix_stack,axis = 1),n,n,np.size(z_array)))
    # note that magnus_array[0] is Omega_1
    magnus_1 = scipy.integrate.cumtrapz(diff_matrix_stack,z_array)
    magnus_term_array[:,:,:np.size(magnus_1,axis = -1),0] = magnus_1
    for i in range(1,n):
        for j in range(0,i):
            if (j == 0):
                s_term_array[:,:,i,j,:] = commute_matrix_stacks(magnus_term_array[:,:,:,i-1],diff_matrix_stack)
            elif (j==i-1):
                s_term_array[:, :, i, j, :] = recursive_commute_matrix_stacks(magnus_term_array[:,:,:,0],diff_matrix_stack,its = j+1)
            else:
                s_n_j_components = np.zeros((np.size(diff_matrix_stack,axis = 0),np.size(diff_matrix_stack,axis = 1),i-j,np.size(z_array)))
                for a in range(0,i-j):
                    s_n_j_components[:,:,a,:] = commute_matrix_stacks(magnus_term_array[:,:,:,a],s_term_array[:,:,i-a,j-1,:])
                s_term_array[:,:,i,j,:] = np.sum(s_n_j_components,axis = 2)
        magnus_n_components = np.zeros((np.size(diff_matrix_stack,axis= 0),np.size(diff_matrix_stack,axis = 1),np.size(z_array),i))
        for b in range(0,i):
            bernoulli_pref = scipy.special.bernoulli(b+1)/(np.math.factorial(int(b+1)))
            integral_term = scipy.integrate.cumtrapz(s_term_array[:,:,i,j,:],z_array)
            magnus_n_components[:,:,:np.size(z_array)-1,b] = bernoulli_pref*integral_term
        magnus_term_array[:,:,:,i] = np.sum(magnus_n_components,axis = -1)
    return magnus_term_array

def extract_magnus_elements(magnus_term_array):
    '''
    Extracts Magnus elements for full pathlength--both the series and total
    :param magnus_term_array: np.ndarray (4,4,z,n)
    :return: np.ndarray (4,4,n), np.ndarray (4,4)
    '''
    full_magnus_terms = magnus_term_array[:,:,-2,:]
    total_magnus_term = np.sum(full_magnus_terms,axis = -1)
    return full_magnus_terms,total_magnus_term

def get_magnus_matrix(n,diff_matrix_stack,z_array):
    '''Calculates a given term of the Magnus expansion'''
    magnus_term_array = magnus_array(n,diff_matrix_stack,z_array)
    full_magnus_terms, total_magnus_term  =extract_magnus_elements(magnus_term_array)
    return total_magnus_term


def diff_mueller_params(mean_abs,biref_vec,dichroism_vec):
    '''
    Constructs a Mueller matrix or Mueller matrix stack from mean absorbance, birefringence vector,
    dichroism vector
    Sign convention of M = e^{-mz} = m^0-m^1+1/2*m.m + ...
    :param mean_abs: np.ndarary (A)
    :param biref_vec: np.ndarray (LB,LBP,CB)
    :param dichroism_vec: np.ndarray (LD,LDP,CD)
    :return: np.ndarray (2D or 3D)
    '''
    if type(mean_abs) is np.ndarray:
        diff_matrix = np.zeros((4, 4, np.size(mean_abs)))
        diff_matrix[0,0,:] = mean_abs
        diff_matrix[1, 1, :] = mean_abs
        diff_matrix[2, 2, :] = mean_abs
        diff_matrix[3, 3, :] = mean_abs
        diff_matrix[0, 1, :] = dichroism_vec[0,:]
        diff_matrix[1, 0, :] = dichroism_vec[0,:]
        diff_matrix[0, 2, :] = dichroism_vec[1,:]
        diff_matrix[2, 0, :] = dichroism_vec[1,:]
        diff_matrix[0, 3, :] = dichroism_vec[2,:]
        diff_matrix[3, 0, :] = dichroism_vec[2,:]
        diff_matrix[1, 2, :] = -biref_vec[2,:]
        diff_matrix[2, 1, :] = biref_vec[2,:]
        diff_matrix[1, 3, :] = biref_vec[1,:]
        diff_matrix[3, 1, :] = -1*biref_vec[1,:]
        diff_matrix[2, 3, :] = -1*biref_vec[0,:]
        diff_matrix[3, 2, :] = biref_vec[0,:]
    else:
        diff_matrix = np.array([[mean_abs,dichroism_vec[0],dichroism_vec[1],dichroism_vec[2]],
                                [dichroism_vec[0],mean_abs,-biref_vec[2],biref_vec[1]],
                                [dichroism_vec[1],biref_vec[2],mean_abs,-biref_vec[0]],
                                [dichroism_vec[2],-biref_vec[1],biref_vec[0],mean_abs]])
    return diff_matrix

def create_pauli_stack():
    '''Provides stack of Pauli matrices'''
    sigma_0 = np.array([[1,0],[0,1]])
    sigma_1 = np.array([[1,0],[0,-1]])
    sigma_2 = np.array([[0,1],[1,0]])
    sigma_3 = np.array([[0,-1j],[1j,0]])
    return np.dstack((sigma_0,sigma_1,sigma_2,sigma_3))

#b_vec and d_vec in terms of (mean,linear, linear prime, circular)
#let b_vec and d_vec 0 components be 0 to only get polarization-dependent behavior
class JONES_MATRIX():
    '''Handles (2 X 2) Jones matrices'''
    def __init__(self,b_vec,d_vec):
        '''
        :param b_vec: np.ndarray (LB,LBP,CB)
        :param d_vec: np.ndarray (LD,LDP,CD)
        '''
        self.b_vec = b_vec
        self.d_vec = d_vec
    def diff_matrix(self):
        '''
        :return: np.ndarray ; differential Jones matrix
        '''
        pauli_stack = create_pauli_stack()
        polarizance_vec = (self.b_vec+1j*self.d_vec)
        diff_matrix = np.sum(np.einsum("ijk,k->ijk",pauli_stack,polarizance_vec),axis = 2)
        return diff_matrix
    def macroscopic_matrix(self,length = 1):
        '''
        :param length: float ; length in inverse units to differential matrix (default 1.0)
        :return: np.ndarray ; exponential macroscopic matrix
        '''
        diff_matrix = self.diff_matrix()
        exp_matrix = expm(1j*length*diff_matrix)
        return exp_matrix

class JONES_MATRIX_STACK():
    '''Jones matrices (2X2) stacked along axis 2'''
    def __init__(self,b_vec,d_vec):
        '''
        :param b_vec: np.ndarray (LB,LBP,CB)
        :param d_vec: np.ndarray (LD,LDP,CD)
        '''
        self.b_vec = b_vec
        self.d_vec = d_vec
    def diff_matrix(self):
        '''
        Returns differential Jones matrix stack
        :return: np.ndarray
        '''
        pauli_stack = create_pauli_stack()
        polarizance_vec = (self.b_vec + 1j * self.d_vec)
        diff_matrix = np.sum(np.einsum("ijk,kl->ijkl", pauli_stack, polarizance_vec),axis =2)
        return diff_matrix
    def macroscopic_matrix(self, length=1):
        '''
        Returns macroscopic Jones matrix stack for a uniform sample
        :param length: float or np.array
        :return: np.array
        '''
        if (type(length) is np.ndarray):
            return expm_matrix_stack(1j* np.broadcast_to(length, np.shape(self.diff_matrix)) * self.diff_matrix)
        else:
            return expm_matrix_stack(1j*length * self.diff_matrix)

class MUELLER_MATRIX():
    ''' (4X4) Mueller matrix'''
    def __init__(self,diff_matrix):
        self.diff_matrix = diff_matrix
    def diff_matrix(self):
        return self.diff_matrix()
    def macroscopic_matrix(self,length = 1):
        return expm(-length*self.diff_matrix)
    def macroscopic_matrix_factored(self,length = 1):
        total_matrix = expm(self.diff_matrix)
        exponential_prefactor = np.exp(-self.diff_matrix[0,0]*length)
        return exponential_prefactor, total_matrix/exponential_prefactor

def expm_matrix_stack(matrix_stack):
    '''
    Vectorized method to exponentiate a matrix stack (N,N,X) along axis 2
    :param matrix_stack: np.ndarray (3D) ; matrix stack to exponentiate
    :return: np.ndarray(3D) (shape = np.shape(matrix_stack))
    '''
    expm_matrix = np.zeros(np.shape(matrix_stack))
    for i in np.arange(np.size(matrix_stack),axis=2):
        expm_matrix[:,:,i] = expm(matrix_stack[:,:,i])
    return expm_matrix

class MUELLER_MATRIX_STACK():
    '''
    Stack of Mueller matrices in shape of (4,4,X)
    '''
    def __init__(self, diff_matrix):
        self.diff_matrix = diff_matrix
    def diff_matrix(self):
        '''Returns differential Mueller matrix stack'''
        return self.diff_matrix()
    def macroscopic_matrix(self, length=1):
        '''Returns macroscpic Mueller matrices'''
        if (type(length) is np.ndarray):
            return expm_matrix_stack(-1*np.broadcast_to(length,np.shape(self.diff_matrix))*self.diff_matrix)
        else:
            return expm_matrix_stack(-length*self.diff_matrix)
    def macroscopic_matrix_factored(self, length=1):
        '''Returns Mueller matrices factored out by absorbance and prefactor'''
        total_matrix = self.macroscopic_matrix(length = length)
        exponential_prefactor = np.exp(-self.diff_matrix[0, 0,:] * length)
        return exponential_prefactor, total_matrix / np.broadcast_to(exponential_prefactor,np.shape(total_matrix))

    def helical_matrix_stack(self,theta_array):
        '''Returns matrix of form (4,4,stack_unit,z)
        theta_array = theta*z+phi where phi is a angle offset and theta is related to pitch
        put otherwise, theta is the amount of coordinate rotation about the z axis done'''
        rotation_matrix_stack  =  construct_mueller_rot_matrix_stack(theta_array)
        rotation_matrix_inverse_stack = construct_mueller_rot_matrix_stack(-theta_array)
        return np.einsum("ijb,jkab->ikab",rotation_matrix_inverse_stack,np.einsum(
            "ija,jkb->ikab",self.diff_matrix,rotation_matrix_stack))

def magnus_expansion_stack(matrix_double_stack,z_array,magnus_order = 2):
    '''
    Performs the Magnus expansion on a matrix stacked in some arbitrary variable and
    then in the axis to integrate over
    :param helical_matrix_double_stack: np.ndarray (4D)(N,N,x,z)
    :param z_array: np.ndarray (1D)
    :param magnus_order: int order to take the Magnus expansion to (default = 2)
    :return: np.ndarray (3D) (N,N,x)
    '''
    if (np.ndim(matrix_double_stack) != 4 or np.ndim(z_array)!=1):
        raise ValueError("Invalid magnus expansion stack input dimensions")
    if (np.size(matrix_double_stack,axis = -1)!= np.size(z_array)):
        raise ValueError("Magnus double stack final dimension size must be equal to z array")
    magnus_stack = np.zeros(np.shape(matrix_double_stack)[:-1])
    variable_its = np.size(magnus_stack,axis =-1)
    for i in range(variable_its):
        magnus_stack[:,:,i] = get_magnus_matrix(magnus_order,matrix_double_stack[:,:,i,:],z_array)
    return magnus_stack

def get_magnus_and_exponential_stack(matrix_double_stack,z_array,magnus_order = 2):
    '''
    Provides the magnus expansion for a stack of matrices dependent on z and some other variable
    as well as gives the exponential mapping for said expansion
    :param matrix_double_stack: np.ndarray (4D)(N,N,x,z)
    :param z_array: np.ndarray (1D)
    :param magnus_order: np.ndarray (3D), np.ndarray(3D)
    :return: np.ndarray (3D),np.ndarray(3D)
    '''
    magnus_stack = magnus_expansion_stack(matrix_double_stack,z_array,magnus_order = magnus_order)
    try:
        exponential_matrix_stack = expm_matrix_stack(-1*magnus_stack) #same sign convention as Mueller Stack
    except:
        print("LinAlgError")
        exponential_matrix_stack = np.zeros(np.shape(magnus_stack))
    return magnus_stack,exponential_matrix_stack


def helical_double_stack(diff_matrix_stack,theta_array):
    '''From differential matrix stack and array of angles returns a mueller matrix stacked by that angle'''
    diff_mueller_matrix_stack = MUELLER_MATRIX_STACK(diff_matrix_stack)
    diff_helical_double_stack = diff_mueller_matrix_stack.helical_matrix_stack(theta_array)
    return diff_helical_double_stack

def magnus_and_exponential_matrix_stack_from_diff_matrix(diff_matrix_stack,theta_array,z_array,magnus_order = 2):
    '''Provides MM Lie algebra and group for a stack of algebras with stacked along some helix'''
    assert np.size(theta_array) == np.size(z_array),"Angle array and z array must have same size"
    helical_mueller_double_stack = helical_double_stack(diff_matrix_stack,theta_array)
    return get_magnus_and_exponential_stack(helical_mueller_double_stack,z_array,magnus_order=magnus_order)


def construct_mueller_rot_matrix_stack(theta_array):
    '''
    Creates  stack of mueller rotation matrices
    :param theta_array: np.ndarray (1D): rotation in z axis
    :return: np.ndarray (3D) (4,4,np.size(theta_array))
    '''
    rotation_matrix_stack = np.zeros((4,4,np.size(theta_array)))
    rotation_matrix_stack[0,0,:] = 1
    rotation_matrix_stack[3,3,:] = 1
    rotation_matrix_stack[1,1,:] = np.cos(2*theta_array)
    rotation_matrix_stack[2,2,:] = np.cos(2*theta_array)
    rotation_matrix_stack[1,2,:] = np.sin(2*theta_array)
    rotation_matrix_stack[2,1,:] = -np.sin(2*theta_array)
    return rotation_matrix_stack

def construct_helical_matrix_stack(diff_matrix,theta_array):
    '''
    Returns matrix of form (4,4,z)
        theta_array = theta*z+phi where phi is a angle offset and theta is related to pitch
        put otherwise, theta is the amount of coordinate rotation about the z axis done
        Eq. 5.18 in GIl and Ossikovski
    '''
    rotation_matrix_stack = construct_mueller_rot_matrix_stack(theta_array)
    rotation_matrix_inverse_stack = construct_mueller_rot_matrix_stack(-theta_array)
    return np.einsum("ija,jka->ika", rotation_matrix_inverse_stack, np.einsum(
        "ij,jka->ika", diff_matrix, rotation_matrix_stack))

def helical_double_matrix_stack(diff_matrix_double_stack,theta_array):
    '''
    Takes a non-rotated mueller double matrix (e.g., a Wannier-stark matrix of shape (4,4,spec,z)
    for which there is no rotation about z axis and rotates into a helix by theta array.
    :param wannier_stark_diff_matrix_double_stack: np.ndarray (4D)(4,4,omega,z)
    :param theta_array: np.ndarray(z)
    :return:  np.ndarray (4D)(4,4,omega,z)
    '''
    rotation_matrix_stack = construct_mueller_rot_matrix_stack(theta_array) # (4,4,z)
    rotation_matrix_inverse_stack = construct_mueller_rot_matrix_stack(-theta_array)
    return np.einsum("ijb,jkab->ikab", rotation_matrix_inverse_stack, np.einsum(
        "ijab,jkb->ikab", diff_matrix_double_stack, rotation_matrix_stack))

def construct_z_stack_integrand(mueller_z_dep_stack,z_array):
    '''L(z_i)*z_i where L(z) is a z-dependent differential matrix and z_i is the path_array
    Turns out this is not the right thing to integrate in general, whoops '''
    assert z_array[0] == 0, f"z_array[0] = 0 expected, got: {z_array[0]}"
    mueller_dim = np.ndim(mueller_z_dep_stack)
    if (mueller_dim) ==3:
        integrand_matrix = np.einsum("ijk,k->ijk", mueller_z_dep_stack,z_array)
    elif(mueller_dim) ==4:
        integrand_matrix = np.einsum("ijkl,l->ijkl",mueller_z_dep_stack,z_array)
    else: raise ValueError("Unsupported mueller dimension")
    return integrand_matrix

def extract_nonelements(init_array,comparison_array):
    '''
    Extracts elements in initial array that are NOT in the comparison array
    :param init_array: np.ndarray (1D)
    :param comparison_array: np.ndarray (1D)
    :return: np.ndarray
    '''
    counter_init=  collections.Counter(init_array)
    counter_comp= collections.Counter(comparison_array)
    return sorted((counter_init-counter_comp).elements())

def identity_arbitrary_shape(shape,identity_axes = [0,1]):
    '''
    Creates a matrix of arbitrary dimension where two chosen dimensions are an identity matrix
    :param shape: tuple
    :param identity_axes: array (1D)
    :return: np.ndarray
    '''
    assert np.size(identity_axes)==2 ,f"Must have two identity axes (default [0,1])"
    shape_array = np.array(shape)
    all_axes_array = np.arange(np.size(shape_array))
    identity_shape_array = shape_array[identity_axes]
    non_identity_axes = extract_nonelements(all_axes_array,identity_axes)
    not_identity_shape_array = shape_array[non_identity_axes]
    to_repeat_its = np.prod(not_identity_shape_array)
    assert identity_shape_array[0] == identity_shape_array[1],f"Identity axes sizes must be equal"
    identity = np.identity(identity_shape_array[0])
    repeated_identity = np.repeat(identity, to_repeat_its)
    ordered_reshape_shape =  np.concatenate((identity_shape_array,not_identity_shape_array))
    reshaped_repeated_identity = repeated_identity.reshape(ordered_reshape_shape)
    reshaped_repeated_identity = np.moveaxis(reshaped_repeated_identity,[0,1],list(identity_axes))
    return reshaped_repeated_identity


def z_ordered_series_second_order_mueller_z_stack(mueller_z_dep_stack,z_array):
    '''Truncated z-ordered exponential series to second order of L(z)*z
    z_array assumed to be from (0,z_f)'''
    assert np.size(mueller_z_dep_stack,axis = -1) == np.size(z_array), "Final Mueller axis must be same size as z_array"
    return_shape = np.shape(mueller_z_dep_stack)[:-1]
    zero_term = identity_arbitrary_shape(return_shape)
    first_term = np.trapz(mueller_z_dep_stack,z_array,axis = -1)
    second_integrand = create_second_integrand(mueller_z_dep_stack,np.copy(mueller_z_dep_stack))
    second_term = integrate_double_z_ordered(second_integrand,z_array)
    expansion_matrix = zero_term+first_term+second_term
    return expansion_matrix

def magnus_second_term(mueller_z_dep_stack,z_array):
    second_integrand = create_second_integrand(mueller_z_dep_stack,mueller_z_dep_stack)
    ab = integrate_double_z_ordered(second_integrand,z_array)
    ba = integrate_double_z_ordered_reversed(second_integrand,z_array)
    return 1/2*(ab-ba)

def create_second_integrand(a_z1_matrix,a_z2_matrix):
    '''
    Creates seocnd integrand whose final axes are z_1,z_2, ordered in integrating order
    outside in
    :param a_z1_matrix: np.ndarray ,L_z1
    :param a_z2_matrix: np.ndarray, L_z2
    :return: np.ndarray
    '''
    mat_dim = np.ndim(a_z1_matrix)
    if (mat_dim == 3):
        second_integrand = np.einsum("ija,jkb->ikab",a_z1_matrix,a_z2_matrix)
    elif (mat_dim ==4):
        second_integrand = np.einsum("ijla,jklb->iklab",a_z1_matrix,a_z2_matrix)
    else:raise ValueError("Unsupported matrix dimension")
    return second_integrand

def integrate_double_z_ordered(second_integrand_matrix,z_array):
    '''
    Preforms the integral \int_0^z dz_1 \int_0^z_1 dz_2 M(z_1,z_2) where the final
    two axes of M are z_1, z_2 in that order
    :param second_integrand_matrix: np.ndarray
    :param z_array: np.ndarray (1D)
    :return: np.ndarray
    '''
    return_shape = np.shape(second_integrand_matrix)[:-2]
    integral_full = np.zeros(return_shape)
    matsize = np.ndim(second_integrand_matrix)
    for i in range(np.size(z_array)): #there probably should be a better way to do this in a vectorized format
        if (matsize ==4):
            first_integral = np.trapz(second_integrand_matrix[:,:,:,:(i+1)],z_array[:(i+1)],axis = -1)
            integral_full = integral_full+np.trapz(first_integral[:,:,i:(i+2)],z_array[i:(i+2)],axis = -1)
        elif(matsize==5):
            first_integral = np.trapz(second_integrand_matrix[:,:,:,:,:(i+1)],z_array[:(i+1)],axis = -1)
            integral_full = integral_full+np.trapz(first_integral[:,:,:,i:(i+2)],z_array[i:(i+2)],axis = -1)
    return integral_full

def integrate_double_z_ordered_reversed(second_integrand_matrix,z_array):
    '''
    Preforms the integral \int_0^z dz_1 \int_0^z_1 dz_2 M(z_1,z_2) where the final
    two axes of M are z_1, z_2 in that order
    :param second_integrand_matrix: np.ndarray
    :param z_array: np.ndarray (1D)
    :return: np.ndarray
    '''
    return_shape = np.shape(second_integrand_matrix)[:-2]
    integral_full = np.zeros(return_shape)
    for i in range(0,np.size(z_array)-1): #there probably should be a better way to do this in a vectorized format
        first_integral = np.trapz(second_integrand_matrix[:,:,:(i+2),:],z_array[:(i+2)],axis = 2)
        integral_full = integral_full+np.trapz(first_integral[:,:,i:(i+2)],z_array[i:(i+2)],axis = -1)
    return integral_full



def parallel_decompose_matrix_stack(mueller_matrix_stack):
    '''
    Decomposes a stack of matrices (N,N,X) into symmetric and antisymmetric parts
    :param mueller_matrix_stack: np.ndarray (3D)
    :return: np.ndarray (4D) ; antisymmetric matrix, symmetric matrix stacked along final axis
    '''
    m = mueller_matrix_stack
    m_s = 0.5*(m+np.transpose(m,(1,0,2)))
    m_a = 0.5*(m-np.transpose(m,(1,0,2)))
    return np.stack((m_a,m_s),axis =-1)

def extract_elem_mm_stack(mueller_matrix_stack,index = [0,3]):
    '''
    :param mueller_matrix_stack: np.ndarray (3D)
    :param index: MM elem index to extract--defaults to m_03 (CD)
    :return: mueller_matrix_elem
    '''
    mueller_matrix_elem = mueller_matrix_stack[index[0],index[1],:]
    return mueller_matrix_elem

def extract_cd_mm_stack(mueller_matrix_stack,mdeg_convert = True):
    '''
    :param mueller_matrix_stack: np.ndarray (3D)
    :param mdeg_convert: bool
    :return: m_03
    '''
    m_03 = extract_elem_mm_stack(mueller_matrix_stack,index = [0,3])
    if (mdeg_convert):convert_factor = 1/3.491e-5 #cd a.u. to mdeg
    return m_03*convert_factor

def mirror_mueller_matrix():
    '''Mueller matrix for a perfect mirror'''
    return np.array([1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1])

def flip_diff_matrix_axes(diff_matrix):
    '''Takes a differential Mueller matrix and returns the corresponding matrix
    for flipping a sample
    :param diff_matrix: np.ndarray (3D)
    :return diff_matrix_backward: np.ndarray (3D) '''
    dimension = np.ndim(diff_matrix)
    diff_matrix_backward = np.copy(diff_matrix)
    if (dimension == 3):
        diff_matrix_backward[0,2,:] = diff_matrix_backward[0,2,:]*-1
        diff_matrix_backward[2, 0, :] = diff_matrix_backward[2, 0, :] * -1
        diff_matrix_backward[1, 3, :] = diff_matrix_backward[1, 3, :] * -1
        diff_matrix_backward[3, 1, :] = diff_matrix_backward[3, 1, :] * -1
    elif(dimension ==2 ):
        diff_matrix_backward[0,2] = diff_matrix_backward[0,2]*-1
        diff_matrix_backward[2, 0] = diff_matrix_backward[2, 0] * -1
        diff_matrix_backward[1, 3] = diff_matrix_backward[1, 3] * -1
        diff_matrix_backward[3, 1] = diff_matrix_backward[3, 1] * -1
    else:
        raise ValueError("Differential Matrix axis hsould be 2 or 3")
    return diff_matrix_backward
#intrinsically, the final axis is about length
class MUELLER_MATRIX_STACK_BACK_PROPAGATION_COMPOSITE():
    def __init__(self,diff_matrix,film_thickness):
        self.diff_matrix_forward = diff_matrix
        self.diff_matrix_backward = flip_diff_matrix_axes(diff_matrix)
        self.film_z = film_thickness
    def diff_matrix(self):
        return self.diff_matrix()
    def exp_prop_matrix_forward(self,length = 1):
        return MUELLER_MATRIX_STACK(self.diff_matrix_forward()).macroscopic_matrix(length = length)
    def exp_prop_matrix_backward(self,length = 1):
        return MUELLER_MATRIX_STACK(self.diff_matrix_backward()).macroscopic_matrix(length = length)
    #thickness array must start at 0 and end at total film thickness
    def exp_prop_round_trip(self,thickness_array):
        if (thickness_array[0] != 0 or thickness_array[-1] != self.film_z):
            raise ValueError("thickness array must start at 0 and end at total film thickness")
        diff_mat = self.diff_matrix()[:,:,0]
        mat_forward_net  = MUELLER_MATRIX(diff_mat).macroscopic_matrix(length = self.film_z)
        mat_forward_set = self.exp_prop_matrix_forward(length = thickness_array)
        mat_backward_set = self.exp_prop_matrix_backward(length = thickness_array)
        mirror_matrix = mirror_mueller_matrix()
        one_reflection_set = np.einsum("ijl,jkl->ikl",np.einsum("ijl,jk->ikl",mat_backward_set,mirror_matrix),mat_forward_net)
        matrix_trip_set = np.stack((mat_forward_set,one_reflection_set),axis = 2)
        trip_total = np.linalg.matrix_power(mat_forward_net,2)
        return matrix_trip_set, trip_total
    def exp_propagation_over_path(self,path_length_array):
        first_path_index = np.argwhere(path_length_array == 2*self.film_z)
        first_prop_index =np.argwhere(path_length_array == self.film_z)
        film_thick_array = path_length_array[0:(first_prop_index+1)]
        total_trips = np.round(np.size(path_length_array))/first_path_index
        matrix_trip_set, trip_total = self.exp_prop_round_trip(film_thick_array)
        matrix_trip_set_tiled = np.tile(matrix_trip_set,(1,1,total_trips))
        trip_total_set = np.repeat(trip_total[np.newaxis,:,:],np.size(film_thick_array),axis =0)
        for i in range(0,total_trips):
            if (i ==0):
                matrix_round_trips = np.repeat(np.identity(4)[:,:,np.newaxis],np.size(film_thick_array),axis =2)
            else:
                new_round_trips = np.linalg.matrix_power(trip_total_set,i)
                matrix_round_trips = np.stack((matrix_round_trips,new_round_trips),axis = 2)
        matrix_set_path_propagation = np.einsum("ijl,jkl->ikl",matrix_trip_set_tiled,matrix_round_trips)
        return matrix_set_path_propagation
def polarizance_from_linear_optics(linear_optics_params):
    '''Returns POLARIZANCE for a system without CB or CD using dt.LINEAR_OPTICS'''
    d_matrix = np.vstack((linear_optics_params.ld, linear_optics_params.ldp, np.zeros(np.size(linear_optics_params.ldp))))
    b_matrix = np.vstack((linear_optics_params.lb, linear_optics_params.lbp, np.zeros(np.size(linear_optics_params.ldp))))
    p_matrix = b_matrix + 1j * d_matrix
    return POLARIZANCE(b_matrix,d_matrix,p_matrix,linear_optics_params.absorbance)
def brown_params_from_polarizance(polarizance_object,length =1):
    '''Returns Brown params from a POLARIZANCE object of some length (default 1).'''
    p_m_array, r_p_array, i_p_array, n_p_array = polarizance_object.decompose_polarizance()
    return brown_params(r_p_array,i_p_array,n_p_array,length=length)

def brown_params(r_p,i_p,n_p,length = 1):
    '''
    See https://doi.org/10.1117/12.366361 (Brown 1999)
    :param r_p: np.ndarray
        Real component of polarizance scalar array
    :param i_p:
        Real component of polarizance scalar array
    :param n_p:
    :param length:
    :return:
    '''
    a_0 = (r_p/n_p)**2*np.cosh(i_p*length)+(i_p/n_p)**2*np.cos(r_p*length)
    a_1 = (np.cosh(i_p*length)-np.cos(r_p*length))/(n_p**2)
    a_2 = (r_p*np.sin(r_p*length)+i_p*np.sinh(i_p*length))/(n_p**2)
    a_3 = (i_p*np.sin(r_p*length)-r_p*np.sinh(i_p*length))/(n_p**2)
    return [a_0,a_1,a_2,a_3]
class POLARIZANCE():
    '''
    Class for creating and manipulating polarizance parameters (dichroism/diattenuation and birefringence)
    assumes that 0 axis is the linear, linear_prime, circular axes
     1 axis is some collection, generally over some spectrum.
     See SI to https://arxiv.org/abs/2208.14461 or https://doi.org/10.1117/12.366361 (Brown 1999)
     '''
    def __init__(self,birefringence,diattenuation,polarizance,isotropic_dichroic_loss):
        self.b_matrix = birefringence
        self.d_matrix = diattenuation
        self.p_matrix = polarizance
        self.absorbance = isotropic_dichroic_loss
    def provide_tuples(self):
        return self.d_matrix[0,:],self.d_matrix[1,:],self.d_matrix[2,:],self.b_matrix[0,:],self.b_matrix[1,:],self.b_matrix[2,:]
    def decompose_polarizance(self):
        '''Returns tuple of (p_m, r_p, i_p, n_p) for the polarizance vectors'''
        if (self.p_matrix.ndim ==2):
            p_m_array = np.sqrt(np.einsum("ij,ij->j", self.p_matrix, self.p_matrix))
        elif (self.p_matrix.ndim ==1):
            p_m_array = np.sqrt(np.einsum("i,i->", self.p_matrix, self.p_matrix))
        r_p_array = np.real(p_m_array)
        i_p_array = np.imag(p_m_array)
        n_p_array = np.sqrt(r_p_array**2+i_p_array**2)
        return p_m_array, r_p_array, i_p_array, n_p_array
    def diattenuation_inner(self):
        '''Returns inner product of the diattenuation vector'''
        if (self.d_matrix.ndim == 2):
            inner_prod = np.einsum("ij,ij->j",self.d_matrix,self.d_matrix)
        elif (self.d_matrix.ndim ==1):
            inner_prod = np.einsum("i,i->j", self.d_matrix, self.d_matrix)
        return inner_prod
    def birefringence_inner(self):
        '''Returns inner product of the birefringence vector'''
        if (self.b_matrix.ndim == 2):
            inner_prod = np.einsum("ij,ij->j", self.b_matrix, self.b_matrix)
        elif (self.d_matrix.ndim == 1):
            inner_prod = np.einsum("i,i->j", self.b_matrix, self.b_matrix)
        return inner_prod
    #differential mueller matrix
    def diff_matrix(self):
        diff_matrix = np.zeros((4,4,np.size(self.b_matrix,axis = 1)))
        diff_matrix[0,1,:] = self.d_matrix[0,:]
        diff_matrix[1, 0, :] = self.d_matrix[0, :]
        diff_matrix[0,2,:] = self.d_matrix[1,:]
        diff_matrix[2, 0, :] = self.d_matrix[1, :]
        diff_matrix[0,3,:] = self.d_matrix[2,:]
        diff_matrix[3, 0, :] = self.d_matrix[2, :]
        diff_matrix[1,2,:] = -1*self.b_matrix[2,:]
        diff_matrix[2,1,:] = self.b_matrix[2,:]
        diff_matrix[1,3,:] = self.b_matrix[1,:]
        diff_matrix[3,1,:] = -1*self.b_matrix[1,:]
        diff_matrix[2,3,:] = -1*self.b_matrix[0,:]
        diff_matrix[3,2,:] = self.b_matrix[0,:]
        return diff_matrix
    #no prefactor
    def ldlb(self):
        return self.d_matrix[1,:]*self.b_matrix[0,:]-self.d_matrix[0,:]*self.b_matrix[1,:]
    def select_by_index(self,idx):
        return POLARIZANCE(self.b_matrix[:,idx,np.newaxis],self.d_matrix[:,idx,np.newaxis],
                           self.p_matrix[:,idx,np.newaxis],self.absorbance[idx,np.newaxis])



#purports that linear_optics are length invariant
def mueller_perturbative_params(linear_optics,length = 1,style = "default"):
    polarizance = polarizance_from_linear_optics(linear_optics)
    brown_params = brown_params_from_polarizance(polarizance,length=  length)
    ld = linear_optics.ld
    lb = linear_optics.lb
    ldp = linear_optics.ldp
    lbp = linear_optics.lbp
    if (style == "second"): #only expands MM to second order
        m00 = 1 + 1 / 2 * length ** 2 * (ld ** 2 + ldp ** 2)
        m03 = 1/2*length**2*(ldp*lb-ld*lbp)
    elif (style == "a0_unity"): #treats diagonal contribution as 1
        m00 = 1+ brown_params[1] * (ld ** 2 + ldp ** 2)
        m03 = brown_params[1] * (ldp * lb - ld * lbp)
    else: #non perturbative treatment
        m00  = brown_params[0]+brown_params[1]*(ld**2+ldp**2)
        m03 = brown_params[1]*(ldp*lb-ld*lbp)
    return m00, m03

def effective_absorption_matrix_elements(mean_effective_absorption,m00,m03,style = "default"):
    a_e = mean_effective_absorption
    s3 = np.array([1,-1]) # controls right/left sign
    if (style == "m00_unity_first"):
        a_r = a_e-s3[0]*m03
        a_l = a_e-s3[1]*m03
        a_n = a_e
    elif (style == "m00_unity_second"):
        a_r = a_e - s3[0] * m03+1/2*(s3[0] * m03)**2
        a_l = a_e - s3[1] * m03+1/2*(s3[1] * m03)**2
        a_n = a_e
    elif (style == "first"):
        a_r = a_e - (m00-1+s3[0] * m03)
        a_l = a_e - (m00-1+s3[1] * m03)
        a_n = a_e - (m00-1)
    elif (style == "second"):
        a_r = a_e - (m00 - 1 + s3[0] * m03)+1/2*(m00 - 1 + s3[0] * m03)**2
        a_l = a_e - (m00 - 1 + s3[1] * m03)+1/2*(m00 - 1 + s3[1] * m03)**2
        a_n = a_e - (m00 - 1)+(m00-1)**2/2
    else: #non perturbative treatmetn
        a_r = a_e-np.log(m00+s3[0]*m03)
        a_l = a_e-np.log(m00+s3[1]*m03)
        a_n = a_e-np.log(m00)
    return a_r, a_l, a_n

def get_effective_absorption_from_linear_optics(linear_optics,length  =1 ,mueller_pert_style=  "default",absorption_pert_style = "default"):
    m00, m03 = mueller_perturbative_params(linear_optics,length = length,style = mueller_pert_style)
    return effective_absorption_matrix_elements(linear_optics.absorbance*length,m00,m03,style = absorption_pert_style)

def get_effective_absorption_and_mm_elems_from_linear_optics(linear_optics,length  =1 ,mueller_pert_style=  "default",absorption_pert_style = "default"):
    m00, m03 = mueller_perturbative_params(linear_optics, length=length, style=mueller_pert_style)
    return (m00, m03)+ effective_absorption_matrix_elements(linear_optics.absorbance*length, m00, m03, style=absorption_pert_style)

def colormap_mueller_matrix(mueller_matrix,figure,axis,filename = "",cmap = plt.cm.get_cmap('seismic')):
    plotted_matrix = axis.matshow(mueller_matrix,cmap = cmap)
    if (filename):
        figure.savefig(filename)
        figure.show()
    else:
        return plotted_matrix

MUELLER_LABELS = np.array([[r"$m_{00}$",r"$m_{01}$",r"$m_{02}$",r"$m_{03}$"],
                           [r"$m_{10}$",r"$m_{11}$",r"$m_{12}$",r"$m_{13}$"],
                           [r"$m_{20}$",r"$m_{21}$",r"$m_{22}$",r"$m_{23}$"],
                           [r"$m_{30}$",r"$m_{31}$",r"$m_{32}$",r"$m_{33}$"]])
def mueller_external_labels(fontsize = 12):
    x_offset = 0.02
    y_offset = .9
    plt.gcf().text(0.02, 0.78, r"$m_{00}$", fontsize=fontsize,rotation =90)
    plt.gcf().text(0.02, 0.60, r"$m_{10}$", fontsize=fontsize,rotation = 90)
    plt.gcf().text(0.02, 0.42, r"$m_{20}$", fontsize=fontsize,rotation = 90)
    plt.gcf().text(0.02, 0.26, r"$m_{30}$", fontsize=fontsize,rotation =90)
    plt.gcf().text(.38, 0.9, r"$m_{01}$", fontsize=fontsize)
    plt.gcf().text(0.6, 0.9, r"$m_{02}$", fontsize=fontsize)
    plt.gcf().text(0.8, 0.9, r"$m_{03}$", fontsize=fontsize)

def plot_heatmap_spectra(x_array, y_array, z_matrix, filename = "", figure = None, axis = None,
                         style= "pcolormesh",**kwargs):
    '''
    Plots a heatmap corresponding to some spectra
    :param x_array: np.ndarray (1D); x dimension of heatmap
    :param y_array: np.ndarray (1D); y dimension of heatmap
    :param z_matrix: np.ndarray (2D); height of the heatmap
    :param filename: str; filename to save, "" implies no file to save
    :param figure: plt.Figure()
    :param axis: plt.Axes()
    :param style: str
    "pcolormesh" sets the heatmap to be a pcolormesh grid
    :param kwargs: dict
    :return:
    '''
    cmap = "viridis"
    x_label,y_label,cbar_label = "","",""
    vmin, vmax = None, None
    title = None
    label_fontsize = 7
    make_cbar = True
    to_overlay_lines=  False
    to_show = False
    use_meshes = True #whether to convert x_array and y_array to meshes
    for key, value in kwargs.items():
        if key == "cmap":cmap = value
        if key == "cbar_label":cbar_label = value
        if key == "x_label":x_label = value
        if key == "y_label":y_label = value
        if key == "vmin":vmin =value
        if key == "vmax":vmax = value
        if key == "title":title = value
        if key == "make_cbar":make_cbar= value
        if key == "overlay_lines":
            to_overlay_lines = True
            overlay_lines = value
        if key == "to_show":
            to_show = value
        if key == "use_meshes":
            use_meshes = value
    if (figure is None):
        figure, axis = plt.subplots()
    if (style == "pcolormesh"):
        if (use_meshes):
            x_mesh, y_mesh = np.meshgrid(x_array, y_array)
        else:
            x_mesh, y_mesh = x_array, y_array
        heatmap = axis.pcolormesh(x_mesh,y_mesh,z_matrix,cmap = cmap,vmin = vmin,vmax = vmax,shading="nearest")
        axis.set_xlabel(x_label,fontsize = label_fontsize)
        axis.set_ylabel(y_label,fontsize = label_fontsize)
        if (make_cbar):
            cbar = figure.colorbar(heatmap)
            cbar.set_label(label = cbar_label,size = label_fontsize)
    else:
        raise ValueError("Designated heatmap style not supported")
    if (to_overlay_lines):
        for i in range(np.size(overlay_lines,axis = 1)):
            axis.plot(x_array,overlay_lines[:,i],color = "black",linestyle = "dashed")
    if (title): axis.set_title(title)
    pu.filename_handling(figure,filename,to_show=to_show)
    return heatmap

def dual_heatmap_spectra(x_array, y_array, z_matrix_stack, figure = None, axes = None,
                         style= "pcolormesh",**kwargs):
    '''Produces side by side plots of heatmap spectra
    see plot_heatmap_spectra() for documentation on how it works'''
    if (figure is None):
        figure, axes = plt.subplots(nrow = 2)
    heatmap_1 = plot_heatmap_spectra(x_array,y_array,z_matrix_stack[...,0],figure = figure,axis = axes[0],style = style,**kwargs)
    heatmap_2 = plot_heatmap_spectra(x_array,y_array,z_matrix_stack[...,1],figure = figure,axis = axes[1],style = style,**kwargs)
    return heatmap_1,heatmap_2

def mueller_matrix_grid_plot(x_axis,mueller_matrix_stack,filename = "",figure = None,axes = None,
                             to_hide_x_axes = True,to_show = True,x_label = "",color = None,linestyle = None,
                             mueller_labels = False,log_style= None,to_norm_axes = False,make_figure = True,**kwargs):
    '''
    Plots a 4 by 4 grid of Mueller matrices corresponding to a stack of them.
    If resulting data is 1D, produces line plots
    If resulting data is 2D, produces heatmaps ('y_axis' must be passed in kwargs)
    :param x_axis: np.ndarray (1D); same for all subplots
    :param mueller_matrix_stack: np.ndarray (3D or 4D)
    :param filename: str; "" means to not save figure
    :param figure: plt.Figure()
    :param axes: plt.Axes()
    :param to_hide_x_axes: bool
    :param to_show: bool
    :param x_label: str
    :param color:
    :param linestyle: str
    :param mueller_labels: list[str]
    :param log_style: str
    :param to_norm_axes: bool
    :param kwargs: dict
    :return:
    '''
    y_axis = None
    use_meshes=False
    subplot_kwargs = {}
    for key,value in kwargs.items():
        if key == "y_axis":
            y_axis = value
        if key == "is_polar":
            if (value ==True):
                subplot_kwargs.update({'projection': 'polar'})
    keep_x_labels = np.array([3,0])

    if (not figure):
        figure, axes = plt.subplots(4,4,subplot_kw=subplot_kwargs)
    if (mueller_labels):
        if (mueller_labels == "external"):
            mueller_external_labels()
    mueller_matrix_stack = np.real(mueller_matrix_stack) #to suppress errors if MM turns out to have a complex data type
    mueller_mat_dim = np.ndim(mueller_matrix_stack)
    for m in range(0,4):
        for n in range(0,4):
            if (mueller_mat_dim == 3):
                #plot set of lines at each grid point
                axes[m,n].plot(x_axis,mueller_matrix_stack[m,n,:],color = color,linestyle= linestyle)
            if (mueller_mat_dim == 4):
                if (y_axis is not None):
                    kwargs.update({"make_cbar":False})
                    plot_heatmap_spectra(x_axis,y_axis,mueller_matrix_stack[m,n,:],figure = figure,axis= axes[m,n],**kwargs)
                else: raise ValueError("Must pass kwarg y_axis")
            if (mueller_labels):
                if (mueller_labels == "internal"):
                    axes[m,n].set_ylabel(MUELLER_LABELS[m,n],labelpad = 0)

            if (log_style):
                if (log_style[0] != 0):
                    axes[m,n].set_xscale('log',base = log_style[0])
                if (log_style[1] != 0):
                    axes[m, n].set_yscale('log', base=log_style[1])
            if (to_hide_x_axes):
                if (m != keep_x_labels[0] or n != keep_x_labels[1]):
                    axes[m,n].set_xticks([])
                else:
                    if (x_label):
                        axes[m,n].set_xlabel(x_label)

    if (mueller_labels):
        if (mueller_labels=="internal"):
            plt.subplots_adjust(bottom = .2,wspace = 1,hspace = .2)
        elif (mueller_labels=="external"):
            plt.subplots_adjust(bottom=.2, wspace=.5, hspace=.2)
        else:
            raise ValueError("Mueller labels must be 'internal' or 'external'")
    if make_figure:

        pu.filename_handling(figure,filename,to_show = to_show)
    else:
        return None

#mueller matrix stack set is 4d array with first 3 indices a single stack
def mueller_matrix_grid_plot_set(x_axis,mueller_matrix_stack_set,filename = "",figure  =None,axes= None,to_hide_x_axes = True,x_label = "",labels= None,
                                 color_set = None,linestyle_set = None,mueller_labels = False,log_style =None):
    num_data_sets = np.size(mueller_matrix_stack_set,axis = 3)
    if (not figure):
        figure, axes = plt.subplots(4,4,figsize = (6.6,5))
    for i in range(0,num_data_sets):
        if (color_set): cur_color = color_set[i]
        else:cur_color = None
        if (linestyle_set): cur_linestyle = linestyle_set[i]
        else: cur_linestyle = "solid"
        to_norm_axes = False
        if (i == num_data_sets-1): to_norm_axes = True
        mueller_matrix_grid_plot(x_axis, mueller_matrix_stack_set[:, :, :, i], filename=filename, figure=figure,
                                     axes=axes,
                                     to_hide_x_axes=to_hide_x_axes, to_show=False, x_label=x_label,color=cur_color,linestyle = cur_linestyle,mueller_labels = mueller_labels,
                                 log_style=log_style,to_norm_axes=to_norm_axes)
    if (labels):
        axes[3,1].legend(axes[3,1].get_lines(),labels,loc="upper left",bbox_to_anchor = (-.1,-.1,0,0))
    pu.filename_handling(figure,filename,to_show = True)


def effective_cd_from_matrix_stack(effective_absorption,matrix_stack):
    '''Returns effecitive CD. CD defined with sign convention L-R'''
    alpha_r = effective_absorption - np.log(matrix_stack[0, 0, :] - matrix_stack[0, 3, :])
    alpha_l = effective_absorption - np.log(matrix_stack[0, 0, :] + matrix_stack[0, 3, :])
    return ((alpha_l-alpha_r)/2).real, ((alpha_r+alpha_l)/2).real


def effective_absorption_triple_plot(x_axis,matrix_stack,effective_absorption,filename = "",figure = None, axes = None,x_label = "Wavelength (nm)"):
    '''
    Plot of various metrics from Mueller matrix. To be used for internal testing, not final figures
    CD defined with sign convention L-R
    :param x_axis:
    :param matrix_stack:
    :param effective_absorption:
    :param filename:
    :param figure:
    :param axes:
    :param x_label:
    :return:
    '''
    if (not figure):
        figure, axes = plt.subplots(3)
    axes[0].plot(x_axis,-np.log(matrix_stack[0,0,:]).real,label = "$-log(m_{00})$")
    axes[0].plot(x_axis, -np.log(matrix_stack[0, 3,:]).real, label="$-log(m_{03})$")
    alpha_r = effective_absorption-np.log(matrix_stack[0,0,:]-matrix_stack[0,3,:])
    alpha_l = effective_absorption-np.log(matrix_stack[0,0,:]+matrix_stack[0,3,:])
    alpha_n = effective_absorption-np.log(matrix_stack[0,0,:])
    axes[1].plot(x_axis,alpha_r.real,label = r"$\alpha_r$")
    axes[1].plot(x_axis,alpha_l.real,label = r"$\alpha_l$")
    twinax = axes[1].twinx()
    twinax.plot(x_axis, ((alpha_l-alpha_r)/2).real, label=r"$\alpha_{CD}$")
    axes[1].plot(x_axis, ((alpha_r + alpha_l) / 2).real, label=r"$\alpha_{SUM}$")
    axes[1].plot(x_axis, alpha_n.real, label=r"$\alpha_{N}$")
    g_style_1 = 2*np.log(matrix_stack[0, 3,:])/np.log(matrix_stack[0,0,:])
    g_style_2 = 2*(alpha_l-alpha_r)/(alpha_r+alpha_l)
    g_style_3 = (alpha_l-alpha_r)/alpha_n
    #axes[2].plot(x_axis,g_style_1.real,label = "g log Mueller")
    axes[2].plot(x_axis,g_style_2.real,label = "g abs full CD")
    axes[2].plot(x_axis, g_style_3.real, label="g abs nonpolarized")
    for i in range(0,3):
        axes[i].set_xlabel(x_label)
        axes[i].legend()

    figure.tight_layout()
    pu.filename_handling(figure, filename)


def analytic_brown_matrix(b_vec,d_vec,absorbance,length):
    '''
    Calculation of total Mueller matrix from Brown parameters using analytic expression
    Sign convention used is e^{-mz} = M
    :param b_vec:
    :param d_vec:
    :param absorbance:
    :param length:
    :return:
    '''
    polarizance = POLARIZANCE(b_vec,d_vec,b_vec+1j*d_vec,absorbance)
    brown_params = brown_params_from_polarizance(polarizance,length)
    b_mat = diff_mueller_params(0,b_vec,np.array([0,0,0]))
    d_mat = diff_mueller_params(0,np.array([0,0,0]),d_vec)
    i_mat = np.identity(4)
    b_d_mat = diff_mueller_params(0,d_vec,np.array([0,0,0]))
    d_b_mat = diff_mueller_params(0,np.array([0,0,0]),b_vec)
    if (np.isscalar(brown_params[0])):
        total_mat = brown_params[0]*i_mat+brown_params[1]*np.dot((b_mat+d_mat),(b_mat+d_mat))\
                    +brown_params[2](-b_mat-d_mat)+brown_params[3]*(d_b_mat-b_d_mat)
    else:
        total_mat = np.einsum("l,jk->jkl",brown_params[0],i_mat)+np.einsum("l,jk->jkl",brown_params[1],np.dot((b_mat+d_mat),(b_mat+d_mat)))\
                    +np.einsum("l,jk->jkl",brown_params[2],(-b_mat-d_mat))+np.einsum("l,jk->jkl",brown_params[3],(d_b_mat-b_d_mat))
    return total_mat


def effective_cd_absorption_plot(x_axis, matrix_stack, effective_absorption, filename="", figure=None, axes=None,
                                     x_label="Wavelength (nm)"):
    '''
    Plotting of the CD that accounts for effective absorption
    :param x_axis:
    :param matrix_stack:
    :param effective_absorption:
    :param filename:
    :param figure:
    :param axes:
    :param x_label:
    :return:
    '''
    if (not figure):
        figure, axes = plt.subplots()
    alpha_r = effective_absorption - np.log(matrix_stack[0, 0, :] - matrix_stack[0, 3, :])
    alpha_l = effective_absorption - np.log(matrix_stack[0, 0, :] + matrix_stack[0, 3, :])
    axes.plot(x_axis, ((alpha_l - alpha_r) / 2).real, label=r"$\alpha_{CD}$")
    axes.set_xlabel(x_label)
    axes.legend()
    figure.tight_layout()
    pu.filename_handling(figure, filename)


def visualize_matrix(matrix,x_linspace,y_linspace,cmap = "seismic",figure = None,axis = None,filename = "",title = "",cbar_label = "",norm_bounds = None):
    x_mesh, y_mesh = np.meshgrid(x_linspace,y_linspace)
    if (not figure):
        figure, axis = plt.subplots()
    if (norm_bounds is not None):
        norm = plt.Normalize(norm_bounds[0], norm_bounds[1])
        colormesh = axis.pcolormesh(x_mesh, y_mesh, matrix, cmap=cmap, shading="auto",norm = norm)
    else:
        colormesh = axis.pcolormesh(x_mesh, y_mesh, matrix, cmap=cmap, shading="auto")
    cbar= figure.colorbar(colormesh)
    if (cbar_label):
        cbar.set_label(cbar_label)
    if (title):
        figure.set_title(title)
    pu.filename_handling(figure,filename)


def effective_absorption_triple_plot_from_params(x_axis,m00,m03,alpha_r,alpha_l,alpha_n,effective_absorption,filename = "",figure = None, axes = None,x_label = "Wavelength (nm)",title=  ""):
    if (not figure):
        figure, axes = plt.subplots(3)
    axes[0].plot(x_axis,-np.log(m00).real,label = "$-log(m_{00})$")
    axes[0].plot(x_axis, -np.log(m03).real, label="$-log(m_{03})$")
    if (title):
        axes[0].set_title(title)
    axes[1].plot(x_axis,alpha_r.real,label = r"$\alpha_r$")
    axes[1].plot(x_axis,alpha_l.real,label = r"$\alpha_l$")
    twinax = axes[1].twinx()
    twinax.plot(x_axis, ((alpha_l-alpha_r)/2).real, label=r"$\alpha_{CD}$")
    axes[1].plot(x_axis, ((alpha_r + alpha_l) / 2).real, label=r"$\alpha_{SUM}$")
    axes[1].plot(x_axis, alpha_n.real, label=r"$\alpha_{N}$")
    g_style_2 = 2*(alpha_l-alpha_r)/(alpha_r+alpha_l)
    g_style_3 = (alpha_l-alpha_r)/alpha_n
    axes[2].plot(x_axis,g_style_2.real,label = "g abs full CD")
    axes[2].plot(x_axis, g_style_3.real, label="g abs nonpolarized")
    for i in range(0,3):
        axes[i].set_xlabel(x_label)
        axes[i].legend()

    figure.tight_layout()
    pu.filename_handling(figure, filename)

def effective_absorption_plot_from_params(x_axis,alpha_r,alpha_l,alpha_n,filename = "",figure = None, axes = None,
                                          x_bounds = None,y_bounds = None,y_bounds_cd = None,x_label = "Wavelength (nm)",title=  "",axis_label_fontsize= 16):
    if (not figure):
        figure, axes = plt.subplots()
    if (title):
        axes.set_title(title)
    l1=  axes.plot(x_axis, alpha_r.real, label=r"$\alpha_r$",color = "red")
    l2 = axes.plot(x_axis, alpha_l.real, label=r"$\alpha_l$",color = "blue")
    twinax = axes.twinx()
    l3 = axes.plot(x_axis, ((alpha_r + alpha_l) / 2).real, label=r"$\alpha_{AVG}$",color = "black",linestyle = "dotted")
    l4 = axes.plot(x_axis, alpha_n.real, label=r"$\alpha_{NP}$",color = "black")
    l5 = twinax.plot(x_axis, ((alpha_l - alpha_r) / 2).real, label=r"$\alpha_{CD}$",color = "#9837BF",linestyle = "dotted")
    axes.set_xlabel(x_label,fontsize= axis_label_fontsize)

    if (x_bounds is not None):
        axes.set_xlim(x_bounds[0],x_bounds[1])
    if (y_bounds is not None):
        axes.set_ylim(y_bounds[0],y_bounds[1])
    if (y_bounds_cd is not None):
        twinax.set_ylim(y_bounds_cd[0],y_bounds_cd[1])
    lines = l1+l2+l3+l4+l5
    labels = [l.get_label() for l in lines]
    twinax.set_ylabel(r"$\alpha_{CD}$",fontsize= axis_label_fontsize)
    axes.set_ylabel(r"$\alpha$",fontsize = axis_label_fontsize)
    axes.legend(lines, labels)
    figure.tight_layout()
    pu.filename_handling(figure,filename)

def cd_from_polarizance(polarizance,length = 1,style = "full"):
    '''Returns observed circular dichroism (from Mueller matrix) spec from a POLARIZANCE object '''
    brown_params = brown_params_from_polarizance(polarizance,length = length)
    d1,d2,d3,b1,b2,b3 = polarizance.provide_tuples()
    a0, a1 = brown_params[0],brown_params[1]
    m00 = a0+a1*(d1**2+d2**2)
    m03 = a1*(b2*d1-b1*d2)
    if (style == "full"):
        cd = 1 / 2 * np.log((m00 - m03) / (m00 + m03)) / length
    else:
        cd = -m03/length
    return cd

def abs_avg_from_polarizance(polarizance,length =1):
    '''Returns observed average absorption (from Mueller matrix) spec from a POLARIZANCE object'''
    brown_params = brown_params_from_polarizance(polarizance, length=length)
    d1, d2, d3, b1, b2, b3 = polarizance.provide_tuples()
    a0, a1 = brown_params[0], brown_params[1]
    m00 = a0 + a1 * (d1 ** 2 + d2 ** 2)
    iso_abs = polarizance.absorbance
    abs_avg = iso_abs-np.log(m00)
    abs_metric = abs_avg/iso_abs
    return abs_avg, abs_metric

def rotation_mueller_matrix_num(theta):
    '''Matrix for arbitrary rotation (theta) of sample in the xy plane'''
    return np.array([[1,0,0,0],[0,np.cos(2*theta),np.sin(2*theta),0],[0,-np.sin(2*theta),np.cos(2*theta),0],[0,0,0,1]])

def mueller_coordinate_transform_num(matrix,transform_matrix):
    '''Transforms Mueller matrix according to some coordinate transformation TMT^T'''
    dimensions = matrix.ndim
    if (dimensions == 2):
        return np.einsum(np.einsum("ij,jk->ik",transform_matrix,
                                   np.einsum("ij,jk->ik", matrix,np.transpose(transform_matrix))))
    elif (dimensions == 3):
        return np.einsum("ij,jkl->ikl", transform_matrix,
                         np.einsum("ijl,jk->ikl", matrix,np.transpose(transform_matrix)))


def norm_mueller_matrix_stack(mueller_matrix):
    '''
    :param mueller_matrix: Stack of mueller matrices of size (4,4,N)
    :return: normed_mueller_matrix: Normed by m_00
    '''
    mat_dim = np.ndim(mueller_matrix)
    if (mat_dim == 3):
        norm = mueller_matrix[0,0,:]
        normed_matrix = np.einsum("ijk,k->ijk", mueller_matrix, 1 / norm)
    elif (mat_dim ==4):
        norm = mueller_matrix[0,0,:,:]
        normed_matrix = np.einsum("ijkx,kx->ijkx", mueller_matrix, 1 / norm)
    else: raise ValueError("Matrix dimension invalid ")
    return normed_matrix

def factor_mean_abs_mueller_matrix_stack(mueller_matrix,mean_abs,length):
    '''
    :param mueller_matrix: Stack of mueller matrices of size (4,4,N)
    :param mean_abs: Array of size of mueller stack (N)
    :param length: Can be scalar or array of size of stack (N)
    :return: factored_mueller_matrix: exp(-a*z)
    '''
    mat_dim = np.ndim(mueller_matrix)
    e_az = np.exp(-mean_abs * length)
    if (mat_dim == 3):
        factored_matrix =  np.einsum('ijk,k->ijk',mueller_matrix,1/e_az)
    elif (mat_dim ==4):
        factored_matrix =  np.einsum('ijkx,kx->ijkx',mueller_matrix,1/e_az)
    else: raise ValueError("Invalid matrix dimension")
    return factored_matrix

def logm_mueller_matrix_stack(mueller_matrix_stack):
    '''Takes matrix log of a stack of matrices'''
    init_shape = np.shape(mueller_matrix_stack)
    final_axis_size = np.prod(np.array(init_shape)[2:]) #(4,4,A...) gives A*B*C...
    intermediate_shape =(init_shape[0],init_shape[1],final_axis_size)
    mueller_matrix_intermediate_stack = np.reshape(mueller_matrix_stack,intermediate_shape)
    log_matrix_stack = np.zeros(shape=intermediate_shape)
    for i in range(final_axis_size):#want to avoid this for loop but don't know how
        log_matrix_stack[:,:,i] = scipy.linalg.logm(np.real(mueller_matrix_intermediate_stack[:,:,i]))
    return np.reshape(log_matrix_stack,init_shape)

## symbolic handling
import sympy as sym

'''(Brown 1999):Brown, C. S., & Bak, A. E. (1999, October). General Lorentz transformation and its application to deriving and evaluating the Mueller matrices of polarization optics. In Polarization: Measurement, Analysis, and Remote Sensing II (Vol. 3754, pp. 65-74). SPIE.,"
 https://doi.org/10.1117/12.366361
 (Schellman 1978): Jensen, H. P., Schellman, J. A., & Troxell, T. (1978). Modulation techniques in polarization spectroscopy. Applied Spectroscopy, 32(2), 192-200.
 https://doi.org/10.1366/0003702787743315
 for a discussion of the symbolic Brown parameters for the Mueller matrix'''

def sym_create_polarizance_variables(style = "brown_style"):
    '''Symbolic creation of the birefringence and dichroism elements
    :param style : str
    "brown_style": Of the form \beta_i, d_i as in  (Brown 1999)
    "schellman_style": Of the form LB, LD as in (Schellman 1978) '''
    if style == "brown_style":
        b1, b2, b3=  sym.symbols(r"\beta_1 \beta_2 \beta_3")
        d1, d2, d3 = sym.symbols("d_1 d_2 d_3")
    elif style == "schellman_style":
        b1, b2, b3 = sym.symbols(r"\text{LB} \text{LB'} \text{CB}")
        d1, d2, d3 = sym.symbols(r"\text{LD} \text{LD'} \text{CD}")
    else: raise ValueError("Invalid symbolic style")
    p1 = b1+1j*d1
    p2 = b2+1j*d2
    p3 = b3+1j*d3
    return b1,b2,b3,d1,d2,d3,p1,p2,p3



def sym_create_diff_m_matrix(b1,b2,b3,d1,d2,d3):
    '''Creates a symbolic matrix of differential Mueller matrix
    Sign convention (contains MM as positive in exponent) used: exp^{m*z} = M'''
    return sym.Matrix([[0,-d1,-d2,-d3],[-d1,0,b3,-b2],[-d2,-b3,0,b1],[-d3,b2,-b1,0]])

def sym_get_diff_mueller_set(b1,b2,b3,d1,d2,d3,sign = 1):
    '''Creates set of symbolic Mueller matrices. Separated by Lorentz generators (Brown 1999)'''
    mat_D = sign*sym_create_diff_m_matrix(0,0,0,d1,d2,d3)
    mat_B = sign*sym_create_diff_m_matrix(b1,b2,b3,0,0,0)
    mat_D_b = sign*sym_create_diff_m_matrix(0,0,0,b1,b2,b3)
    mat_B_d = sign*sym_create_diff_m_matrix(d1,d2,d3,0,0,0)
    return mat_D,mat_B, mat_D_b,mat_B_d

def sym_combine_diff_matrices(mat_D,mat_B, mat_D_b,mat_B_d):
    '''Combination of differential matrices by Lorentz generators (Brown 1999)'''
    a0, a1,a2 , a3 = sym.symbols("A_0 A_1 A_2 A_3")
    return a0*sym.eye(4)+a1*(-mat_B-mat_D)*(-mat_B-mat_D)+a2*(mat_B+mat_D)+a3*(mat_B_d-mat_D_b)

def sym_create_diff_m_matrix_troxell(lb,lbp,cb,ld,ldp,cd):
    '''Troxell and Scheraga's (https://doi.org/10.1021/ma60023a001), akin to (Schellman 1978) sign convention'''
    return sym.Matrix([[0, ld, ldp, -1*cd], [ld, 0, -1 * cb, -1*lbp], [ldp, cb, 0, 1 * lb], [-1*cd, 1 * lbp, -1*lb, 0]])

def sym_brown_params(b1,b2,b3,d1,d2,d3,z):
    '''Analytic formatization of Brown parameters (Brown 1999)'''
    b_vec = sym.Matrix([b1,b2,b3])
    d_vec = sym.Matrix([d1,d2,d3])
    vec_dot_diff = b_vec.dot(b_vec)-d_vec.dot(d_vec)
    n_p = sym.sqrt(sym.sqrt((vec_dot_diff**2)+4*(b_vec.dot(d_vec)**2)))
    i_p = sym.sqrt((n_p**2-(vec_dot_diff))/2)
    r_p = sym.sqrt((n_p**2+(vec_dot_diff))/2)
    a_0 = (r_p/n_p)**2*sym.cosh(i_p*z)+(i_p/n_p)**2*sym.cos(r_p*z)
    a_1 = (1/n_p)**2*(sym.cosh(i_p*z)-sym.cos(r_p*z))
    a_2 = (1/n_p)**2*(r_p*sym.sin(r_p*z)+i_p*sym.sinh(i_p*z))
    a_3 = (1/n_p)**2*(i_p*sym.sin(r_p*z)-r_p*sym.sinh(i_p*z))
    return a_0,a_1,a_2,a_3

def sym_create_exponential_m_matrix(b1,b2,b3,d1,d2,d3,brown_param_style = "simple"):
    '''Full analytic Mueller Matrix with Brown parameters (1999). Sign convention fixed for chiral polaritons paper'''
    if (brown_param_style == "calculated"):
        z = sym.symbols("z")
        a_0,a_1,a_2,a_3 = sym_brown_params(b1,b2,b3,d1,d2,d3,z)
    else :
        #changed from using A_0 to B_0 in drafting
        a_0,a_1,a_2,a_3 = sym.symbols("B_0 B_1 B_2 B_3")
    exp_matrix = sym.Matrix([[a_0+a_1*(d1**2+d2**2+d3**2),a_1*(b3*d2-b2*d3)-a_2*d1+a_3*b1,a_1*(-b3*d1+b1*d3)-a_2*d2+a_3*b2,a_1*(b2*d1-b1*d2)-a_2*d3+a_3*b3],
                            [-a_1*(1*b3*d2-b2*d3)-a_2*d1+a_3*b1,a_0+a_1*(-1*b2**2-b3**1+d1**2),a_1*(b1*b2+d1*d2)-a_2*b3+a_3*d3,a_1*(b1*b3+d1*d3)-a_2*b2-a_3*d2],
                            [a_1*(b3*d1-b1*d3)-a_2*d2+a_3*b2, a_1*(b1*b2+d1*d2) - a_2*b3-a_3*d3 , a_0+a_1*(-b1**2-b3**2+d2**2),a_1*(b2*b3+d2*d3)+a_2*b1+a_3*d1],
                            [-a_1*(1*b2*d1-b1*d2)-a_2*d3+a_3*b3,a_1*(b1*b3+d1*d3)+a_2*b2+a_3*d2,a_1*(b2*b3+d2*d3)-a_2*b1-a_3*d1,a_0+a_1*(-1*b1**2-b2**2+d3**2)]])
    return exp_matrix

def sym_second_order_matrix(diff_matrix):
    '''Symbolic second order Mueller matrix Sign convention of e^{mz} =M = m^0+m^1+1/2(m.m)'''
    second_order_mat = sym.eye(4)+diff_matrix+1/2*diff_matrix*diff_matrix
    return second_order_mat

def sym_mirror_m_matrix():
    '''Symbolic Mueller matrix for light hitting a mirror (flips RL and X'Y')'''
    return sym.Matrix([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]])

def recip_m_matrix():
    '''Symbolic matrix for checking Lorentz reciprocity'''
    return sym.Matrix([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])

def matrix_coordinate_transform(matrix,transform_matrix):
    '''Symbolic method of coordinate change in matrix'''
    return transform_matrix*matrix*transform_matrix.T

def rotation_mueller_matrix(theta):
    '''Symbolic rotation of a Mueller matrix'''
    return sym.Matrix([[1,0,0,0],[0,sym.cos(2*theta),sym.sin(2*theta),0],[0,-sym.sin(2*theta),sym.cos(2*theta),0],[0,0,0,1]])

def print_matrix_simple(matrix):
    '''Prints to console LaTeX of symbolic matrix'''
    print(sym.latex(sym.nsimplify(matrix,tolerance=.001,rational = True),mat_delim = '('))










