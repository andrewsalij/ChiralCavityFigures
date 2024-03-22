import numpy as np

'''Handling of full Berreman matrix with magento-optical and permeability tensors
for use via pyllama.'''

'''Equations equivalent to those in
Connie Beaverson Mazur, "Modeling light propagation through liquid crystal cell utilizing a 
4 x 4 matrix technique", U of T Arlington, 1994.
or in Azzam and Bashara's Ellipsometry and Polarized Light

Mazur's sign convention is opposite of pyllama, which means that
elements 0-2 flip sign (3) does not as Mazur uses H_x, which cancels out 
Pyllama convention -H_x. (3,3) must flip sign as it maps -H_x to -H_x

Indexing here starts from 0 as does Python, Mazur's starts from 1. 

Andrew Salij 
'''

def calc_berreman_matrix(dielectric_tensor,magneto_optic_tensor,magneto_optic_prime_tensor,permeability_tensor,K_x):
    '''
    Berreman matrix for an arbitary anisotropic system with electric, magneto-electric, and magnetic effects.
    Note that the dimensional convention here is that the H field is adjusted by impedance to have the same dimension
    as the E field.
    :param dielectric_tensor:
    dimensionless
    :param magneto_optic_tensor:
    dimensionless
    :param magneto_optic_prime_tensor:
    dimensionless
    :param permeability_tensor:
    dimensionless
    :param K_x:
    dimensionless (k_x/k0)
    :return:
    '''
    #changing names to more manageable variables
    eps = dielectric_tensor
    rho = magneto_optic_tensor
    rhop= magneto_optic_prime_tensor
    mu = permeability_tensor

    berre_matrix = np.zeros((4,4),dtype = np.cdouble)
    #notation borrowed from Azzam and Bashara's textbook, pg. 344
    d = 1/(eps[2,2]*mu[2,2]-rho[2,2]*rhop[2,2])
    #all 16 berreman elements are uniquely calculated--reduces to standard Pyllama
    #method when mu = iden(3), rho = rhop = zeros(3,3)
    a1 = (rhop[2,0]*rho[2,2]-eps[2,0]*mu[2,2])/d
    a2 = ((rhop[2,1]-K_x)*rho[2,2]-eps[2,1]*mu[2,2])/d
    a3 = (mu[2,1]*rho[2,2]-rho[2,0]*mu[2,2])/d
    a4 = (mu[2,1]*rho[2,2]-(rho[2,1]+K_x)*mu[2,2])/d
    a5 = (rhop[2,2]*eps[2,0]-eps[2,2]*rhop[2,0])/d
    a6 = (rhop[2,2]*eps[2,1]-(rhop[2,1]-K_x)*eps[2,2])/d
    a7 = (rhop[2,2]*rho[2,1]-eps[2,2]*mu[2,0])/d
    a8 = ((rho[2,1]+K_x)*rhop[2,2]-eps[2,2]*mu[2,1])/d
    berre_matrix[0,0] = rhop[1,0]+(rhop[1,2]+K_x)*a1+mu[1,2]*a5
    berre_matrix[0,1] = mu[1,1]+(rhop[1,2]+K_x)*a4+mu[1,2]*a8
    berre_matrix[0,2] = rhop[1,1]+(rhop[1,2]+K_x)*a2+mu[1,2]*a6
    berre_matrix[0,3] = (mu[1,0]+(rhop[1,2]+K_x)*a3+mu[1,2]*a7)*-1
    berre_matrix[1,0] = eps[0,0]+eps[0,2]*a1+rhop[0,2]*a5
    berre_matrix[1,1] = rho[0,1]+eps[0,2]*a4+rhop[0,2]*a8
    berre_matrix[1,2] = eps[0,1]+eps[0,2]*a2+rhop[0,2]*a6
    berre_matrix[1,3] = (rho[0,0]+eps[0,2]*a3+rhop[0,2]*a7)*-1
    berre_matrix[2,0] = (rhop[0,0]+rhop[0,2]*a1+mu[0,2]*a5)*-1
    berre_matrix[2,1] = (mu[0,1]+rhop[0,2]*a4+mu[0,2]*a8)*-1
    berre_matrix[2,2] = (rhop[0,1]+rhop[0,2]*a2+mu[0,2]*a6)*-1
    berre_matrix[2,3] = mu[0,0]+rhop[0,2]*a3+mu[0,2]*a7
    berre_matrix[3,0] = eps[1,0]+eps[1,2]*a1+(rho[1,2]-K_x)*a5
    berre_matrix[3,1] = rho[1,1]+eps[1,2]*a4+(rho[1,2]-K_x)*a8
    berre_matrix[3,2] = eps[1,1]+eps[1,2]*a2+(rho[1,2]-K_x)*a6
    berre_matrix[3,3] = (rho[1,0]+eps[1,2]*a3+(rho[1,2]-K_x)*a7)*-1
    return berre_matrix