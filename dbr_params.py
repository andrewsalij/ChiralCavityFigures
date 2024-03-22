import numpy as np


'''Parameters relating to the modelling of DBR arrays
'''

def get_finesse_from_reflectivities(r1,r2):
    return -2*np.pi/np.log(r1*r2)



