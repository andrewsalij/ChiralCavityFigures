import numpy as np

mueller_array = np.load("full_mueller_array.npy")

def fix_elements_mueller_array(mueller_array,x_linspace,y_linspace):
    x_mesh_intended, y_mesh_intended = np.meshgrid(x_linspace,y_linspace)
    x_array_intended = x_mesh_intended.flatten()
    y_array_intended = y_mesh_intended.flatten()
    for i in np.arange(np.size(mueller_array,axis=1)):
        for j in np.arange(np.size(mueller_array,axis= 2)):
            mueller_slice =  mueller_array[:,i,j,:]
            mueller_slice[:,0] = x_array_intended
            mueller_slice[:,1] = y_array_intended
    return mueller_array
full_mueller_array_fixed = fix_elements_mueller_array(mueller_array,np.linspace(5,-5,11),np.linspace(5,-5,11))
np.save("full_mueller_array_fixed",full_mueller_array_fixed)

