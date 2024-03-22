import mueller
import matplotlib.pyplot as plt
import numpy as np
import scipy
import copy
import matplotlib.animation as animation
import time

#FIXED MEANS THAT THE XY VALUES HAVE BEEN REWRITTEN TO THE MESH
#x_mesh,y_mesh = np.meshgrid(np.linspace(5,-5,11),np.linspace(5,-5,11))
mueller_array = np.real(np.load("full_mueller_array_fixed.npy"))
#mueller_err_array = np.load("full_mueller_err_array.npy")
wvl_array = np.load("mueller_wvl_array.npy")

def plot_heatmap_at_wvl(wvl_selected,mueller_array=mueller_array,wvl_array = wvl_array,figure=None,axes= None,make_figure = True,logm_plot = False):
    wvl_idx = np.argmin(np.abs(wvl_array-wvl_selected))
    #taken directly from xy_matrix--not used due to duplicates
    x_array, y_array =mueller_array[:,0,0,0],mueller_array[:,0,0,1]
    mueller_to_plot = mueller_array[:,:,:,wvl_idx+2]
    if (logm_plot):
        for i in np.arange(np.size(mueller_to_plot,axis = 0)):
            mueller_to_plot[i,:,:] = np.nan_to_num(mueller_to_plot[i,:,:],nan = 1) #setting M_00 to 1--it was stored as nan
            mueller_to_plot[i,:,:] = np.real(scipy.linalg.logm(mueller_to_plot[i,:,:]))
    x_mesh,y_mesh = np.reshape(x_array,(11,11)),np.reshape(y_array,(11,11))
    mueller_to_plot = np.moveaxis(mueller_to_plot,0,-1).reshape(4,4,11,11)
    kwargs = dict(cmap = plt.get_cmap("seismic"),vmin = -1,vmax = 1,y_axis =y_mesh,make_figure=make_figure,use_meshes = False,make_cbar = True,cbar_label = "Mueller element")
    mueller.mueller_matrix_grid_plot(x_mesh,mueller_to_plot,figure = figure,axes = axes,**kwargs)



plot_heatmap_at_wvl(450,logm_plot=False)