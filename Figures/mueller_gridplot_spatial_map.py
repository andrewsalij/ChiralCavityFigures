import mueller
import matplotlib.pyplot as plt
import numpy as np
import copy
import matplotlib.animation as animation
import time


#FIXED MEANS THAT THE XY VALUES HAVE BEEN REWRITTEN TO THE MESH
#x_mesh,y_mesh = np.meshgrid(np.linspace(5,-5,11),np.linspace(5,-5,11))
mueller_array = np.load("full_mueller_array_fixed.npy")
#mueller_err_array = np.load("full_mueller_err_array.npy")
wvl_array = np.load("mueller_wvl_array.npy")


def sort_mueller_array(mueller_array,idx_to_sort,rebase_top_value=np.array([0.5,0.5])):
    sorted_mueller_array = np.zeros(np.shape(mueller_array))
    for i in np.arange(np.size(mueller_array,axis = 1)):
        for j in np.arange((np.size(mueller_array,axis = 2))):
            mueller_slice = mueller_array[:,i,j,:]
            sorted_xy_args = mueller_slice[:,idx_to_sort].argsort()
            sorted_mueller_array[:,i,j,:] = mueller_slice[sorted_xy_args,:]
    return sorted_mueller_array

def order_mueller_array_at_wvl(mueller_array,n_x =11,n_y=11,wvl_idx = 0):
    '''
    Orders the mueller array by wavelength
    :param mueller_array:
    :param wvl_array:
    :return:
    '''
    mueller_selection = copy.deepcopy(mueller_array[:,:,:,wvl_idx+2].reshape(n_x*n_y,4,4,1))
    xy_array = copy.deepcopy(mueller_array[:,:,:,:2])
    total_array = np.concatenate((xy_array,mueller_selection),axis = -1)
    total_array = sort_mueller_array(total_array,1)
    for i in np.arange(n_y):
        sub_array = total_array[i*n_x:(i+1)*n_x,:,:,:]
        total_array[i*n_x:(i+1)*n_x,:,:,:] = sort_mueller_array(sub_array,0)
    mueller_array = total_array[:,:,:,2]
    xy_matrix = total_array[:,:,:,:2]
    return xy_matrix,mueller_array

def plot_heatmap_at_wvl(wvl_selected,mueller_array=mueller_array,wvl_array = wvl_array,figure=None,axes= None,make_figure = True):
    wvl_idx = np.argmin(np.abs(wvl_array-wvl_selected))
    xy_matrix, mueller_to_plot = order_mueller_array_at_wvl(mueller_array,wvl_idx = wvl_idx)
    #taken directly from xy_matrix--not used due to duplicates
    x_mesh = xy_matrix[:,0,3,0].reshape(11,11)
    y_mesh = xy_matrix[:,0,3,1].reshape(11,11)
    # x_array, y_array = np.linspace(-5,5,11),np.linspace(-5,5,11)
    # x_mesh,y_mesh = np.meshgrid(x_array,y_array)
    mueller_to_plot = np.moveaxis(mueller_to_plot,0,-1).reshape(4,4,11,11)
    kwargs = dict(cmap = plt.get_cmap("seismic"),vmin = -1,vmax = 1,y_axis =y_mesh,make_figure=make_figure,use_meshes = False)
    mueller.mueller_matrix_grid_plot(x_mesh,mueller_to_plot,figure = figure,axes = axes,**kwargs)

plot_heatmap_at_wvl(450)
# plot_heatmap_at_wvl(550)

make_animation = True
anim_its = 5 #number of frames to skip
wvl_animate = wvl_array[::anim_its]
if (make_animation):
    #change to your ffmpeg path
    plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\andre\ffmpeg-6.1-full_build\ffmpeg-6.1-full_build\bin\ffmpeg.exe'
    anim_filename = "mueller_animation.mp4"
    figure, axes = plt.subplots(4,4)
    frames_list= []
    for i in np.arange(np.size(wvl_animate)):
        artist_list =[]
        plot_heatmap_at_wvl(wvl_animate[i],figure=  figure,axes = axes,make_figure=False)
        for a in np.arange(np.size(axes,axis=0)):
            for b in np.arange(np.size(axes,axis=1)):
                artist_list.extend(axes[a,b].get_children())
        wvl_label = figure.text(0.05,0.05,"Wavelength: "+str(wvl_animate[i])+" nm",fontsize= 12)
        artist_list.extend(wvl_label)
        frames_list.append(artist_list)
        time.sleep(1) #to prevent to many requests to ffmpeg
    anim = animation.ArtistAnimation(figure,frames_list, interval=20, blit=True)
    anim.save(anim_filename,fps = 5,writer = "ffmpeg")

