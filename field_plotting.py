import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt


def plot_field_mesh(x_array, y_array, field_mesh, fname = "", x_label="Z (nm)", y_label="X (nm)", figure=None, axis=None,norm = matplotlib.colors.Normalize(),cmap = "inferno"):
    '''

    :param x_array:
    :param y_array:
    :param field_mesh:
    :param x_label:
    :param y_label:
    :param figure:
    :param axis:
    :return:
    '''
    if (figure is None):
        figure, axis = plt.subplots()
    y_mesh,x_mesh = np.meshgrid(y_array,x_array)
    pmesh = axis.pcolormesh(x_mesh,y_mesh,field_mesh,norm= norm,cmap = cmap)
    if (x_label != ""):axis.set_xlabel(x_label)
    if (y_label != ""):axis.set_ylabel(y_label)
    if (fname != ""): figure.savefig(fname)
    if (isinstance(figure,plt.Figure)): figure.show()
    return pmesh
def norm_all_array_to_baseline(array_tuple):
    '''
    Takes a tuple of arrays and norms them to the max absolute value
    :param array_tuple:
    :return:
    '''
    max_value = 0
    for i in np.arange(len(array_tuple)):
        cur_max = np.max(np.abs(array_tuple[i]))
        if (cur_max>max_value):
            max_value= cur_max
    new_array_tuple = tuple()
    for i in np.arange(len(array_tuple)):
        new_array_tuple = new_array_tuple+tuple((array_tuple[i]/max_value),)
    return new_array_tuple

def plot_field_mesh_array(x_array,y_array,field_meshes, fname = "", x_label="Z (nm)", y_label="X (nm)",cbar_label = "", figure=None, axes=None,norm = matplotlib.colors.Normalize(),
                          row_titles = [],col_titles = [],cmap = "inferno"):
    '''
    :param x_array:
    :param y_array:
    :param field_meshes:
    :param fname:
    :param x_label:
    :param y_label:
    :param cbar_label:
    :param figure:
    :param axes:
    :param norm:
    :return:
    '''
    field_mesh_plot_shape = np.array(np.shape(field_meshes)[:2])
    if (figure is None):
        figure,axes = plt.subplots(nrows=field_mesh_plot_shape[0],ncols = field_mesh_plot_shape[1],sharex=True,sharey= True)
    axes_shape = np.array(np.shape(axes))
    if (~np.array_equal(axes_shape,field_mesh_plot_shape)):ValueError("Invalid number of field meshes")

    for i in np.arange(axes_shape[0]):
        for j in np.arange(axes_shape[1]):
            ax = axes[i,j]
            #labels set to "" as they are made for shared axes later
            pmesh = plot_field_mesh(x_array,y_array,field_meshes[i,j,:,:],figure = figure,axis = ax,norm = norm,x_label="",y_label ="",cmap = cmap)
    for i in np.arange(axes_shape[0]):
        axes[i,0].set_ylabel(y_label)
        for j in np.arange(1,axes_shape[1]):
            axes[i,j].get_yaxis().set_visible(False)
        if (i<len(row_titles)):
            axes[i,0].text(-.45,.5,row_titles[i],rotation = "vertical",verticalalignment = "center",transform = axes[i,0].transAxes)
    for j in np.arange(axes_shape[1]):
        axes[-1,j].set_xlabel(x_label)
        for i in np.arange(0,axes_shape[0]-1):
            axes[i,j].get_xaxis().set_visible(False)
        if (j<len(col_titles)):
            axes[0,j].text(.5,1.1,col_titles[j],horizontalalignment = "center",transform = axes[0,j].transAxes)
    cbar = figure.colorbar(pmesh,ax = axes[:,:],location = "top",shrink = .8,anchor =(.4,1.2))
    if (cbar_label != ""): cbar.set_label(cbar_label,labelpad= 2)
    figure.subplots_adjust(left = .15,right = .8,hspace= .1,wspace = .1,top = .75)
    if (fname != ""): figure.savefig(fname)
    if (isinstance(figure,plt.Figure)): figure.show()