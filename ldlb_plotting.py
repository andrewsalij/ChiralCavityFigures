import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import mpl_toolkits.axes_grid1.inset_locator as inset_locator
import itertools
import dielectric_tensor as dt
import python_util as pu
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
import numpy.ma as ma

def eV_to_color_hex(vals, bounds, color_map="rainbow"):
    cmap = cm.get_cmap(color_map)
    vals_normed = (vals - bounds[0]) / (bounds[1] - bounds[0])
    rgb_array = cmap(vals_normed)
    num_colors = np.size(rgb_array, axis=0)
    hex_array = []
    for i in range(0, num_colors):
        hex_array.append(colors.rgb2hex(rgb_array[i, :]))
    return hex_array

def plot_dipoles(dipole_matrix, filename="", figure="", axis="",
                 colors_set=["#FF0000", "#0900FF", "#8800FF", "#0900FF"], bounds=np.array([0, 1]), labels=[],label_positions=np.zeros((100, 2)), **kwargs):
    axis_style = "equal"
    label_angle=  False
    for key,value in kwargs.items():
        if key == "axis_style":
            axis_style = value
        if key =="label_angle":
            label_angle = value
    dipole_matrix = pu.remove_unnecessary_indices(dipole_matrix)
    num_dipoles = np.size(dipole_matrix, axis=0)
    if (len(colors_set)<num_dipoles):
        repeat_colors_its = np.ceil(num_dipoles/len(colors_set))
        colors_set = colors_set*int(repeat_colors_its)
    alphas = list(itertools.repeat(1, num_dipoles))
    dipole_matrix = dipole_matrix.real
    if (np.size(dipole_matrix,axis = 1)== 3):
        dipole_matrix = dipole_matrix[:,:2]
    for key, value in kwargs.items():
        if key == "alphas":
            alphas = value
    if (figure == ""):
        figure, axis = plt.subplots()
    else:
        figure.show() #I have no idea why this horribleness is needed, but the
        #figure won't display properly otherwise
    dip_mags = np.sqrt(np.sum(dipole_matrix ** 2, axis=1))
    max_magnitude = np.max(dip_mags)
    if (axis_style == "square"):
        plt.gca().set_aspect(1. / axis.get_data_ratio())
    elif (axis_style == "equal"):
        plt.gca().set_aspect('equal', adjustable='box')
        axis.axis("equal")
    if (not (bounds == np.array([0, 1])).all()):
        cbax = figure.add_axes([.05, .05, 0.8, .05])
        norm = colors.Normalize(vmin=np.min(bounds), vmax=np.max(bounds))
        cbar = figure.colorbar(cm.ScalarMappable(cmap="rainbow", norm=norm), cax=cbax, orientation='horizontal',
                               ticks=bounds)
        cbar.set_label(label="Wavelength (nm)", size=20)
        cbar.ax.tick_params(labelsize=20)
    axis.axis("off")
    for i in range(0, num_dipoles):
        axis.arrow(0, 0, dipole_matrix[i, 0], dipole_matrix[i, 1], color=colors_set[i],
                   width=max_magnitude * .005, head_width=max_magnitude * .05, alpha=alphas[i])
        if (len(labels) > i):
            if (label_positions[i, 0] == 0):
                offset_angle = -.2
                arrow_pos = dipole_matrix[i, :]
                rot_mat = np.array(
                    [[np.cos(offset_angle), -np.sin(offset_angle)], [np.sin(offset_angle), np.cos(offset_angle)]])
                new_pos = np.inner(rot_mat, .5 * arrow_pos)
            else:
                new_pos = label_positions[i, :]
            if (label_angle == True and i == 0):
                t_array = np.linspace(0,np.pi/6,100)
                r_val =  1/3*np.max(np.abs(dipole_matrix[i,:]))
                x_arr = r_val*np.cos(t_array)
                y_arr = r_val*np.sin(t_array)
                axis.plot(x_arr,y_arr,linestyle = "dashed",color = "black")
                axis.text(new_pos[0], new_pos[1], labels[i], size=30, color="black")
            else:
                axis.text(new_pos[0], new_pos[1], labels[i], size=30, color=colors_set[i])

    if (filename != ""):
        figure.savefig(filename,bbox_inches='tight')
        figure.show()

def plot_dipoles_energies_color(dipole_matrix, dipole_energies, filename="", figure="", axis="",
                                bounds=np.array([1, 5])):
    colors_array = eV_to_color_hex(dipole_energies, bounds)
    plot_dipoles(dipole_matrix, filename=filename, figure=figure, axis=axis, colors_set=colors_array, bounds=bounds)


def spectrum_plot_dipole_inset(filename, spectrum, to_plot, dipole_matrix, dipole_energies, dipole_labels, spec_labels,
                               figure="", axis="", x_label="", y_label="", style=""):
    if (figure == ""):
        figure, axis = plt.subplots()
    #see https://matplotlib.org/stable/gallery/axes_grid1/inset_locator_demo.html
    axis_inset = inset_locator.inset_axes(axis, width="30%", loc=4)
    dim = to_plot.ndim
    if (dim == 1):
        axis.plt(spectrum, to_plot)
    elif (dim == 2):
        for i in range(0, np.size(to_plot, axis=0)):
            axis.plt(spectrum, to_plot[:, i], label=spec_labels[i])
            if (style == "components" and i > 0):
                axis.plt(spectrum, to_plot[:, i], label=spec_labels[i], linestyle="dashed")
    plot_dipoles_energies_color(dipole_matrix, dipole_energies, figure=figure, axis=axis_inset)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if (filename != ""):
        figure.savefig(filename)
        figure.show()


#dp is dielectric_params
def plot_ldlb_orientations(filename,dipole_mat,e_array, spec, second_film_rotation, dp,
                           rot_array = np.array([0,0,0]),rot_array_2 = np.array([0,0,0]),cd_conversion_factor = 1):
    rot_array_rev = rot_array * np.array([0, 0, -1])
    unit_defs = dt.UNIT_DEFINITIONS(1,1,1/(4*np.pi*0.007297))
    rotate_dip_mat = dt.rotate_vector(rot_array,dipole_mat,transpose = True)
    rotate_dip_mat_rev = dt.rotate_vector(rot_array_rev,dipole_mat,transpose = True)
    rotated_dielectric = dt.create_dielectric_tensor(dp,rotate_dip_mat,e_array,spec,unit_defs,**{"dimension":3})
    dielectric_rev =  dt.create_dielectric_tensor(dp,rotate_dip_mat_rev,e_array,spec,unit_defs,**{"dimension":3})
    lin_opt_params = dt.linear_optics_from_dielectric_tensor(rotated_dielectric,spec)
    lin_opt_params_rev = dt.linear_optics_from_dielectric_tensor(dielectric_rev,spec)


    net_rotation = rot_array_2+second_film_rotation
    dip_mat_second_film = dt.rotate_vector(net_rotation,dipole_mat,transpose = True)
    dielectric_tensor_second_film = dt.create_dielectric_tensor(dp,dip_mat_second_film,e_array,spec,unit_defs,**{"dimension":3})
    lin_opt_params_2 = dt.linear_optics_from_dielectric_tensor(dielectric_tensor_second_film,spec)
    ldlb_two_film_response = dt.ldlb_two_film_from_params(lin_opt_params,lin_opt_params_2)

    net_rotation_rev = rot_array_2-second_film_rotation
    dip_mat_second_film_rev = dt.rotate_vector(net_rotation_rev,dipole_mat,transpose = True)
    dielectric_tensor_second_film_rev = dt.create_dielectric_tensor(dp,dip_mat_second_film_rev,e_array,spec,unit_defs,**{"dimension":3})
    lin_opt_params_2_rev = dt.linear_optics_from_dielectric_tensor(dielectric_tensor_second_film_rev ,spec)
    ldlb_two_film_response_rev = dt.ldlb_two_film_from_params(lin_opt_params,lin_opt_params_2_rev)



    r_flip = rot_array+np.array([np.pi,0,0])
    net_rotation_flip= rot_array_2+second_film_rotation+np.array([np.pi,0,0])
    rotate_dip_mat_second_film_flip = dt.rotate_vector(net_rotation_flip,dipole_mat,transpose = True)
    rotated_dielectric_second_film_flip = dt.create_dielectric_tensor(dp,rotate_dip_mat_second_film_flip,e_array,spec,unit_defs,**{"dimension":3})
    dip_mat_flip = dt.rotate_vector(r_flip,dipole_mat,transpose = True)
    dielectric_flip = dt.create_dielectric_tensor(dp,dip_mat_flip,e_array,spec,unit_defs,**{"dimension":3})
    lin_opt_params_flip = dt.linear_optics_from_dielectric_tensor(dielectric_flip,spec)
    lin_opt_params_2_flip = dt.linear_optics_from_dielectric_tensor(rotated_dielectric_second_film_flip,spec)
    flipped_response = dt.ldlb_two_film_from_params(lin_opt_params_2_flip,lin_opt_params_flip)

    spec_nm = dt.eV_to_nm(spec)
    fig, ax  = plt.subplots()

    ax.plot(spec_nm,ldlb_two_film_response*cd_conversion_factor,label = "LDLB")
    ax.plot(spec_nm,ldlb_two_film_response_rev*cd_conversion_factor,label = "$\phi$ reversed")
    ax.plot(spec_nm,lin_opt_params.ldlb()*cd_conversion_factor,linestyle = "dotted",label = "solo film")
    ax.plot(spec_nm,lin_opt_params_flip.ldlb()*cd_conversion_factor,linestyle = "dotted",label = "solo film flipped")
    ax.plot(spec_nm,(ldlb_two_film_response+flipped_response)/(2)*cd_conversion_factor,linestyle= "dashed",label = "semi-sum")
    ax.plot(spec_nm,flipped_response*cd_conversion_factor,label = "Flipped Apparatus")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("CD")
    ax.legend()
    fig.savefig(filename+"spec.png")
    fig.show()


def plot_set(filename, x_axis,y_axis_set,figure = None, axis = None):
    if (figure is None):
        figure, axis = plt.subplots()
    num_its = np.size(y_axis_set,axis =0 )
    for i in range(num_its):
        axis.plot(x_axis,y_axis_set[i,:])
    pu.filename_handling(figure,filename)


def plot_set_colored(filename,x_axis,y_axis_set,y_axis_set_color_values,figure=  None,axis = None,x_label = None,y_label = None,opacity = 1,norm_max =1,norm_min = 0,colorbar_label = "",cmap = 'seismic',**kwargs):
    '''
    Plots a line collection of colored lines according to some color map and value array
    Note that for a value, a line is drawn centered at that value and extending to the previous and subsequent (x,y) values.
    This means that boundary values are truncated and this plotting is most suitable for intermediate regions
    Also note that elements are ordered such that those with greater absolute values (y_axis_set_color_values) are drawn last
    In the convention adhered to here, the center corresponds to the (x,y) pair that has a color_value (z), and segments are
    drawn between the averages of adjacent values (i.e., segments = np.concatenate([((1-draw_buffer)*points[:-2]+points[1:-1])/(2), points[1:-1], ((1+draw_buffer)*points[2:]+points[1:-1])/(2)], axis=1))
    The draw_buffer (default 1e-4) is a value near 0 that slightly offsets the averaging so that adjacent segments barely overlap, preventing whitespace. Note that this
    draw_buffer should be 0 if one wants no such behavior and negative if x_axis is ordered in decreasing order
    :param filename:
    :param x_axis:
    :param y_axis_set:
    :param y_axis_set_color_values:
    :param figure:
    :param axis:
    :param x_label:
    :param y_label:
    :param opacity:
    :param norm_max:
    :param norm_min:
    :param colorbar_label:
    :param cmap:
    :param kwargs:
    :return:
    '''
    interp_factor = 1#linear interpolation for smoother colors--defaults to none
    draw_buffer = 1e-4  # slight offset from perfect averaging to ensure (very) slight overlap in drawn objects
    # this is necessary to prevent tiny whitespaces that look bad and make image appear lighter than it is
    to_show_min_max = False
    to_set_x_bounds = False
    to_set_y_bounds = False
    show_colorbar = True
    return_lc = False
    to_show = True
    to_set_y_ticks = False
    to_set_x_ticks = False
    norm_style = "default" #linear
    x_labelpad,y_labelpad= -1,0 
    label_fontsize = 16
    linewidth = 5
    to_adjust_subplots = False
    for key,value in kwargs.items():
        if key == "show_min_max":
            to_show_min_max = value
        if key == "x_bounds":
            to_set_x_bounds = True
            x_bounds = value
        if key == "y_bounds":
            to_set_y_bounds = True
            y_bounds = value
        if key == "label_fontsize":
            label_fontsize = value
        if key == "show_colorbar":
            show_colorbar  =value
        if key == "return_lc":
            return_lc = value
        if key == "to_show":
            to_show = value
        if key == "interp_factor":
            interp_factor = value
        if key == "y_ticks":
            to_set_y_ticks = True
            y_ticks = value
        if key == "x_ticks":
            to_set_x_ticks=  True
            x_ticks = value
        if key == "x_labelpad":
            x_labelpad = value
        if key == "y_labelpad":
            y_labelpad = value
        if key == "norm_style":
            norm_style = value
        if key == "linewidth":
            linewidth = value
        if key == "subplot_adjustments":
            to_adjust_subplots = True
            subplot_adjustments = value
        if key == "draw_buffer":
            draw_buffer = value
    if (figure is None):
        figure, axis = plt.subplots()

    if (not to_set_x_bounds):
        axis.set_xlim(x_axis.min(), x_axis.max())
    else:
        axis.set_xlim(x_bounds[0],x_bounds[1])
    if (not to_set_y_bounds):
        axis.set_ylim(y_axis_set.min(), y_axis_set.max())
    else:
        axis.set_ylim(y_bounds[0],y_bounds[1])
    if (to_set_y_ticks):
        axis.set_yticks(y_ticks)
    if (to_set_x_ticks):
        axis.set_xticks(x_ticks)
    if (x_label is not None):
        axis.set_xlabel(x_label,fontsize= label_fontsize,labelpad = x_labelpad)
    if (y_label is not None):
        axis.set_ylabel(y_label,fontsize= label_fontsize,labelpad= y_labelpad)
    if (y_axis_set.ndim == 1):
        num_its = 1
        y_axis_set = np.expand_dims(y_axis_set,0)
        y_axis_set_color_values = np.expand_dims(y_axis_set_color_values,0)
    else:
        num_its = np.size(y_axis_set, axis=0)
    x_axis_interp = np.interp(np.linspace(x_axis[0],x_axis[-1],interp_factor*np.size(x_axis)),x_axis,x_axis)
    #y_axis_set,y_axis_set_color_values = tetris_nan(y_axis_set,y_axis_set_color_values)
    for i in range(num_its):
        y_axis_set_interp = np.interp(x_axis_interp,x_axis,y_axis_set[i,:])
        y_axis_color_set_interp = ma.masked_greater(np.interp(x_axis_interp,x_axis, y_axis_set_color_values[i,:]),10)
        y_axis_color_set_interp = ma.filled(y_axis_color_set_interp,np.nan)
        x_axis_interp_with_bounds =np.pad(x_axis_interp,(1,1),'constant',constant_values=(x_axis_interp[0],x_axis_interp[-1]))
        y_axis_set_interp_with_bounds = np.pad(y_axis_set_interp,(1,1),'constant',constant_values=(y_axis_set_interp[0],y_axis_set_interp[-1]))
        # see https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
        points = np.array([x_axis_interp_with_bounds, y_axis_set_interp_with_bounds]).T.reshape(-1, 1, 2)
        segments = np.concatenate([((1-draw_buffer)*points[:-2]+points[1:-1])/(2), points[1:-1], (points[2:]+points[1:-1]*(1+draw_buffer))/(2)], axis=1)

        if (i == 0):
            y_axis_color_set_total = y_axis_color_set_interp
            segments_total = segments
        else:
            y_axis_color_set_total = np.concatenate((y_axis_color_set_total,y_axis_color_set_interp),axis =0)
            segments_total =np.concatenate((segments_total,segments),axis= 0)
    segments_total,y_axis_color_set_total = order_by_absolute(segments_total,y_axis_color_set_total)
    if (norm_style == "exp_tanh"):
        exp =1.5
        # see https://matplotlib.org/stable/users/explain/colors/colormapnorms.html
        def _forward(x):
            return np.where(x>0,np.tanh(x**exp),np.tanh(-(-x)**exp))
        def _backward(x):
            return  np.where(x>0,np.arctanh(x)**(1/exp),-(-np.arctanh(x))**(1/exp))

        norm = colors.FuncNorm((_forward,_backward),vmin = norm_min,vmax = norm_max)
    else: norm = plt.Normalize(norm_min,norm_max)
    lc = LineCollection(segments_total,cmap = cmap, norm = norm,alpha = opacity,linewidths=linewidth)
    lc.set_array(y_axis_color_set_total)
    axis.add_collection(lc)
    if (show_colorbar):
        axcb = figure.colorbar(lc)
        axcb.solids.set(alpha = 1)
        axcb.set_label(colorbar_label,fontsize = label_fontsize)
    if (to_show_min_max):
        min_max_str = "Min: "+"%.3f" % np.min(y_axis_set_color_values)+"    Max: "+"%.3f" % np.max(y_axis_set_color_values)
        axis.text(.3, 0.1, min_max_str, horizontalalignment='center',
                  verticalalignment = 'center', transform = axis.transAxes)
    if (to_adjust_subplots):
        figure.subplots_adjust(**subplot_adjustments)
    pu.filename_handling(figure,filename,to_show= to_show)
    if (return_lc):
        return lc


def order_by_absolute(y_values,y_color_values):
    y_abs = np.abs(y_color_values)
    indices = np.argsort(y_abs)
    return y_values[indices,:,:],y_color_values[indices]

def plot_set_on_axis(x_axis,y_axis_set,axis,color = "black",linewidth = 1,linestyle_set = None):
    n_lines = np.size(y_axis_set,axis=  0)
    for i in range(0,n_lines):
        if (linestyle_set is None): axis.plot(x_axis,y_axis_set[i,:],color = color,linewidth = linewidth)
        else: axis.plot(x_axis, y_axis_set[i, :], color=color, linewidth=linewidth,linestyle = linestyle_set[i])

def plot_double_set_colored_shared_y_axis(filename,x_axis,y_axis_stack,y_axis_set_color_values_stack,figure=  None,axis = None,
                                          x_label = None,y_label = None,opacity = 1,norm_max =1,norm_min = 0,colorbar_label = "",
                                          cmap = 'seismic',figsize = (10,5),cbar_style= None,**kwargs):
    show_dispersion_lines = False
    cbar_ticks = None
    for key, value in kwargs.items():
        if key == "show_lines":
            show_dispersion_lines = value
        if key == "cbar_ticks":
            cbar_ticks= value
    params = {'xtick.labelsize':10,'ytick.labelsize':10,'axes.labelsize':12}
    plt.rcParams.update(params)
    if (axis == None):
        figure, axes = plt.subplots(1,2,sharey=True)
    else:
        if (np.size(axis)==2):
            axes = axis
            axes[0].get_shared_y_axes().join(axes[0],axes[1])
        else:
            raise ValueError("Double Plot Axis Must Have 2 Axes")
    figure.set_size_inches(figsize)
    if figsize[0] <4:
        label_fontsize =12
    else:
        label_fontsize =16
    figure.subplots_adjust(hspace = 0)
    kwargs_no_cbar = kwargs.copy()
    kwargs_no_cbar["label_fontsize"] = label_fontsize
    kwargs_no_cbar["show_colorbar"] = False
    kwargs_no_cbar["to_show"] = False
    kwargs_return_lc = kwargs_no_cbar.copy()
    kwargs_return_lc["return_lc"] = True

    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    st_fontsize = 11
    axes[0].set_title("a)", fontsize=st_fontsize, x=.08, y=.8)
    axes[1].set_title("b)", fontsize=st_fontsize, x=.08, y=.8)
    plot_set_colored("",x_axis,y_axis_stack[:,:,0],y_axis_set_color_values_stack[:,:,0],figure=  figure,axis = axes[0]
                     ,x_label = x_label,y_label = y_label,opacity = opacity,norm_max =norm_max,norm_min = norm_min,
                     cmap = cmap,**kwargs_no_cbar)
    lc = plot_set_colored("", x_axis, y_axis_stack[:, :, 1], y_axis_set_color_values_stack[:,:,1], figure=figure,
                     axis=axes[1]
                     , x_label=x_label, y_label=None, opacity=opacity, norm_max=norm_max, norm_min=norm_min,
                      cmap=cmap, **kwargs_return_lc)
    if (show_dispersion_lines):
        #plot_set_on_axis(x_axis,y_axis_stack[:,:,0],axis = axes[0])
        plot_set_on_axis(x_axis, y_axis_stack[:, :, 1], axis=axes[1])
    plt.subplots_adjust(right = 0.8,bottom = .15,left= .1,top = .96)
    if cbar_style == "horizontal":
        cbar_ax = figure.add_axes([.15, .19, 0.8, .05], )
        cb = figure.colorbar(lc, cax=cbar_ax, label=colorbar_label, orientation='horizontal')
        figure.subplots_adjust(bottom=.44,right = .99,left = .17)
    else:
        cbar_ax = figure.add_axes([.82,.16,0.05,.7])
        cb = figure.colorbar(lc,cax = cbar_ax,label = colorbar_label)
    cb.solids.set(alpha=1)
    cb.set_label(colorbar_label, fontsize=label_fontsize,labelpad =0)
    if (cbar_ticks): cb.set_ticks(cbar_ticks)
    cb.ax.tick_params(labelsize=10)
    pu.filename_handling(figure,filename,dpi= 400)

def axes_tuple_along_axis(axes_in_grid,index,axis = 0):
    axis_list = []
    n_axes = np.size(axes_in_grid,axis = axis)
    for i in range(0,n_axes):
        if (axis == 0):
            axis_list.append(axes_in_grid[i,index])
        if (axis == 1):
            axis_list.append(axes_in_grid[index,i])
    return tuple(axis_list)

def share_square_axes(axes_in_grid):
    num_rows = np.size(axes_in_grid,axis= 0)
    num_cols = np.size(axes_in_grid,axis =1)
    for i in range(0,num_rows):
        base_ax = axes_in_grid[i,0]
        tuple_axes_cur_row = axes_tuple_along_axis(axes_in_grid,i,1)
        base_ax.get_shared_y_axes().join(tuple_axes_cur_row)
    for i in range(0,num_cols):
        base_ax = axes_in_grid[0,i]
        tuple_axes_cur_row = axes_tuple_along_axis(axes_in_grid,i,0)
        base_ax.get_shared_x_axes().join(tuple_axes_cur_row)

def sum_to_linestyle(hel_pol_sums,standard_style= "solid",alt_style = "dashed"):
    linestyles = []
    for i in range(0,np.size(hel_pol_sums)):
        if hel_pol_sums[i]> 0:
            linestyles.append(standard_style)
        else:
            linestyles.append(alt_style)
    return linestyles

#plots comparison between four heatmaps where x and y axes are all the same
def plot_quad_set_shared_axes(filename,x_axis,y_axis_stack,y_axis_set_color_values_stack,figure=  None,axis = None,x_label = None,y_label = None,opacity = 1,norm_max =1,norm_min = 0,colorbar_label = "",cmap = 'seismic',figsize = (6.6,6.6),**kwargs):
    show_dispersion_lines = False
    to_make_single_x_label = False
    to_make_ellipses = False
    for key, value in kwargs.items():
        if key == "show_lines":
            show_dispersion_lines = value
        if key == "make_single_x_label":
            to_make_single_x_label = value
        if key == "ellipses":
            to_make_ellipses = True
            ellipse_matrix = value # matrix of xy, width, height of ellipses
    params = {'xtick.labelsize':10,'ytick.labelsize':10,'axes.labelsize':12}
    plt.rcParams.update(params)
    if (axis == None):
        figure, axes_grid = plt.subplots(2,2,sharey=True,sharex=True)
        axes = axes_grid.reshape(-1)
    else:
        if (np.size(axis)==4):
            share_square_axes(axis)
            axes = axis.reshape(-1)
        else:
            raise ValueError("Quad Plot Axis Must Have 2 Axes")
    figure.set_size_inches(figsize)
    figure.subplots_adjust(hspace = 0.05)
    figure.subplots_adjust(wspace = 0)
    kwargs_no_cbar = kwargs.copy()
    if (not to_make_single_x_label):
        kwargs_no_cbar["x_label"] = x_label
    kwargs_no_cbar["show_colorbar"] = False
    kwargs_no_cbar["to_show"] = False
    kwargs_return_lc = kwargs_no_cbar.copy()
    kwargs_return_lc["return_lc"] = True
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    sub_tit_x = .09
    axes[0].set_title("a)", fontsize=11, x=sub_tit_x, y=.77)
    axes[1].set_title("b)", fontsize=11, x=sub_tit_x, y=.77)
    axes[2].set_title("c)",fontsize =11,x=sub_tit_x,y =.77)
    axes[3].set_title("d)", fontsize=11, x=sub_tit_x, y=.77)

    for i in range(0,3):
        cur_kwargs = kwargs_no_cbar.copy()
        if (i == 0 or i == 1):
            cur_kwargs["x_label"] = ""
            axes[i].tick_params(axis = "x",direction = "in")
        plot_set_colored("",x_axis,y_axis_stack[:,:,i],y_axis_set_color_values_stack[:,:,i],figure=  figure,axis = axes[i],opacity = opacity,norm_max =norm_max,norm_min = norm_min,
                     cmap = cmap,**cur_kwargs)
    lc = plot_set_colored("", x_axis, y_axis_stack[:, :, 3], y_axis_set_color_values_stack[:,:,3], figure=figure,
                     axis=axes[3]
                     , opacity=opacity, norm_max=norm_max, norm_min=norm_min,
                      cmap=cmap, **kwargs_return_lc)
    if (show_dispersion_lines):
        for i in range(0,4):
            hel_pol_sums = np.sum(np.abs(y_axis_set_color_values_stack[:,:,i]),axis = 1)
            linestyle_set = sum_to_linestyle(hel_pol_sums,"solid","dashed")
            plot_set_on_axis(x_axis,y_axis_stack[:,:,i],axis = axes[i],linestyle_set = linestyle_set)
    if (to_make_ellipses):
        for i in range(np.size(ellipse_matrix, axis=0)):
            params = ellipse_matrix[i, :]
            ellipse(xy=(params[0], params[1]), width=params[2], height=params[3], color="orange", ax=axes[i])
    figure.subplots_adjust(left = .13,right = 0.99,bottom = .31,top = .99)

    if (to_make_single_x_label):
        plt.gcf().text(.54,.24,x_label,va='center',ha='center',fontsize = 14)
    plt.gcf().text(0.04,.47,y_label,va='center',ha='center',rotation = 'vertical',fontsize = 12)
    plt.gcf().text(0.04, .83, y_label, va='center', ha='center', rotation='vertical', fontsize=12)
    cbar_ax = figure.add_axes([.15,.12,0.8,.05],)
    cb = figure.colorbar(lc,cax = cbar_ax,label = colorbar_label,orientation='horizontal',pad = 1)
    if (cb.vmax == .5 and cb.vmin == -.5):
        cb.set_ticks([-0.5,-.25, 0.0, .25,0.5])
        cb.set_ticklabels([-0.5,"",0.0,"",0.5])
    cb.solids.set(alpha=1)
    cb.set_label(colorbar_label, fontsize=12,labelpad=  0)
    pu.filename_handling(figure,filename,dpi= 1000)

def plot_triple_set_di_bari(filename,x_axis,y_axis_stack,y_axis_set_color_values_stack,figure=  None,axis = None,x_label = None,y_label = None,opacity = 1,norm_max =1,norm_min = 0,colorbar_label = "",cmap = 'seismic',figsize = (6.6,6.6),**kwargs):
    show_dispersion_lines = False
    to_make_single_x_label = False
    ur_placeholder = False
    cbar_style = "horizontal"
    for key, value in kwargs.items():
        if key == "show_lines":
            show_dispersion_lines = value
        if key == "make_single_x_label":
            to_make_single_x_label = value
        if key == "ur_placeholder":
            ur_placeholder = value
        if key == "cbar_style":
            cbar_style = value
    params = {'xtick.labelsize':10,'ytick.labelsize':10,'axes.labelsize':12}
    plt.rcParams.update(params)
    if (axis == None):
        figure, axes_grid = plt.subplots(2,2,sharey=True,sharex=True)
        axes = axes_grid.reshape(-1)
    else:
        if (np.size(axis)==4):
            share_square_axes(axis)
            axes = axis.reshape(-1)
        else:
            raise ValueError("Quad Plot Axis Must Have 2 Axes")
    axes[1].axis('off')
    figure.set_size_inches(figsize)
    figure.subplots_adjust(hspace = 0.05)
    figure.subplots_adjust(wspace = 0)
    kwargs_no_cbar = kwargs.copy()
    if (not to_make_single_x_label):
        kwargs_no_cbar["x_label"] = x_label
    kwargs_no_cbar["show_colorbar"] = False
    kwargs_no_cbar["to_show"] = False
    kwargs_return_lc = kwargs_no_cbar.copy()
    kwargs_return_lc["return_lc"] = True
    kwargs_no_cbar_no_xlabel = kwargs_no_cbar.copy()
    kwargs_no_cbar_no_xlabel["x_label"] = None
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    sub_tit_x = .09
    subtitle_set = ["a)","b)","c)"]
    if(ur_placeholder):
        subtitle_set = ["a)","c)","d)"]
    axes[0].set_title(subtitle_set[0], fontsize=11, x=sub_tit_x, y=.77)
    axes[2].set_title(subtitle_set[1],fontsize =11,x=sub_tit_x,y =.77)
    axes[3].set_title(subtitle_set[2], fontsize=11, x=sub_tit_x, y=.77)

    plot_set_colored("",x_axis,y_axis_stack[:,:,0],y_axis_set_color_values_stack[:,:,0],figure=  figure,axis = axes[0],opacity = opacity,norm_max =norm_max,norm_min = norm_min,
                     cmap = cmap,**kwargs_no_cbar_no_xlabel)
    plot_set_colored("", x_axis, y_axis_stack[:, :, 1], y_axis_set_color_values_stack[:, :, 1], figure=figure,
                     axis=axes[2], opacity=opacity, norm_max=norm_max, norm_min=norm_min,
                     cmap=cmap, **kwargs_no_cbar)
    lc = plot_set_colored("", x_axis, y_axis_stack[:, :, 2], y_axis_set_color_values_stack[:,:,2], figure=figure,
                     axis=axes[3]
                     , opacity=opacity, norm_max=norm_max, norm_min=norm_min,
                      cmap=cmap, **kwargs_return_lc)
    if (show_dispersion_lines):
        for i in range(1,3):
            hel_pol_sums = np.sum(np.abs(y_axis_set_color_values_stack[:,:,i]),axis = 1)
            linestyle_set = sum_to_linestyle(hel_pol_sums,"solid","dashed")
            plot_set_on_axis(x_axis,y_axis_stack[:,:,i],axis = axes[i+1],linestyle_set = linestyle_set)

    if (to_make_single_x_label):
        plt.gcf().text(.54,.24,x_label,va='center',ha='center',fontsize = 14)
    plt.gcf().text(0.04,.45,y_label,va='center',ha='center',rotation = 'vertical',fontsize = 12)
    plt.gcf().text(0.04, .8, y_label, va='center', ha='center', rotation='vertical', fontsize=12)
    if (cbar_style == "horizontal"):
        cbar_ax = figure.add_axes([.15,.13,0.8,.05],)
        cb = figure.colorbar(lc,cax = cbar_ax,label = colorbar_label,orientation='horizontal')
        figure.subplots_adjust(left=.13, right=0.99, bottom=.30, top=.99)
    elif (cbar_style == "vertical"):
        cbar_ax = figure.add_axes([.8, .1, 0.05, .7], )
        cb = figure.colorbar(lc, cax=cbar_ax, label=colorbar_label, orientation='vertical')
        figure.subplots_adjust(left=.18, right=0.7, bottom=.15, top=.98)
    else:
        raise ValueError("Invalid cbar_style")
    if (cb.vmax == .5 and cb.vmin == -.5):
        cb.set_ticks([-0.5,-.25, 0.0, .25,0.5])
        cb.set_ticklabels([-0.5,"",0.0,"",0.5])
    cb.solids.set(alpha=1)
    cb.set_label(colorbar_label, fontsize=12)
    pu.filename_handling(figure,filename,dpi= 2000)

def plot_log_manifold(filename,x_linspace,y_linspace,manifold,figure= None,axis = None,x_label = "",y_label = "",cbar_label = ""):
    x_mesh, y_mesh = np.meshgrid(x_linspace,y_linspace)
    if (figure is None):
        figure, axis = plt.subplots()
    cs = axis.pcolor(x_mesh,y_mesh,manifold, norm=colors.LogNorm(vmin = .1,vmax = 10),cmap = plt.get_cmap("seismic"))
    cbar = figure.colorbar(cs)
    if (x_label):
        axis.set_xlabel(x_label)
    if (y_label):
        axis.set_ylabel(y_label)
    if (cbar_label):
        cbar.set_label(cbar_label)
    pu.filename_handling(figure,filename)


def mask_array_by_other_array_condition(array_to_mask,masking,op,op_comparison):
    mask = op(masking, op_comparison)
    if (ma.is_masked(array_to_mask)==True):
        return ma.masked_array(array_to_mask,mask= np.logical_or(~mask,ma.getmask(array_to_mask)))
    else:
        return ma.masked_array(array_to_mask,mask = ~mask)


def ellipse(xy, width, height, edgecolor="black", facecolor="none", axis=None ):
    """ Adds an ellipse
    modified to show on arbitrary axis and above main plot
    from https://matplotlib.org/stable/gallery/shapes_and_collections/ellipse_demo.html
    (BSD-compatible, see license file in Licenses)
    """
    ellipse = Ellipse(xy=xy,width = width,height = height,zorder = 2)
    if axis is None: axis = plt.gca()
    axis.add_artist(ellipse)
    ellipse.set_clip_box(axis.bbox)
    ellipse.set_edgecolor(edgecolor)
    ellipse.set_facecolor(facecolor)


