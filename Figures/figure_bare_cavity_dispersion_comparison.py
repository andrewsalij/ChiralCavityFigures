import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
'''
Construction of figure that compares theoretical and experiemental maps 
Note that "Backward" and "Forward" are opposite of "Front and "Back"
'''
fig_file_ext = ".png"

plt.rcParams.update({'font.size': 7,'axes.unicode_minus':False})
create_source_files = True
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
    if (figure is None):
        figure, axis = plt.subplots()
    if (style == "pcolormesh"):
        x_mesh, y_mesh = np.meshgrid(x_array, y_array)
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
    return heatmap


plot_style = "cd_mdeg"

cwd = os.getcwd()
base_dir,cur_folder = os.path.split(cwd)
theory_data_path = os.sep.join((base_dir,"PTPO_Theory"))
exp_data_path =os.sep.join((base_dir,r"Experimental Cavity Data\angle-map-raw-data.xlsx"))


#numpy arrays of raw transmissions-used for most things now
exp_npy_data_path = os.sep.join((base_dir,r"Experimental Cavity Data"))

#EXPERIMENTAL DATA LOADING
backside_g_full = pd.read_excel(exp_data_path,sheet_name="u35spot2bs")
backside_g_full = np.array(backside_g_full)
exp_energy_array = backside_g_full[:,1]
exp_angle_array = np.linspace(-21,21,15)


exp_tl_bare = np.load((os.sep.join((exp_npy_data_path,r"empty_cavity_front_datav2\empty_cavity_front_tl.npy"))))
exp_tr_bare = np.load((os.sep.join((exp_npy_data_path,r"empty_cavity_front_datav2\empty_cavity_front_tr.npy"))))
exp_tl_bare_rev = np.load((os.sep.join((exp_npy_data_path,r"empty_cavity_back_datav2\empty_cavity_back_tl.npy"))))
exp_tr_bare_rev = np.load((os.sep.join((exp_npy_data_path,r"empty_cavity_back_datav2\empty_cavity_back_tr.npy"))))
exp_tl_open = np.load((os.sep.join((exp_npy_data_path,r"open_cavity_raw_data\bare_tl.npy"))))
exp_tr_open = np.load((os.sep.join((exp_npy_data_path,r"open_cavity_raw_data\bare_tr.npy"))))

#THEORETICAL DATA LOADING

t_l_bare = np.load(os.sep.join((theory_data_path,"bare_cavitytrans_LHP.npy")))
t_r_bare =  np.load(os.sep.join((theory_data_path,"bare_cavitytrans_RHP.npy")))

t_l_bare_rev = np.load(os.sep.join((theory_data_path,"bare_cavity_revtrans_LHP.npy")))
t_r_bare_rev =  np.load(os.sep.join((theory_data_path,"bare_cavity_revtrans_RHP.npy")))

t_l_open = np.load(os.sep.join((theory_data_path,"open_cavitytrans_LHP.npy")))
t_r_open =  np.load(os.sep.join((theory_data_path,"open_cavitytrans_RHP.npy")))

exp_angle_array = np.linspace(-21,21,15)

if (plot_style == "g_trans"):
    theory_g_bare = 2 * (t_l_bare - t_r_bare) / (t_l_bare + t_r_bare)
    theory_g_bare_rev = 2 * (t_l_bare_rev - t_r_bare_rev) / (t_l_bare_rev + t_r_bare_rev)
    theory_g_open = 2 * (t_l_open - t_r_open) / (t_l_open + t_r_open)
    exp_g_bare = 2*(exp_tl_bare-exp_tr_bare)/(exp_tl_bare+exp_tl_bare)
    exp_g_bare_rev = 2*(exp_tl_bare_rev-exp_tr_bare_rev)/(exp_tl_bare_rev+exp_tl_bare_rev)
    exp_open_cavity_g_matrix = 2*(exp_tl_open-exp_tr_open)/(exp_tl_open+exp_tl_open)
elif(plot_style=="cd_mdeg"):
    theory_g_bare = 32980*np.log10(t_r_bare/t_l_bare)
    theory_g_bare_rev = 32980*np.log10(t_r_bare_rev/t_l_bare_rev)
    theory_g_open = 32980*np.log10(t_r_open/t_l_open)
    exp_g_bare =32980*np.log10(exp_tr_bare/exp_tl_bare)
    exp_g_bare_rev =32980*np.log10(exp_tr_bare_rev/exp_tl_bare_rev)
    exp_open_cavity_g_matrix = 32980*np.log10(exp_tr_open/exp_tl_open)

fig, ax = plt.subplots(2,3,sharex=True,sharey=True,figsize = (7.08,5.5)) #180 mm width

kwargs = {"make_cbar":False,'cmap':'seismic','title':''}

theory_energy_array = np.load(os.sep.join((theory_data_path,"theory_ptpo_back_eV_array.npy")))
theory_angle_array = np.load(os.sep.join((theory_data_path,"theory_ptpo_back_angle_set.npy")))
if plot_style == "g_trans":
    g_factor_title = r"$g$ factor"
    kwargs.update({'vmin': -.03, 'vmax': .03})
elif (plot_style == "cd_mdeg"):
    g_factor_title = "CD (mdeg)"
    kwargs.update(({'vmin':-1000,'vmax':1000}))
to_normalize =False
if to_normalize==True:
    if (plot_style == "g_trans"):g_factor_title = r"$g$ factor (Norm.)"
    if (plot_style == "cd_mdeg"):g_factor_title = r"CD (Norm.)"
    exp_g_bare = exp_g_bare/np.max(np.abs(exp_g_bare))
    exp_g_bare_rev = exp_g_bare_rev/np.max(np.abs(exp_g_bare_rev))
    exp_open_cavity_g_matrix = exp_open_cavity_g_matrix/np.max(np.abs(exp_open_cavity_g_matrix))
    theory_g_bare = theory_g_bare/np.max(np.abs(theory_g_bare))
    theory_g_bare_rev = theory_g_bare_rev/np.max(np.abs(theory_g_bare_rev))
    theory_g_open = theory_g_open/np.max(np.abs(theory_g_open))
    kwargs.update(({'vmin': -1, 'vmax': 1}))

plot_wl = True
def eV_to_nm(array):
    '''Converts eV (energy) to nm (wavelength)'''
    return 1e9*2.998e8*4.136e-15/array

if (plot_wl):
    exp_energy_array = eV_to_nm(exp_energy_array)
    theory_energy_array= eV_to_nm(theory_energy_array)

plot_heatmap_spectra(exp_angle_array,exp_energy_array,exp_g_bare,figure = fig,axis= ax[0,0],
                        **kwargs) #BACKWARD
plot_heatmap_spectra(exp_angle_array,exp_energy_array,exp_g_bare_rev,figure = fig,axis= ax[0,1],
                        **kwargs) #FORWARD
plot_heatmap_spectra(exp_angle_array,exp_energy_array,exp_open_cavity_g_matrix,figure = fig,axis= ax[0,2],
                        **kwargs)


plot_heatmap_spectra(theory_angle_array,theory_energy_array,theory_g_bare,figure = fig,axis= ax[1,0],
                        **kwargs) #BACKWARD
plot_heatmap_spectra(theory_angle_array,theory_energy_array,theory_g_bare_rev,figure = fig,axis= ax[1,1],
                        **kwargs) #FORWARD
plot_heatmap_spectra(theory_angle_array,theory_energy_array,theory_g_open,figure = fig,axis= ax[1,2],
                        **kwargs)

if (create_source_files):
    source_path = os.sep.join((os.getcwd(), "Source_Files"))
    source_dir = os.sep.join((source_path, "Fig_S6"))
    os.makedirs(source_dir, exist_ok=True)

    exp_g_stack =[exp_g_bare,exp_g_bare_rev,exp_open_cavity_g_matrix]
    theory_g_stack =[theory_angle_array,theory_energy_array,theory_g_open]

    pd.DataFrame(exp_angle_array,columns = ["Angle (deg)"]).to_csv(\
        os.sep.join((source_dir,"exp_angle_array.csv")),index = False)
    pd.DataFrame(theory_angle_array,columns = ["Angle (deg)"]).to_csv(\
        os.sep.join((source_dir,"theory_angle_array.csv")),index = False)
    pd.DataFrame(exp_energy_array,columns = ["Wvl (nm)"]).to_csv(\
        os.sep.join((source_dir,"exp_energy_array.csv")),index = False)
    pd.DataFrame(theory_energy_array,columns = ["Wvl (nm)"]).to_csv(\
        os.sep.join((source_dir,"theory_energy_array.csv")),index = False)
    exp_excel_writer = pd.ExcelWriter(os.sep.join((source_dir,"exp_cd_stack.xlsx")))
    sheet_names = ["bare_cavity","bare_cavity_rev","open_cavity"]
    for i in np.arange(len(exp_g_stack)):
        cur_df = pd.DataFrame(exp_g_stack[i])
        cur_df.to_excel(exp_excel_writer,sheet_name = sheet_names[i],index = False)
    exp_excel_writer.close()
    theory_excel_writer = pd.ExcelWriter(os.sep.join((source_dir,"theory_cd_stack.xlsx")))
    for i in np.arange(len(theory_g_stack)):
        cur_df = pd.DataFrame(theory_g_stack[i])
        cur_df.to_excel(theory_excel_writer,sheet_name = sheet_names[i],index = False)
    theory_excel_writer.close()
ax[1,1].set_xlim(-21,21)
for i in range(3):
    ax[1,i].set_xlabel("Angle (deg.)")
for i in range(2):
    if (plot_wl):
        ax[i, 0].set_ylabel("Wavelength (nm)")
    ax[i,0].set_ylabel("Wavelength (nm)")
sub_tit_str = ["a","b","c","d","e","f"]
for i in range(6):
    cur_ax = ax.flatten()[i]
    cur_ax.text(.05,.92,sub_tit_str[i],transform=cur_ax.transAxes,fontsize=7,weight ="bold")

text_fontsize = 7
fig.text(.53,.85,"Experiment",fontsize = text_fontsize,ha = 'center')
fig.text(.53,.44,"Theory",fontsize=text_fontsize,ha = 'center')

y_str = .825
fig.text(.85,y_str,"Half Cavity",fontsize = text_fontsize,ha='center')
fig.text(.53,y_str,"Forward",fontsize = text_fontsize,ha='center')
fig.text(.22,y_str,"Backward",fontsize = text_fontsize,ha='center')

mappable_image = ax[1,1].collections[0]
cax = fig.add_axes([0.16, 0.94, 0.8, 0.05])
cbar = fig.colorbar(mappable=mappable_image,cax =cax,orientation = 'horizontal')
cbar.set_label(g_factor_title,fontsize = 7,labelpad=  2)
fig.subplots_adjust(wspace=.08, hspace=.12,top = .82,bottom = .08,left = .08,right = .98)
fig.savefig("bare_cavity_dispersion_comparison_v4"+fig_file_ext,bbox_inches='tight', pad_inches=0.01,dpi=500)
fig.show()