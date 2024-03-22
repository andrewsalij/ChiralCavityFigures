import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
from mueller import plot_heatmap_spectra
'''
Construction of figure that compares theoretical and experimental maps 
Note that "Backward" and "Forward" are opposite of "Front and "Back"
'''
plt.rcParams.update({'font.size': 7,'axes.unicode_minus':False})
plt.rcParams.update({'axes.labelsize': 7})
cwd = os.getcwd()
base_dir,cur_folder = os.path.split(cwd)
theory_data_path = os.sep.join((base_dir,"PTPO_Theory"))
exp_data_path =os.sep.join((base_dir,r"Experimental Cavity Data\angle-map-raw-data.xlsx"))


#numpy arrays of raw transmissions-used for most things now
exp_npy_data_path = os.sep.join((base_dir,r"Experimental Cavity Data"))

plot_style = "cd_mdeg"

#EXPERIMENTAL DATA LOADING
backside_g_full = pd.read_excel(exp_data_path,sheet_name="u35spot2bs")
backside_g_full = np.array(backside_g_full)
exp_energy_array = backside_g_full[:,1]
exp_angle_array = np.linspace(-21,21,15)

exp_tl_front = np.load((os.sep.join((exp_npy_data_path,r"u35_front_data\ptpo_cavity_front_tl.npy"))))
exp_tr_front = np.load((os.sep.join((exp_npy_data_path,r"u35_front_data\ptpo_cavity_front_tr.npy"))))
exp_tl_back = np.load((os.sep.join((exp_npy_data_path,r"u35_back_data\ptpo_cavity_back_tl.npy"))))
exp_tr_back = np.load((os.sep.join((exp_npy_data_path,r"u35_back_data\ptpo_cavity_back_tr.npy"))))
exp_tl_bare = np.load((os.sep.join((exp_npy_data_path,r"empty_cavity_front_datav2\empty_cavity_front_tl.npy"))))
exp_tr_bare = np.load((os.sep.join((exp_npy_data_path,r"empty_cavity_front_datav2\empty_cavity_front_tr.npy"))))
#THEORETICAL DATA LOADING
theory_energy_array = np.load(os.sep.join((theory_data_path,"theory_ptpo_back_eV_array.npy")))
theory_angle_array = np.load(os.sep.join((theory_data_path,"theory_ptpo_back_angle_set.npy")))

# theory_g_back_matrix = np.load(os.sep.join((theory_data_path,"theory_ptpo_back_g_abs_matrix.npy")))
# theory_g_front_matrix =  np.load(os.sep.join((theory_data_path,"theory_ptpo_front_g_abs_matrix.npy")))

t_l_for = np.load(os.sep.join((theory_data_path,"theory_ptpo_front_trans_lhp.npy")))
t_r_for =  np.load(os.sep.join((theory_data_path,"theory_ptpo_front_trans_rhp.npy")))

t_l_rev = np.load(os.sep.join((theory_data_path,"theory_ptpo_back_trans_lhp.npy")))
t_r_rev =  np.load(os.sep.join((theory_data_path,"theory_ptpo_back_trans_rhp.npy")))

t_l_bare = np.load(os.sep.join((theory_data_path,"bare_cavitytrans_LHP.npy")))
t_r_bare =  np.load(os.sep.join((theory_data_path,"bare_cavitytrans_RHP.npy")))

if (plot_style == "g_trans"):
    theory_g_back_matrix =2*(t_l_rev-t_r_rev)/(t_l_rev+t_r_rev)
    theory_g_front_matrix = 2*(t_l_for-t_r_for)/(t_l_for+t_r_for)
    theory_g_bare = 2*(t_l_bare-t_r_bare)/(t_l_bare+t_r_bare)
    exp_g_front_matrix = 2*(exp_tl_front-exp_tr_front)/(exp_tl_front+exp_tr_front)
    exp_g_back_matrix = 2*(exp_tl_back-exp_tr_back)/(exp_tl_back+exp_tr_back)
    exp_bare_cavity_g_matrix = 2*(exp_tl_bare-exp_tr_bare)/(exp_tl_bare+exp_tl_bare)
elif(plot_style=="cd_mdeg"):
    theory_g_back_matrix =32980*np.log10(t_r_rev/t_l_rev)
    theory_g_front_matrix = 32980*np.log10(t_r_for/t_l_for)
    theory_g_bare = 32980*np.log10(t_r_bare/t_l_bare)
    exp_g_front_matrix = 32980*np.log10(exp_tr_front/exp_tl_front)
    exp_g_back_matrix = 32980*np.log10(exp_tr_back/exp_tl_back)
    exp_bare_cavity_g_matrix = 32980*np.log10(exp_tr_bare/exp_tl_bare)


def eV_to_nm(array):
    '''Converts eV (energy) to nm (wavelength)'''
    return 1e9*2.998e8*4.136e-15/array
theory_wavelength_array = eV_to_nm(theory_energy_array)
exp_wavelength_array = eV_to_nm(exp_energy_array)


fig= plt.figure(figsize = (7.08,5.5))
gs = fig.add_gridspec(18, 21,left=0.07, right=0.98, bottom=0.10, top=0.82, wspace=-.3, hspace=0)
ax1 = fig.add_subplot(gs[:8,:6])
ax2 = fig.add_subplot(gs[:8,7:13])
ax3 = fig.add_subplot(gs[:8,15:])

ax1.sharey(ax2)
ax2.sharey(ax3)

ax4 = fig.add_subplot(gs[10:,:6])
ax5 = fig.add_subplot(gs[10:,7:13])
ax6 = fig.add_subplot(gs[10:,15:])

ax4.sharey(ax5)
ax5.sharey(ax6)

ax4.sharex(ax1)
ax5.sharex(ax2)
ax6.sharex(ax3)

kwargs = {"make_cbar":False, 'cmap':'seismic','title':'','to_show':False}
if plot_style == "g_trans":
    g_factor_title = r"$g$ factor"
    kwargs.update({'vmin': -.12, 'vmax': .12})
elif (plot_style == "cd_mdeg"):
    g_factor_title = "CD (mdeg)"
    kwargs.update(({'vmin':-4000,'vmax':4000}))
to_normalize =True
if to_normalize==True:
    if (plot_style == "g_trans"):g_factor_title = r"$g$ factor (Norm.)"
    if (plot_style == "cd_mdeg"):g_factor_title = r"CD (Norm.)"
    kwargs.update({'vmin':-1,'vmax':1})
    exp_g_front_matrix = exp_g_front_matrix/np.max(np.abs(exp_g_front_matrix))
    exp_g_back_matrix = exp_g_back_matrix/np.max(np.abs(exp_g_back_matrix))
    theory_g_back_matrix = theory_g_back_matrix/np.max(np.abs(theory_g_back_matrix))
    theory_g_front_matrix = theory_g_front_matrix/np.max(np.abs(theory_g_front_matrix))
    exp_bare_cavity_g_matrix= exp_bare_cavity_g_matrix/np.max(np.abs(exp_bare_cavity_g_matrix))
    theory_g_bare = theory_g_bare/np.max(np.abs(theory_g_bare))

plot_heatmap_spectra(exp_angle_array,exp_wavelength_array,exp_g_front_matrix,figure = fig,axis= ax1,
                        **kwargs)
plot_heatmap_spectra(exp_angle_array,exp_wavelength_array,exp_g_back_matrix,figure = fig,axis= ax2,
                        **kwargs)

plot_heatmap_spectra(exp_angle_array,exp_wavelength_array,exp_bare_cavity_g_matrix,figure = fig,axis= ax3,
                        **kwargs)

plot_heatmap_spectra(theory_angle_array,theory_wavelength_array,theory_g_back_matrix,figure = fig,axis= ax4,
                        **kwargs)
plot_heatmap_spectra(theory_angle_array,theory_wavelength_array,theory_g_front_matrix,figure = fig,axis= ax5,
                        **kwargs)

plot_heatmap_spectra(theory_angle_array,theory_wavelength_array,theory_g_bare,figure = fig,axis= ax6,
                        **kwargs)


make_source_files = True
if (make_source_files):
    source_path = os.sep.join((os.getcwd(), "Source_Files"))
    source_dir = os.sep.join((source_path, "Fig_3"))
    os.makedirs(source_dir, exist_ok=True)

    exp_g_stack = [exp_g_front_matrix, exp_g_back_matrix, exp_bare_cavity_g_matrix]
    theory_g_stack = [theory_angle_array, theory_energy_array, theory_g_bare]

    pd.DataFrame(exp_angle_array, columns=["Angle (deg)"]).to_csv( \
        os.sep.join((source_dir, "exp_angle_array.csv")), index=False)
    pd.DataFrame(theory_angle_array, columns=["Angle (deg)"]).to_csv( \
        os.sep.join((source_dir, "theory_angle_array.csv")), index=False)
    pd.DataFrame(exp_energy_array, columns=["Wvl (nm)"]).to_csv( \
        os.sep.join((source_dir, "exp_energy_array.csv")), index=False)
    pd.DataFrame(theory_energy_array, columns=["Wvl (nm)"]).to_csv( \
        os.sep.join((source_dir, "theory_energy_array.csv")), index=False)
    exp_excel_writer = pd.ExcelWriter(os.sep.join((source_dir, "exp_cd_stack.xlsx")))
    sheet_names = ["cavity_front", "cavity_rev", "bare_cavity"]
    for i in np.arange(len(exp_g_stack)):
        cur_df = pd.DataFrame(exp_g_stack[i])
        cur_df.to_excel(exp_excel_writer, sheet_name=sheet_names[i], index=False)
    exp_excel_writer.close()
    theory_excel_writer = pd.ExcelWriter(os.sep.join((source_dir, "theory_cd_stack.xlsx")))
    for i in np.arange(len(theory_g_stack)):
        cur_df = pd.DataFrame(theory_g_stack[i])
        cur_df.to_excel(theory_excel_writer, sheet_name=sheet_names[i], index=False)
    theory_excel_writer.close()


axes = [ax1,ax2,ax3,ax4,ax5,ax6]
for ax in axes:
    ax.set_ylim(410,520)
ax4.set_xlim(-21,21)
ax5.set_xlim(-21,21)
ax6.set_xlim(-21,21)
fs=  7

for ax in [ax4,ax5,ax6]:
    ax.set_xlabel("Angle (deg.)",fontsize=fs)
for ax in [ax1,ax2,ax3]:
    ax.set_xticklabels([])
for ax in [ax1,ax4]:
    ax.set_ylabel("Wavelength (nm)",fontsize= fs)
for ax in [ax2,ax3,ax5,ax6]:
    ax.set_yticklabels([])
sub_tit_str = ["a","b","c","d","e","f"]
for i in range(6):
    cur_ax = axes[i]
    cur_ax.text(.05,.92,sub_tit_str[i],transform=cur_ax.transAxes,fontsize=7,weight = "bold")


y_str=  .83
x_center = .5
fig.text(x_center,.87,"Experiment",fontsize = fs,ha = 'center')
fig.text(x_center,.44,"Theory",fontsize=fs,ha = 'center')
fig.text(.84,y_str,"Empty Cavity",fontsize = fs,ha='center')
fig.text(x_center,y_str,"Forward",fontsize = fs,ha='center')
fig.text(.2,y_str,"Backward",fontsize = fs,ha='center')

mappable_image = ax6.collections[0]
cax = fig.add_axes([0.05, 0.94, 0.9, 0.05])
cbar = fig.colorbar(mappable=mappable_image,cax =cax,orientation = 'horizontal')
cbar.set_label(g_factor_title,fontsize = fs,labelpad=  -25)


left, bottom, width, height = (.68, .01, .32, .87)
rect = plt.Rectangle((left, bottom), width, height,
                     facecolor="black", alpha=0.1,zorder= -10,transform=fig.transFigure,figure = fig)
fig.patches.extend([rect])

fig.subplots_adjust(wspace=.08, hspace=.1,top = .92,bottom = .08,left = .12,right = .98)
fig.savefig("fig_dispersion_comparisonv10.pdf",bbox_inches='tight', pad_inches=0.01,dpi = 500)
fig.show()

idx_440 = np.argmin(np.abs(theory_wavelength_array-440))
idx_21_deg= np.argmin(np.abs(theory_angle_array-21))
idx_0_deg = np.argmin(np.abs(theory_angle_array-00))
g_for_440_21 = theory_g_front_matrix[idx_440,idx_21_deg]
g_back_440_21 = theory_g_back_matrix[idx_440,idx_21_deg]
g_for_440_0 = theory_g_front_matrix[idx_440,idx_0_deg]
g_back_440_0 = theory_g_back_matrix[idx_440,idx_0_deg]