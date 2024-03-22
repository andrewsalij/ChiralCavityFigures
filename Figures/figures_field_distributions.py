import matplotlib.colors
import field_plotting
import numpy as np
import matplotlib.pyplot as plt
import os

import pandas as pd
'''
Creates SI figures for in-plane field distributions (propagated via transfer matrices)
Data listed as (forward,backward)
'''

plt.rcParams.update({'font.size': 7,'axes.unicode_minus':False})
cwd = os.getcwd()
basedir, cur_dir= os.path.split(cwd)
theory_data_path = os.sep.join((basedir,"PTPO_Theory"))

intensities_440_0_deg = np.load(os.sep.join((theory_data_path,"440_0_deg_intensities.npy")))
intensities_440_21_deg = np.load(os.sep.join((theory_data_path,"440_21_deg_intensities.npy")))

z_array = np.load(os.sep.join((theory_data_path,"z_cavity_intensities_plotting.npy")))
x_array= np.load(os.sep.join((theory_data_path,"x_cavity_intensities_plotting.npy")))

def norm_intensities_get_diff(intensities_matrix):
    '''
    Just for usage here due to unique data structuring
    :param intesities_matrix: (2,2,z,x)
    :return:
    '''
    max_vals = np.max(intensities_matrix,axis=(1,2,3)) #all axes except (forward, backward)
    normed_intensities = np.einsum("ijka,i->ijka",intensities_matrix,1/max_vals)
    diff_mat = normed_intensities[:,0,:,:]-normed_intensities[:,1,:,:]
    diff_mat=  diff_mat.reshape((np.size(diff_mat,axis=0),1,np.size(diff_mat,axis=1),np.size(diff_mat,axis=2)))
    return normed_intensities,diff_mat

int_norm_440_0_deg ,diff_norm_440_0_deg = norm_intensities_get_diff(intensities_440_0_deg)
int_norm_440_21_deg ,diff_norm_440_21_deg = norm_intensities_get_diff(intensities_440_21_deg)

int_normalization = matplotlib.colors.Normalize(vmin=0,vmax= 1)
diff_normalization =  matplotlib.colors.Normalize(vmin=-.5,vmax= .5)

row_titles = ["Forward","Backward"]
col_titles = ["RHP","LHP","RHP-LHP"]

total_figure = plt.figure(figsize=  (8,4))
subfigs = total_figure.subfigures(1,2,width_ratios = [2,1])
axes_left = subfigs[0].subplots(2,2,sharex= True,sharey= True)
axes_right = np.atleast_2d(subfigs[1].subplots(2,1,sharex= True)).T
field_plotting.plot_field_mesh_array(z_array, x_array, int_norm_440_0_deg, figure = subfigs[0], axes= axes_left,
                                     norm = int_normalization, cmap ="inferno", cbar_label=r"$|E_{||}|^2$ (normed)",
                                     row_titles=row_titles, col_titles=col_titles[:2])

field_plotting.plot_field_mesh_array(z_array, x_array, diff_norm_440_0_deg, figure = subfigs[1], axes= axes_right,
                                     norm = diff_normalization, cmap ="seismic", cbar_label=r"$\Delta|E_{||}|^2$ (normed)",
                                     row_titles=row_titles, col_titles=[col_titles[2]])
total_figure.savefig("440_nm_field_distributions_cavity_0_deg.pdf",dpi = 500)
total_figure.show()

total_figure = plt.figure(figsize=  (8,4))
subfigs = total_figure.subfigures(1,2,width_ratios = [2,1])
axes_left = subfigs[0].subplots(2,2,sharex= True,sharey= True)
axes_right = np.atleast_2d(subfigs[1].subplots(2,1,sharex= True)).T
field_plotting.plot_field_mesh_array(z_array, x_array, int_norm_440_21_deg, figure = subfigs[0], axes= axes_left,
                                     norm = int_normalization, cmap ="inferno", cbar_label=r"$|E_{||}|^2$ (normed)",
                                     row_titles=row_titles, col_titles=col_titles[:2])

field_plotting.plot_field_mesh_array(z_array, x_array, diff_norm_440_21_deg, figure = subfigs[1], axes= axes_right,
                                     norm = diff_normalization, cmap ="seismic", cbar_label=r"$\Delta|E_{||}|^2$ (normed)",
                                     row_titles=row_titles, col_titles=[col_titles[2]])
total_figure.savefig("440_nm_field_distributions_cavity_21_deg.pdf",dpi = 500)
total_figure.show()

make_source_files = True
if (make_source_files):
    int_norm_440_0_deg = int_norm_440_0_deg[:,:2,:,:].reshape(4,np.size(z_array),np.size(x_array))
    int_norm_440_21_deg = int_norm_440_21_deg[:,:2,:,:].reshape(4,np.size(z_array),np.size(x_array))
    diff_norm_440_0_deg = diff_norm_440_0_deg.reshape(2,np.size(z_array),np.size(x_array))
    diff_norm_440_21_deg = diff_norm_440_21_deg.reshape(2,np.size(z_array),np.size(x_array))
    source_path = os.sep.join((os.getcwd(), "Source_Files"))
    source_dir = os.sep.join((source_path, "Fig_S14"))
    os.makedirs(source_dir, exist_ok=True)

    np.savetxt(os.sep.join((source_dir,"z_array.csv")),z_array,delimiter = ",")
    np.savetxt(os.sep.join((source_dir,"x_array.csv")),x_array,delimiter = ",")

    excel_writer = pd.ExcelWriter(os.sep.join((source_dir, "440_nm_0_deg_intensities.xlsx")))
    excel_sheets=  ["int_rhp_for","int_lhp_for","int_rhp_back","int_lhp_back"]
    for i in np.arange(len(excel_sheets)):
        pd.DataFrame(int_norm_440_0_deg[i,:,:]).to_excel(excel_writer,sheet_name = excel_sheets[i],index = False,header = False)
    excel_writer = pd.ExcelWriter(os.sep.join((source_dir, "440_nm_21_deg_intensities.xlsx")))
    excel_sheets = ["int_rhp_for", "int_lhp_for", "int_rhp_back", "int_lhp_back"]
    for i in np.arange(len(excel_sheets)):
        pd.DataFrame(int_norm_440_21_deg[i,:,:]).to_excel(excel_writer,sheet_name = excel_sheets[i],index = False,header = False)
    excel_writer = pd.ExcelWriter(os.sep.join((source_dir, "440_nm_0_deg_intensities_diff.xlsx")))
    excel_sheets=  ["diff_for","diff_back"]
    for i in np.arange(len(excel_sheets)):
        pd.DataFrame(diff_norm_440_0_deg[i,:,:]).to_excel(excel_writer,sheet_name = excel_sheets[i],index = False,header = False)

    excel_writer = pd.ExcelWriter(os.sep.join((source_dir, "440_nm_21_deg_intensities_diff.xlsx")))
    excel_sheets=  ["diff_for","diff_back"]
    for i in np.arange(len(excel_sheets)):
        pd.DataFrame(diff_norm_440_21_deg[i,:,:]).to_excel(excel_writer,sheet_name = excel_sheets[i],index = False,header = False)