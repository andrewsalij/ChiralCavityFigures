import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import mueller
import berreman_mueller as bm

'''Figure S12 (tentative numbering)
    2X2 plot of the corner elements of the Mueller matrix for the 450 nm sample
    shared colorbar
'''

fig_file_ext = ".png"
data_dir = os.sep.join((os.getcwd(),"MuellerMat_Data"))
#DATA LOADING
plt.rcParams.update({'font.size': 7,'axes.unicode_minus':False})

'''This Excel file is formatted in a manner acceptable for a source file (.xslx, has labelled columns)'''
mueller_mat_elems_450_nm_df = pd.read_excel(os.sep.join((data_dir,"new_mueller_data_450nm.xlsx")),sheet_name = "m14 and m41")

mueller_mat_data_array_450_nm = mueller_mat_elems_450_nm_df.to_numpy()
#these values seem to be incorrectly written as 0--adding them fixes the mesh
mueller_mat_data_array_450_nm[0,0] = .5
mueller_mat_data_array_450_nm[0,1] = .5

n_x =  11
n_y =  11
assert n_x*n_y == np.size(mueller_mat_data_array_450_nm,0)
sorted_y_args = mueller_mat_data_array_450_nm[:,1].argsort()
mueller_mat_data_array_450_nm = mueller_mat_data_array_450_nm[sorted_y_args,:]
for i in np.arange(n_y):
    sorted_x_args = mueller_mat_data_array_450_nm[i*n_x:(i+1)*n_x,:][:,0].argsort()
    mueller_mat_data_array_450_nm[i*n_x:(i+1)*n_x,:] = mueller_mat_data_array_450_nm[i*n_x:(i+1)*n_x,:][sorted_x_args,:]

x_space = mueller_mat_data_array_450_nm[:,0].reshape(11,11)
y_space = mueller_mat_data_array_450_nm[:,1].reshape(11,11)

#converting to mm
x_space = x_space*10
y_space = y_space*10

m14 = mueller_mat_data_array_450_nm[:,2].reshape(11,11)
m41 = mueller_mat_data_array_450_nm[:,3].reshape(11,11)

m_sym = (m14+ m41)/2
m_asym = (m14 - m41)/2

fig, axes = plt.subplots(2,2,figsize= (3.46,2.9),sharex=True,sharey =True)
subtit_strs = ["a","b","c","d"]
stx, sty = 0.05,1.05

cmap  = plt.get_cmap("seismic")
kwargs = dict(cmap = cmap,vmin = -.2,vmax = .2)

m14_pmesh = axes[0,0].pcolormesh(x_space,y_space,m14,**kwargs)
axes[0,0].set_title(r"$M_{03}$",pad = 4)
axes[0,0].text(stx,sty,subtit_strs[0],weight ="bold",fontsize=7, transform = axes[0,0].transAxes)

m41_pmesh = axes[0,1].pcolormesh(x_space,y_space,m41,**kwargs)
axes[0,1].set_title(r"$M_{30}$",pad = 4)
axes[0,1].text(stx,sty,subtit_strs[1],weight ="bold",fontsize= 7, transform = axes[0,1].transAxes)

m_sym_pmesh = axes[1,0].pcolormesh(x_space,y_space,m_sym,**kwargs)
axes[1,0].set_title(r"$\frac{M_{03}+M_{30}}{2}$",pad = 6)
axes[1,0].text(stx,sty,subtit_strs[2],weight ="bold",fontsize = 7, transform = axes[1,0].transAxes)

m_asym_pmesh = axes[1,1].pcolormesh(x_space,y_space,m_asym,**kwargs)
axes[1,1].set_title(r"$\frac{M_{03}-M_{30}}{2}$",pad = 6)
axes[1,1].text(stx,sty,subtit_strs[3],weight ="bold",fontsize= 7, transform = axes[1,1].transAxes)

shared_label_fs = 8
fig.text(.38,.01,"X (mm)",fontsize = shared_label_fs)
fig.text(.02,.43,"Y (mm)",fontsize= shared_label_fs,rotation = 90)

cax = fig.add_axes([0.8,0.1,0.02,0.82])
cbar = fig.colorbar(m14_pmesh,ax = axes[0,0],cax = cax,orientation = "vertical")
cbar.set_label(r"Value ($M_{00}$ normalized)")

fig_dir = os.getcwd()
fig.subplots_adjust(wspace = 0.12,hspace = 0.28,left = .15,right = 0.78,top  =.93)
fig.savefig(os.sep.join((fig_dir,"figure_s2_mueller_mat_450nm"+fig_file_ext)),dpi = 600)
fig.show()

make_source_files = True
if (make_source_files):
    source_path = os.sep.join((os.getcwd(), "Source_Files"))
    source_dir = os.sep.join((source_path, "Fig_S2"))
    os.makedirs(source_dir, exist_ok=True)

    objs = [x_space,y_space,m14,m41,m_sym,m_asym]
    excel_writer = pd.ExcelWriter(os.sep.join((source_dir, "mueller_maps.xlsx")))
    sheet_names = ["x_space","y_space","m03_pmesh","m30_pmesh","m_sym_pmesh","m_asym_pmesh"]

    for i in np.arange(len(objs)):
        cur_array = objs[i]
        cur_df = pd.DataFrame(cur_array)
        cur_df.to_excel(excel_writer,sheet_name = sheet_names[i],index = False)
    excel_writer.close()


