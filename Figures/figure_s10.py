import matplotlib.pyplot as plt
import berreman_mueller as bm
import os
import pandas as pd
cwd = os.getcwd()

source_dir = os.sep.join((cwd,"Source_Files","Fig_S10"))

dataframe_100nm = pd.read_excel(os.sep.join((source_dir,"cd_100_nm.xlsx")),index_col = 0)
dataframe_300nm = pd.read_excel(os.sep.join((source_dir,"cd_300_nm.xlsx")),index_col = 0)

def df_to_numpy_data(dataframe):
    column_label = dataframe.columns.to_numpy()
    index_column = dataframe.index.to_numpy()
    data = dataframe.to_numpy()
    return column_label,index_column,data
angle_set_degrees_100nm,wl_nm_array_100nm,cd_100nm = df_to_numpy_data(dataframe_100nm)
angle_set_degrees_300nm,wl_nm_array_300nm,cd_300nm = df_to_numpy_data(dataframe_300nm)

plt.rcParams.update({'font.size': 7,'axes.unicode_minus':False})

fig, axes = plt.subplots(ncols = 2,figsize=(7.08,3),sharey=True)



bm.plot_heatmap_spectra(filename="",x_array=angle_set_degrees_100nm,y_array=wl_nm_array_100nm,z_matrix=cd_100nm,
                        **{'x_label':"Angle (deg.)",'y_label':"Wavelength (nm)",'make_cbar':False,
                           'cmap':'seismic','title':'','figure':fig,'axis':axes[0],
                           'vmin':-2000,'vmax':2000,'dpi':500})
bm.plot_heatmap_spectra(filename="",x_array=angle_set_degrees_300nm,y_array=wl_nm_array_300nm,z_matrix=cd_300nm,
                        **{'x_label':"Angle (deg.)",'make_cbar':False,
                           'cmap':'seismic','title':'','figure':fig,'axis':axes[1],
                           'vmin':-2000,'vmax':2000,'dpi':500})
cbax = fig.add_axes([0.85, 0.15, 0.03, 0.8])
cbar = plt.colorbar(axes[1].collections[0], cax=cbax)
cbar.set_label(r'CD (mdeg)')

titles = ["100 nm","300 nm"]
sub_titles = ["a","b"]
for i in range(2):
    axes[i].set_title(titles[i])
    axes[i].text(.03,.92,sub_titles[i],weight="bold",transform=axes[i].transAxes)

fig.subplots_adjust(wspace = 0.1,right= .8,bottom=.15,top = .9)
fig.savefig("ptpo_cd_comparison.pdf",dpi = 500)
fig.show()
