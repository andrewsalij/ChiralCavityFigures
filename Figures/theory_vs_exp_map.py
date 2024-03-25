import numpy as np
import matplotlib.pyplot as plt
import os

import pandas as pd

import berreman_mueller as bm

plt.rcParams.update({'font.size': 7,'axes.unicode_minus':False})

cwd = os.getcwd()
base_dir,cur_folder = os.path.split(cwd)
theory_data_path = os.sep.join((base_dir,"PTPO_Theory"))
exp_data_path =os.sep.join((base_dir,r"Experimental Cavity Data\angle-map-raw-data.xlsx"))

map_3_path = os.sep.join((base_dir,r"\Experimental Cavity Data\u27_map3_front_data"))
map_3_tl = np.load(os.sep.join((map_3_path,"map_3_front_tl.npy")))
map_3_tr = np.load(os.sep.join((map_3_path,"map_3_front_tr.npy")))

map_2_path = os.sep.join((base_dir,r"\Experimental Cavity Data\u27_map2_front_data"))
map_2_tl = np.load(os.sep.join((map_2_path,"map_2_front_tl.npy")))
map_2_tr = np.load(os.sep.join((map_2_path,"map_2_front_tr.npy")))

g_type= "cd"
if (g_type == "cd"):
    scaling_factor = np.array([-846.21381,639.42113]) #from CD measurements
    g_str = "CD (mdeg)"
    map2_all = 32980*np.log10(map_2_tr/map_2_tl) #in mdeg
    map3_all = 32980*np.log10(map_3_tr/map_3_tl)
elif (g_type =="g_trans"):
    scaling_factor = 0.05366216778162327
    g_str = r"$g$ factor"
    map2_all = 2*(map_2_tl-map_2_tr)/(map_2_tr+map_2_tl)
    map3_all = 2 * (map_3_tl - map_3_tr) / (map_3_tr + map_3_tl)

map2 = bm.slice_max_abs_val(map2_all)
map3 = bm.slice_max_abs_val(map3_all)


theory_map = np.load("ising_model_spin_mesh_si.npy")
#theory_map = theory_map[0::10,0::10]
theory_map = theory_map/np.max(theory_map) #rescaling to -1, 1
#from TMM/SMM 420-520 nm of PTPO 300 nm (input_params v6)

if np.isscalar(scaling_factor):
    theory_map_scaled = theory_map*scaling_factor
else:
    theory_pos = theory_map<0 #note that theory_ising orientation is opposite from sign of CD signal
    theory_neg = theory_map>0
    theory_map_scaled = theory_neg*scaling_factor[0]+theory_pos*scaling_factor[1]
all_g_factors = np.concatenate((map3.flatten(),map2.flatten()))

num_factors = np.size(all_g_factors)
pos_g = all_g_factors>0
neg_g = all_g_factors<0

pos_g_factors = all_g_factors[pos_g]
neg_g_factors = all_g_factors[neg_g]

num_factors_theory = np.size(theory_map)
pos_g_theory = theory_map>0
neg_g_theory = theory_map<0

pos_g_factors_theory = theory_map[pos_g_theory]
neg_g_factors_theory = theory_map[neg_g_theory]

if (g_type == "g_trans"): cd_bounds = [-.1,.1]
elif g_type== "cd": cd_bounds = [-2000,2000]

fig, ax = plt.subplots(1,2,figsize=(7.08,5))

ax[0].matshow(map3,cmap = "bwr",vmin = cd_bounds[0],vmax = cd_bounds[1])
image = ax[1].matshow(theory_map_scaled,cmap = "bwr",vmin = cd_bounds[0],vmax = cd_bounds[1])

for i in range(2):
    ax[i].get_xaxis().set_ticks([])
    ax[i].get_yaxis().set_ticks([])

cax = fig.add_axes([0.2, 0.15, 0.6, 0.04])
cbar = fig.colorbar(mappable=image,cax =cax,orientation = 'horizontal')
cbar.set_label(g_str,fontsize = 7)

sub_tit_str = ["a","b"]
y_list = [.88,.88,.92]
for i in range(2):
    ax[i].text(.05,y_list[i],sub_tit_str[i],transform=ax[i].transAxes,fontsize=7,weight = "bold")

fig.savefig("figure_theory_exp_comparison_v3.png",dpi=500)
fig.show()

make_source_files = True
if (make_source_files):
    source_path = os.sep.join((os.getcwd(), "Source_Files"))
    source_dir = os.sep.join((source_path, "Fig_S12"))
    os.makedirs(source_dir, exist_ok=True)
    maps= [map3,theory_map_scaled]
    excel_writer = pd.ExcelWriter(os.sep.join((source_dir,"spatial_maps.xlsx")))
    sheet_names = ["exp","theory"]
    for i in np.arange(len(maps)):
        df = pd.DataFrame(maps[i])
        df.to_excel(excel_writer,sheet_name=sheet_names[i])
    excel_writer.close()