import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
import os
import berreman_mueller as bm


plt.rcParams.update({'font.size': 7,'axes.unicode_minus':False})
plot_style = "cd_mdeg"

cwd = os.getcwd()
base_dir,cur_folder = os.path.split(cwd)
theory_data_path = os.sep.join((base_dir,"PTPO_Theory"))
exp_data_path =os.sep.join((base_dir,r"Experimental Cavity Data\angle-map-raw-data.xlsx"))

map3_path = os.sep.join((base_dir,r"\Experimental Cavity Data\u27_map3_front_data"))
map2_path = os.sep.join((base_dir,r"\Experimental Cavity Data\u27_map2_front_data"))

map2_t_l = np.load(os.sep.join((map2_path,"map_2_front_tl.npy")))
map2_t_r = np.load(os.sep.join((map2_path,"map_2_front_tr.npy")))

map3_t_l = np.load(os.sep.join((map3_path,"map_3_front_tl.npy")))
map3_t_r = np.load(os.sep.join((map3_path,"map_3_front_tr.npy")))

theory_map = np.load("ising_model_spin_mesh_si.npy")

ising_results = np.load("metropolis_100transits_results.npy")
averaged_ising_results = np.average(ising_results,axis=1) #negative spin,positive spin, average spi
std_ising_results = np.std(ising_results,axis=1)
if (plot_style =="cd_mdeg"):
    # ptpo_angle_resolved.py, running np.min(cd[20:,5])
    # note that sign convention changes
    scaling_factor = -768.79

    map2 = 32980*np.log10(map2_t_r/map2_t_l)
    map3 = 32980*np.log10(map3_t_r/map3_t_l)

    disymmetry_label = "CD (mdeg)"
elif (plot_style == "g_trans"):
    #from SMM 420-520 nm of PTPO 300 nm (input_params v6)
    #ptpo_angle_resolved.py , running np.max(g[20:,5])
    scaling_factor = 0.05366216778162327
    map2 = 2*(map2_t_l-map2_t_r)/(map2_t_r+map2_t_l)
    map3 =  2*(map3_t_l-map3_t_r)/(map3_t_r+map3_t_l)
    disymmetry_label = r"$g$ factor"

map2 = bm.slice_max_abs_val(map2)
map3 = bm.slice_max_abs_val(map3)

theory_map = theory_map*scaling_factor

all_g_factors = np.vstack((map3.flatten(),map2.flatten()))

num_factors = np.size(all_g_factors,axis=1)

pos_g_factors = np.where(all_g_factors>0,1,0)
neg_g_factors = np.where(all_g_factors<0,1,0)
pos_g_factors_values = np.where(all_g_factors>0,all_g_factors,np.nan)
neg_g_factors_values = np.where(all_g_factors<0,all_g_factors,np.nan)
percent_pos_g_factors = np.sum(pos_g_factors,axis=1)/num_factors
percent_neg_g_factors = np.sum(neg_g_factors,axis=1)/num_factors

avg_percent_pos_g_factors = np.average(percent_pos_g_factors)
avg_percent_neg_g_factors = np.average(percent_neg_g_factors)

std_pos_g = np.std(percent_pos_g_factors)
std_neg_g = np.std(percent_neg_g_factors)

avg_exp_factors = np.array([np.mean(neg_g_factors),np.mean(pos_g_factors)])
percent_exp_factors = np.array([avg_percent_neg_g_factors,avg_percent_pos_g_factors])
std_exp_factors = np.array([std_neg_g,std_pos_g])

if (plot_style == "g_trans"):bins = list(np.linspace(-.16,.16,37))
elif(plot_style=="cd_mdeg"):bins=  list(np.linspace(-2000,2000,41))

fig= plt.figure(figsize = (7.08,6))
gs = fig.add_gridspec(20, 20,
                      left=0.04, right=0.98, bottom=0.13, top=0.96,
                      wspace=0, hspace=0.02)
ax1 = fig.add_subplot(gs[:9,:9])
ax2 = fig.add_subplot(gs[:9,8:17])
ax3 = fig.add_subplot(gs[10:,1:12])
ax4 = fig.add_subplot(gs[10:,13:18])

if (plot_style == "g_trans"):
    cd_bounds = [-.1,.1]
elif (plot_style == "cd_mdeg"):
    cd_bounds = [-2000,2000]

ax1.matshow(map3,cmap = "bwr",vmin = cd_bounds[0],vmax = cd_bounds[1])
image = ax2.matshow(map2,cmap = "bwr",vmin = cd_bounds[0],vmax = cd_bounds[1])

ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])
ax2.get_xaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])

cax = fig.add_axes([0.84, 0.59, 0.04, 0.32])
cbar = fig.colorbar(mappable=image,cax =cax,orientation = 'vertical')
fig.text(.82,.93,disymmetry_label,fontsize =7)
ax3.hist(pos_g_factors_values.flatten(),alpha = .7,color = "red",weights=np.ones(np.size(pos_g_factors))/np.size(all_g_factors),bins = bins,label = "Positive")
ax3.hist(neg_g_factors_values.flatten(),alpha = .7,color = "blue",weights=np.ones(np.size(neg_g_factors))/np.size(all_g_factors),bins = bins,label ="Negative")

if (plot_style == "g_trans"):
    bar_width = .01
    bar_plot_order = [-1,0]
if (plot_style == "cd_mdeg"):
    bar_width = .07
    bar_plot_order = [0,1]
ax4.set_xlabel("CD Sign",fontsize=7)
index  =np.array([-.1,.1])
error_config = {'ecolor': '0.3','capsize':8}
ax4.bar(index,percent_exp_factors,label = "Exp.",color = "green",width=bar_width,yerr= std_exp_factors,error_kw=error_config)
ax4.bar(index+bar_width,averaged_ising_results[:2],label = "Ising",color = "orange",width = bar_width,yerr= std_ising_results[:2],
        error_kw=error_config)
ax4.set_xticks(index+bar_width/2)
ax4.set_xticklabels(["Negative","Positive"])

ax3.set_xlabel(disymmetry_label,fontsize=7)
if (plot_style == "g_trans"):
    ax3.set_xlim(-.14,.14)
elif (plot_style == "cd_mdeg"):ax3.set_xlim(-2000,2000)
ax3.yaxis.set_major_formatter(PercentFormatter(1))

ax4.yaxis.set_major_formatter(PercentFormatter(1))
ax4.set_ylim(0,1)
ax4.legend(loc = "upper left",bbox_to_anchor = (0,.85))
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()

sub_tit_str = ["a","b","c","d"]
axes_list = [ax1,ax2,ax3,ax4]
x_list = [.05,.05,.05,.05]
y_list = [.92,.92,.92,.92]
for i in range(4):
    cur_ax = axes_list[i]
    cur_ax.text(x_list[i],y_list[i],sub_tit_str[i],transform=cur_ax.transAxes,fontsize=7,weight= "bold")

fig.savefig("figure_spatial_comparisonv5.pdf")
fig.show()

neg_avg = np.average(np.ma.masked_array(neg_g_factors_values,np.isnan(neg_g_factors_values)))
pos_avg = np.average(np.ma.masked_array(pos_g_factors_values,np.isnan(pos_g_factors_values)))
print((neg_avg,pos_avg))

make_source_files = True
if (make_source_files):
    source_path = os.sep.join((os.getcwd(), "Source_Files"))
    source_dir = os.sep.join((source_path, "Fig_4"))
    os.makedirs(source_dir, exist_ok=True)

    excel_writer = pd.ExcelWriter(os.sep.join((source_path,"cd_spatial_maps.xlsx")))
    maps = [map3,map2]
    sheet_names = ["a","b"]
    for i in np.arange(len(maps)):
        cur_df = pd.DataFrame(maps[i])
        cur_df.to_excel(excel_writer,sheet_name=sheet_names[i])
    excel_writer.close()
    np.savetxt("all_cd_values.csv",all_g_factors)
    bar_data = np.column_stack((percent_exp_factors,averaged_ising_results[:2],std_exp_factors,std_ising_results[:2]))
    bar_df = pd.DataFrame(bar_data,columns = ["Exp.","Ising","Exp. Std.","Ising Std."])
    bar_df.to_csv("bar_data.csv",index = False)


