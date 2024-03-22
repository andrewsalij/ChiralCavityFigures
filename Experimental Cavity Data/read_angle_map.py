import numpy as np
import pandas as pd
import berreman_mueller as bm
backside_g_full = pd.read_excel("angle-map-raw-data.xlsx",sheet_name="u35spot2bs")
frontside_g_full = pd.read_excel("angle-map-raw-data.xlsx",sheet_name="u35spot2fs")

backside_g_full = np.array(backside_g_full)
frontside_g_full = np.array(frontside_g_full)

backside_g = backside_g_full[:,2:]
frontside_g = frontside_g_full[:,2:]

energy_array = backside_g_full[:,1]
angle_set_array = np.linspace(-21,21,15)

bm.plot_heatmap_spectra(angle_set_array,energy_array,backside_g)
bm.plot_heatmap_spectra(angle_set_array,energy_array,frontside_g)



