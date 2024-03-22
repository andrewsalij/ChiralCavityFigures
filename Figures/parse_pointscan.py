import pandas as pd
import os
import copy
import numpy as np
data_dir = os.sep.join((os.getcwd(),"MuellerMat_Data"))
#DATA LOADING

'''This Excel file is formatted in a manner acceptable for a source file (.xslx, has labelled columns)'''
pointscan_df = pd.read_excel(os.sep.join((data_dir,"121pointscan.xlsx")))

column_labels = copy.copy(pointscan_df.loc[1])
pointscan_df.rename(axis = 'columns',mapper = column_labels,inplace = True)
#see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.array.html#pandas.Index.array
pointscan_df.columns.array[0] = "info_label"
#0 is nan, 1 is the column labels
pointscan_df.drop([0,1],inplace=True)

mueller_strs = np.array([["m11","m12","m13","m14"],["m21","m22","m23","m24"],["m31","m32","m33","m34"],["m41","m42","m43","m44"]])
mueller_err_strs = np.array([["m11_err","m12_err","m13_err","m14_err"],["m21_err","m22_err","m23_err","m24_err"],["m31_err","m32_err","m33_err","m34_err"],["m41_err","m42_err","m43_err","m44_err"]])

n_cols = len(pointscan_df.columns)
n_wvl = n_cols - 4

wvl_array = np.array(pointscan_df.columns[4:].to_list())

mueller_shape = (121,4,4,np.size(wvl_array)+2)  # ((X,Y)),m_i0, m0j, wvl+(X+Y))

mueller_array = np.zeros(mueller_shape,dtype = np.float64)
mueller_err_array = np.zeros(mueller_shape,dtype = np.float64)

for i in np.arange(4):
    for j in np.arange(4):
        selected_mm_array = pointscan_df.loc[pointscan_df['info_label']==mueller_strs[i,j]].to_numpy()
        selected_err_array = pointscan_df.loc[pointscan_df['info_label']==mueller_err_strs[i,j]].to_numpy()
        mueller_array[:,i,j,:] = selected_mm_array[:,2:]
        mueller_err_array[:,i,j,:] = selected_err_array[:,2:]

np.save("full_mueller_array.npy",mueller_array)
np.save("full_mueller_err_array.npy",mueller_err_array)
np.save("mueller_wvl_array.npy",wvl_array)


