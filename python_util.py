import numpy as np
import os
import re
import csv
import pickle
'''
Utility functions, primarily for numpy and os operations 
'''

def get_index_in_list_containing_substring(list_to_check,substr):
    '''
    Provides first instance in a list of substring. Returns None if substring cannot be found
    :param the_list:
    :param substr:
    :return:
    '''
    for i in np.arange(len(list_to_check)):
        if (substr in list_to_check[i]): return i
    return None

def get_2d_index(total_index,col):
    '''Gives the modulus and remainder, corresponding to i,j for a 2D ndarray'''
    mod = np.floor(total_index/col)
    rem = total_index-col*mod
    return mod, rem

def get_2d_index_array(total_index_array,col):
    '''Gives the modulus and remainder, corresponding to i,j for a 2D ndarray as an array'''
    mod = np.floor(total_index_array/col)
    rem = total_index_array-col*mod
    return np.vstack((mod,rem))


def get_3d_indices(total_index,y,z):
    '''Gives the three indices i,j,k for a 3D ndarray corresnding to a total index
    y and z are the sizes of the final two indices'''
    a = np.floor(total_index/(y*z))
    r1 = total_index-a*(y*z)
    b,c  = get_2d_index(r1,z)
    return a,b,c
def remove_unnecessary_indices(array):
    '''alternate call for np.squeeze()'''
    return np.squeeze(array)

def norm_array(array):
    '''Normalizes an array. If array is all 0, does nothing to array and returns it'''
    if (np.max(np.abs(array)) == 0):return array
    else: return array/(np.max(np.abs(array)))

def tuple_to_array(*params):
    '''Converts some tuple to a 1D ndarray'''
    return np.array(params).flatten()

def arb_to_array(arb):
    '''takes scalar, array, or list and returns ndarray'''
    if (type(arb)==np.ndarray):return arb
    elif(np.isscalar(arb)):return np.array([arb])
    elif(type(arb)==list):return np.array(arb)
    else: raise ValueError("Input must be scalar, list, or array")
def interweave_arrays(array_1,array_2):
    '''
    Interweaves two arrays that share size and datatype
    :param array_1:
    :param array_2:
    :return:
    '''
    if (np.ndim(array_1)!= 1 or np.ndim(array_2)!= 1): raise ValueError("Array dimensions must be 1")
    if (np.size(array_1)!= np.size(array_2)):raise ValueError("Arrays must be same size to be interwoven")
    else: interweaved_size = np.size(array_1)+np.size(array_2)
    if (array_1.dtype != array_2.dtype):raise ValueError("Arrays must have save datatype")
    else: array_dtype = array_2.dtype
    interweaved_array = np.empty(shape = (interweaved_size),dtype= array_dtype)
    interweaved_array[0::2] = array_1
    interweaved_array[1::2] = array_2
    return interweaved_array

def filename_handling(figure,filename,dpi = None,to_show = True):
    if (filename):
        figure.savefig(filename,dpi = dpi)
        figure.show()
    else:
        if (to_show):figure.show()

def array_and_scalar_selector(array_or_scalar,index):
    if (np.isscalar(array_or_scalar)):return array_or_scalar
    else:return array_or_scalar[index]

def repeat_array_along_new_axis(array,repeats,axis = 0):
    new_array = np.expand_dims(array,axis = axis)
    return np.repeat(new_array,repeats,axis = axis)


def save_object(object,filename):
    '''
    :param object: object
    :param filename: str
    :return: None
    '''
    return pickle.dump(object,open(filename,"wb"),pickle.HIGHEST_PROTOCOL)

def open_object_file(filename):
    '''
    :param filename: str
    :return: object
    '''
    return pickle.load(open(filename,'rb'))

def clean_up_string(string):
    '''
    Takes arbitrary string and returns one without spaces and in lowercase
    :param string: str
    :return: str
    '''
    return re.sub(" ","",string).lower()


def convert_folder_to_csv(directory = None,init_file_extension = ".asc",run_subdirs = False):
    '''
    Converts all files in a folder with a given extension (default asc) to csv
    :param directory: str
    :param init_file_extension: str (default = ".asc")
    :param run_subdirs: bool (default = False)
         Does not run on subsubdirectories
    :return: None
    '''
    if (directory is None): base_folder,csv_folder = os.getcwd(),os.getcwd()
    else: base_folder,csv_folder = directory, directory
    root, dirs, filenames = os.walk(base_folder)
    for filename_with_extension in filenames:
        filename, file_extension = os.path.splitext(filename_with_extension)
        if file_extension.lower() == init_file_extension.lower():
            init_file = open(os.sep.join((base_folder, filename_with_extension)), 'r')
            csv_file = open(os.sep.join((csv_folder, filename + ".csv")), "w", newline='')
            csv_writer = csv.writer(csv_file, delimiter=',')
            for line in init_file: csv_writer.writerow(line.strip().split())
    if (run_subdirs):
        for subdir in dirs: convert_folder_to_csv(os.sep.join((base_folder,subdir)),init_file_extension,run_subdirs)


