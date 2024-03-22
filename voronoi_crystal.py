import numpy as np
import scipy.spatial as spatial
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
'''Handling for the construction and manipulation of region maps created as voronoi diagrams as well as
Ising models and Metropolis-Hastings on them '''

def get_voronoi_points(n_points,x_array,y_array):
    '''
    Creates uniform voronoi points within bounds given by x and y arrays
    :param n_points:
    :param x_array:
    :param y_array:
    :return:
    '''
    x_random = np.random.uniform(np.min(x_array),np.max(x_array),n_points)
    y_random = np.random.uniform(np.min(y_array),np.max(y_array),n_points)
    return np.vstack((x_random,y_random)).T
def partition_mesh_into_voronoi(x_mesh,y_mesh,voronoi_points, p_norm =1):
    '''
    Using scipy's KDTree, partitions a mesh given by an x_mesh and y_mesh
    p_norm is the Minkowski p norm used (default : 1)
    :param x_mesh:
    :param y_mesh:
    :param voronoi_points:
    :param p_norm:
    :return:
    '''
    grid_points = np.vstack((x_mesh.flatten(),y_mesh.flatten())).T
    grid_point_distances, grid_point_indices = spatial.KDTree(voronoi_points).query(grid_points,p = p_norm)
    return grid_point_distances, grid_point_indices

def get_adjacent_regions_and_edge_lengths(grid_point_region_mesh,region_index):
    '''
    Provides values of the indices of regions (given by grid_point_region_indices, a 2D matrix)
    that are adjacent to some target index (given by region_index)
    :param grid_point_region_indices: np.ndarray (2D)
    :param region_index: int
    :return: np.ndarray (1D)
    '''
    grid_point_reigon_flat = grid_point_region_mesh.flatten()
    selected_region = np.where(grid_point_region_mesh==region_index,1,0)
    shift_amounts = np.array([[-1,0],[1,0],[0,-1],[0,1]])
    adjacent_regions= []
    for i in np.arange(4): #shifts along all directions, checking for adjacencies
        shifted_region = ndimage.shift(selected_region,shift=shift_amounts[i,:],order=1)
        edge = np.logical_and(np.where(selected_region==0,True,False),np.where(shifted_region==1,True,False))
        edge_region_indices = np.unique(grid_point_reigon_flat[edge.flatten()])
        adjacent_regions.extend(edge_region_indices.tolist())
    adjacent_regions = np.array(adjacent_regions)
    adjacent_regions = np.sort(np.unique(adjacent_regions)) #remove duplicates
    edge_lengths = np.zeros(np.size(adjacent_regions))
    for i in np.arange(np.size(adjacent_regions)):
        selected_adjacent_region = np.where(grid_point_region_mesh==adjacent_regions[i],1,0)
        edge_length = 0
        for j in np.arange(4):
            shifted_region = ndimage.shift(selected_region, shift=shift_amounts[j, :], order=1)
            adjacent_region_edge = np.logical_and(np.where(selected_adjacent_region==1,True,False), np.where(shifted_region==1,True,False))
            edge_length = edge_length+np.sum(adjacent_region_edge)
        edge_lengths[i] = edge_length
    return adjacent_regions, edge_lengths

def get_region_adjacency_matrix(grid_point_region_mesh,weight_edge_length = True):
    '''
    :param grid_point_region_mesh:
    :param weight_edge_length:
    :return:
    '''
    if (not weight_edge_length):
        dtype = int
    else: dtype = float
    if (int(np.min(grid_point_region_mesh))!= 0):
        raise ValueError("Grid point indexing does not begin at 0")
    num_regions = int(np.max(grid_point_region_mesh)+1) #assumes regions indexed 0-value
    region_adjacency_matrix = np.zeros((num_regions,num_regions),dtype =dtype)
    for i in np.arange(num_regions):
        adjacent_regions,adjacent_regions_edge_lengths = get_adjacent_regions_and_edge_lengths(grid_point_region_mesh,region_index=i)
        adjacent_regions_row = np.zeros(num_regions)
        if (not weight_edge_length):
            adjacent_regions_row[adjacent_regions] = 1
        else:
            adjacent_regions_row[adjacent_regions] = adjacent_regions_edge_lengths
        region_adjacency_matrix[i,:] = adjacent_regions_row
        region_adjacency_matrix[:,i] = adjacent_regions_row
    return region_adjacency_matrix

def get_region_sizes(grid_point_region_mesh,weight_region_size = True):
    '''
    :param grid_point_region_mesh:
    :param weight_region_size:
    :return:
    '''
    if (int(np.min(grid_point_region_mesh))!= 0):
        raise ValueError("Grid point indexing does not begin at 0")
    num_regions = int(np.max(grid_point_region_mesh) + 1)  # assumes regions indexed 0-value
    if (not weight_region_size): #if not weighting, returns uniform sizes
        return np.ones(num_regions)
    region_size_array = np.zeros(num_regions)
    for i in np.arange(num_regions):
        cur_region = grid_point_region_mesh==i
        cur_region_size = np.sum(cur_region)
        region_size_array[i] = cur_region_size
    return region_size_array

def hamiltonian_voronoi_ising(region_spin_array,region_adjacency_matrix,region_size_array,substrate_field =1,coupling_constant =1):
    '''
    Determines Hamiltonian for a potentially non-uniform Ising model characterized by an adjacency
    matrix w/ edge weights and an array of the respective sizes of the regions. Setting both to
    1 reconstructs the standard Ising model
    :param region_spin_array:
    :param region_adjacency_matrix:
    :param region_size_array:
    :param substrate_field:
    :param coupling_constant:
    :return:
    '''
    region_spin_matrix = np.outer(region_spin_array,region_spin_array)
    external_field_term = np.sum(region_spin_array*region_size_array*substrate_field)
    coupling_term = 1/2*np.sum(coupling_constant*region_adjacency_matrix*region_spin_matrix)
    total_energy = -coupling_term-external_field_term
    return total_energy


def get_random_spin_configuration(n):
    '''
    Provides n random spins of (-1,1)
    :param n: int
    :return:
    '''
    spin_config = np.random.randint(low= 0,high = 2,size = n)
    spin_config = np.where(spin_config == 1,spin_config,-1)
    return spin_config

def convert_regions_mesh_to_spin_mesh(grid_point_region_mesh,regions_array,regions_spin_array):
    '''
    Takes the mesh of regions numbered (0,... n-1) and applies a spin configuration (default (-1, 1)) to it
    :param grid_point_region_mesh:
    :param regions_array:
    :param regions_spin_array:
    :return:
    '''
    grid_point_spin_mesh = np.zeros(np.shape(grid_point_region_mesh))
    for i in np.arange(np.size(regions_array)):
        cur_region = grid_point_region_mesh==regions_array[i]
        grid_point_spin_mesh = grid_point_spin_mesh+cur_region*regions_spin_array[i]
    return grid_point_spin_mesh

def visualize_regions_mesh_spin(grid_point_region_mesh,regions_array,regions_spin_array):
    '''
    Visualizes a spin configuration on a voronoi diagram constructed from a mesh
    Does not save a figure
    :param grid_point_region_mesh:
    :param regions_array:
    :param regions_spin_array:
    :return:
    '''
    grid_point_spin_mesh = convert_regions_mesh_to_spin_mesh(grid_point_region_mesh,regions_array,regions_spin_array)
    plt.imshow(grid_point_spin_mesh,cmap="seismic",vmin = -1.5,vmax = 1.5)
    plt.show()


def metropolis_hastings_ising(grid_point_region_mesh,metropolis_steps,substrate_field,coupling_constant,effective_beta,uniform_region_graph = False,update_style = "random",
                              visualization_its = None,talk = False):
    '''
        Determines the spin configuration of a 2D Ising model on some mesh using the Metropolis Hastings algorithm
    Assumes Boltzmann statistics and a modified Ising hamiltonian (see hamiltonian_voronoi_ising())
    For a good introduction to the algorithm, see Kotze, J. (2008). Introduction to Monte Carlo methods for an Ising Model of a Ferromagnet. arXiv preprint arXiv:0803.0217.
    :param grid_point_region_mesh:
    :param metropolis_steps
    :param effective_beta
    :return:
    '''
    if (uniform_region_graph):
        weight_edge_length = False
        weight_region_size = False
    else:
        weight_edge_length = True
        weight_region_size = True
    if (int(np.min(grid_point_region_mesh))!= 0):
        raise ValueError("Grid points indexing does not begin at 0")
    num_regions = int(np.max(grid_point_region_mesh)+1)
    regions_array = np.arange(num_regions)
    adjacency_matrix = get_region_adjacency_matrix(grid_point_region_mesh,weight_edge_length=weight_edge_length)
    region_size_array = get_region_sizes(grid_point_region_mesh,weight_region_size=weight_region_size)

    spin_configuration = get_random_spin_configuration(num_regions)
    if (talk):percent_its = int(metropolis_steps/100)
    for i in np.arange(metropolis_steps):
        if (update_style=="random"):
            region_to_flip = np.random.randint(num_regions)
        elif (update_style =="sweep"):
            if (i == 0):
                region_counter = 0
            region_to_flip = regions_array[region_counter%num_regions]
            region_counter = region_counter+1
        new_spin_configuration = np.copy(spin_configuration)
        new_spin_configuration[region_to_flip] = -1*new_spin_configuration[region_to_flip]
        init_energy = hamiltonian_voronoi_ising(spin_configuration,adjacency_matrix,region_size_array,
                                                substrate_field=substrate_field,coupling_constant=coupling_constant)
        new_energy = hamiltonian_voronoi_ising(new_spin_configuration,adjacency_matrix,region_size_array,
                                                substrate_field=substrate_field,coupling_constant=coupling_constant)
        markov_weight = np.exp(-effective_beta*(new_energy-init_energy))
        random_weight = np.random.uniform()
        #decides whether or not to accept new config
        if (np.less_equal(random_weight,markov_weight)):
            spin_configuration = new_spin_configuration
        if talk:
            if (i%percent_its == 0):
                print(str(int(100*i/metropolis_steps))+"%")
        if (visualization_its is not None):
            if (i%visualization_its == 0):
                visualize_regions_mesh_spin(grid_point_region_mesh,regions_array,spin_configuration)
    return spin_configuration

def average_grid(grid_mesh,pixel_average):
    if np.size(grid_mesh,axis =0)%pixel_average !=0  or np.size(grid_mesh,axis =1)%pixel_average !=0 :
        raise ValueError("pixel size must divide mesh dimension")
    x_dim = int(np.size(grid_mesh,axis =0)/pixel_average)
    y_dim = int(np.size(grid_mesh,axis =1)/pixel_average)
    new_grid = np.zeros((x_dim,y_dim))
    for i in np.arange(x_dim):
        for j in np.arange(y_dim):
            cur_region = grid_mesh[i*pixel_average:(i+1)*pixel_average,j*pixel_average:(j+1)*pixel_average]
            new_grid[i,j] = np.average(cur_region)
    return new_grid