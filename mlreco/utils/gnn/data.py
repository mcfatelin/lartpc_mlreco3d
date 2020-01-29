# Defines inputs to the GNN networks
import numpy as np
import scipy as sp
from mlreco.utils.gnn.cluster import get_cluster_voxels, get_cluster_features, get_cluster_dirs, group_bipartite

def cluster_vtx_features(data, clusts, delta=0.0):
    """
    Function that returns the an array of 16 features for
    each of the clusters in the provided list.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        delta (float)        : Orientation matrix regularization
    Returns:
        np.ndarray: (C,16) tensor of cluster features (center, orientation, direction, size)
    """
    return get_cluster_features(data, clusts, delta)


def cluster_vtx_dirs(data, cs, delta=0.0):
    """
    Function that returns the direction of the listed clusters,
    expressed as its normalized covariance matrix.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        delta (float)          : Orientation matrix regularization
    Returns:
        np.ndarray: (C,9) tensor of cluster directions
    """
    return get_cluster_dirs(data, clusts, delta)


def cluster_edge_dir(data, c1, c2):
    """
    Function that returns the edge direction between for a
    given pair of connected clusters.

    Args:
        data (np.ndarray): (N,8) [x, y, z, batchid, value, id, groupid, shape]
        c1 (np.ndarray)  : (M1) Array of voxel IDs associated with the first cluster
        c2 (np.ndarray)  : (M2) Array of voxel IDs associated with the second cluster
    Returns:
        np.ndarray: (10) Array of edge direction (orientation, distance)
    """
    from scipy.spatial.distance import cdist
    x1 = get_cluster_voxels(data, c1)
    x2 = get_cluster_voxels(data, c2)
    d12 = cdist(x1, x2)
    imin = np.argmin(d12)
    i1, i2 = np.unravel_index(imin, d12.shape)
    v1 = x1[i1,:] # closest point in c1
    v2 = x2[i2,:] # closest point in c2
    disp = v1 - v2 # displacement
    lend = np.linalg.norm(disp) # length of displacement
    if lend > 0:
        disp = disp / lend
    B = np.outer(disp, disp).flatten()

    return np.concatenate([B, [lend]])


def cluster_edge_dirs(data, clusts, edge_index):
    """
    Function that returns a tensor of edge directions for each of the
    edges in the graph.

    Args:
        data (np.ndarray)      : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray])  : (C) List of arrays of voxel IDs in each cluster
        edge_index (np.ndarray): (2,E) Incidence matrix
    Returns:
        np.ndarray: (E,10) Tensor of edge directions (orientation, distance)
    """
    return np.vstack([cluster_edge_dir(data, clusts[e[0]], clusts[e[1]]) for e in edge_index.T])


def cluster_edge_feature(data, c1, c2):
    """
    Function that returns the edge features for a
    given pair of connected clusters.

    Args:
        data (np.ndarray): (N,8) [x, y, z, batchid, value, id, groupid, shape]
        c1 (np.ndarray)  : (M1) Array of voxel IDs associated with the first cluster
        c2 (np.ndarray)  : (M2) Array of voxel IDs associated with the second cluster
    Returns:
        np.ndarray: (19) Array of edge features (point1, point2, displacement, distance, orientation)
    """
    from scipy.spatial.distance import cdist
    x1 = get_cluster_voxels(data, c1)
    x2 = get_cluster_voxels(data, c2)
    d12 = cdist(x1, x2)
    imin = np.argmin(d12)
    i1, i2 = np.unravel_index(imin, d12.shape)
    v1 = x1[i1,:] # closest point in c1
    v2 = x2[i2,:] # closest point in c2
    disp = v1 - v2 # displacement
    lend = np.linalg.norm(disp) # length of displacement
    if lend > 0:
        disp = disp / lend
    B = np.outer(disp, disp).flatten()
    return np.concatenate([v1, v2, disp, [lend], B])


def cluster_edge_features(data, clusts, edge_index):
    """
    Function that returns a tensor of edge features for each of the
    edges connecting clusters in the graph.

    Args:
        data (np.ndarray)      : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray])  : (C) List of arrays of voxel IDs in each cluster
        edge_index (np.ndarray): (2,E) Incidence matrix
    Returns:
        np.ndarray: (E,19) Tensor of edge features (point1, point2, displacement, distance, orientation)
    """
    return np.vstack([cluster_edge_feature(data, clusts[e[0]], clusts[e[1]]) for e in edge_index.T])


def edge_feature(data, i, j):
    """
    Function that returns the edge features for a
    given pair of connected voxels.

    Args:
        data (np.ndarray): (N,8) [x, y, z, batchid, value, id, groupid, shape]
        i (int)            : Index of the first voxel
        j (int)            : Index of the second voxel
    Returns:
        np.ndarray: (12) Array of edge features (displacement, orientation)
    """
    xi = data[i,:3]
    xj = data[j,:3]
    disp = xj - xi
    B = np.outer(disp, disp).flatten()
    return np.concatenate([B, disp])


def cluster_edge_features(data, edge_index):
    """
    Function that returns a tensor of edge features for each of the
    edges connecting voxels in the graph.

    Args:
        data (np.ndarray)      : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        edge_index (np.ndarray): (2,E) Incidence matrix
    Returns:
        np.ndarray: (E,12) Tensor of edge features (displacement, orientation)
    """
    return np.vstack([edge_feature(data, e[0], e[1]) for e in edge_index.T])

def edge_assignment(edge_index, batches, groups, cuda=True, dtype=torch.float, binary=False, device=None):
    """
    edge assignment as same group/different group
    
    inputs:
    edge_index: torch tensor of edges
    batches: torch tensor of batch id for each node
    groups: torch tensor of group ids for each node
    """
    if isinstance(batches, torch.Tensor):
        batches = batches.cpu().detach().numpy()
    if isinstance(groups, torch.Tensor):
        groups = groups.cpu().detach().numpy()
    edge_assn = torch.tensor([np.logical_and(
        batches[edge_index[0,k]] == batches[edge_index[1,k]],
        groups[edge_index[0,k]] == groups[edge_index[1,k]]) for k in range(edge_index.shape[1])], 
                             dtype=dtype, requires_grad=False)
    if binary:
        # transform to -1,+1 instead of 0,1
        edge_assn = 2*edge_assn - 1
    if not device is None:
        edge_assn = edge_assn.to(device)
    elif cuda:
        edge_assn = edge_assn.cuda()
    return edge_assn
    

def get_input_node_features(data, batch_ids, group_ids, cuda=True, device=None):
    '''
    Function to return an array (G, F_n)
    The function is necessary because the batch and group ids in group-wise data (G, 2+F_n) are not necessarily the same as batch_ids and group_ids, which are generated in class internal function.
    Inputs:
        - data: (G, 2+F_n) tensor with batch_id, group id, and input node features (size F_n), G is number of groups
        - batch_ids: group-wise batch ids
        - group_ids: group-wise group ids, unique group shall have unique (batch_id, group_id)
    Outputs:
        - node_features: tensor (G, F_n) group-wise
    Note:
        - We assume there is no ambiguity in data batch and group ids, meaning data[:, 0:2] has no unique list (axis=0) equal to itself
    '''
    # put data into numpy array
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    # check if ids in data include all ids in batch_ids and group_ids
    if not np.equal(
        np.unique(np.isin(batch_ids, data[:, 0])),
        np.asarray([True]),
    ):
        raise ValueError('batch_ids does not match ones in input data structure!')
    if not np.equal(
        np.unique(np.isin(group_ids, data[:, 1])),
        np.asarray([True]),
    ):
        raise ValueError('group_ids does not match ones in input data structure!')

    # Loop over (batch_ids, group_ids)
    node_features = []
    for batch_id, group_id in zip(batch_ids, group_ids):
        # select the index for data batch and group id matches the input one
        selection = np.logical_and(
            data[:,0]==batch_id,
            data[:,1]==group_id
        )
        # get the feature array
        feature_array = data[np.where(selection)[0],2:]
        # check if there's only one feature_vector left
        if feature_array.shape[0]>1:
            raise ValueError('There is ambiguity in input data for node features!')
        # append
        node_features.append(feature_array[0])

    # return
    node_features = torch.tensor(node_features, dtype=torch.float, requires_grad=False)
    if device is not None:
        node_features = node_features.to(device)
    elif cuda:
        node_features = node_features.cuda()
    return node_features

def get_input_edge_features(data, batch_ids, group_ids, cuda=True, device=None):
    '''
    Function to return an array (sum_B (N_i*(N_i+1)/2) , F_e) and edge_index (2, sum_B (N_i*(N_i+1)/2) )
    G is the number of groups.
    sum_B (N_i*(N_i+1)/2) is the total number of edges that can be possible between two nodes. Nodes with different batch id cannot be connected
    The function is necessary because the batch and group ids in group-wise data (G, 2+F_n) are not necessarily the same as batch_ids and group_ids, which are generated in class internal function.
    Inputs:
        - data: (G, 3+F_e) tensor with batch_id, group id 1, group id 2, and input node features (size F_e), G is number of groups, not necesarily same as the following two
        - batch_ids: group-wise batch ids (G',)
        - group_ids: group-wise group ids (G',), unique group shall have unique (batch_id, group_id)
    Outputs:
        - edge_features: tensor (G', F_e) group-wise
    Note:
        - We assume there is no ambiguity in data batch and group ids, meaning data[:, 0:2] has no unique list (axis=0) equal to itself
    '''
    # put data into numpy array
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    # get the edge_index from network.group_bipartite so that it is consistent with other edge feature modes of InteractionModel. Shape is (2, -1)
    edge_group_ids = group_bipartite(batch_ids, group_ids, cuda=False, return_id=True)
    # check if the (group_id_1, group_id_2) contains all the edge_group_ids
    if not np.equal(
        np.logical_and(
            np.logical_and(
                np.isin(edge_group_ids[0,:], data[:,0]),
                np.isin(edge_group_ids[1,:], data[:,1])
            ),
            np.isin(edge_group_ids[2,:], data[:,2]),
        ),
        np.asarray([True])
    ):
        raise ValueError(
            'Input edge feature does not contain feature for some group id pairs!'
        )

    # loop over edge_group_ids to extract edge_features
    output_edge_features = []
    for batch_id, group_id_1, group_id_2 in edge_group_ids.T:
        # select the matched the group id pair in data
        selection = np.logical_and(
            np.logical_and(
                data[:,0]==batch_id,
                data[:,1]==group_id_1
            ),
            data[:,2]==group_id_2,
        )
        # get the feature array
        feature_array = data[np.where(selection)[0], 3:]
        # check if there's only one feature_vector left
        if feature_array.shape[0] > 1:
            raise ValueError('There is ambiguity in input data for edge features!')
        # append
        output_edge_features.append(feature_array[0])

    # return
    output_edge_features = torch.tensor(output_edge_features, dtype=torch.float, requires_grad=False)
    if device is not None:
        output_edge_features = output_edge_features.to(device)
    elif cuda:
        output_edge_features = output_edge_features.cuda()
    return output_edge_features, group_bipartite(batch_ids, group_ids, device=device)
