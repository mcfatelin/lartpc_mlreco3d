# creates inputs to GNN networks
import torch
from mlreco.utils.gnn.cluster import get_cluster_centers, get_cluster_voxels, get_cluster_features, get_cluster_energies, get_cluster_dirs
import numpy as np
import scipy as sp
from mlreco.utils.gnn.groups import group_bipartite


# TODO: add vertex orientation information


def cluster_vtx_features_old(data, cs, device=None, cuda=True):
    """
    Cluster vertex features - center
    returned as a pytorch tensor of size (n_clusts, 3)
    optional flag to put features on gpu
    """
    f = torch.tensor(np.concatenate((get_cluster_centers(data, cs), (get_cluster_energies(data, cs)).reshape(-1,1)), axis=1), dtype=torch.float, requires_grad=False)
    if not device is None:
        f = f.to(device)
    elif cuda:
        f = f.cuda()
    return f


def cluster_vtx_features(data, cs, cuda=True, device=None):
    """
    Cluster vertex features - center, orientation, and direction
    returned as pytorch tensor of size (n_clusts, 15)
    optional flag to put features on gpu
    """
    f = torch.tensor(get_cluster_features(data, cs), dtype=torch.float, requires_grad=False)
    if not device is None:
        f = f.to(device)
    elif cuda:
        f = f.cuda()
    return f


def cluster_vtx_dirs(data, cs, cuda=True, device=None, delta=0.0):
    """
    Cluster directions - vectorized 3x3 matrices of normalized principal vector
    """
    f = torch.tensor(get_cluster_dirs(data, cs, delta=delta), dtype=torch.float, requires_grad=False)
    if not device is None:
        f = f.to(device)
    elif cuda:
        f = f.cuda()
    return f


def cluster_edge_dir(data, c1, c2):
    """
    feature that includes closest points, displacement, and length of edge
    """
    x1 = get_cluster_voxels(data, c1)
    x2 = get_cluster_voxels(data, c2)
    d12 = sp.spatial.distance.cdist(x1, x2,'euclidean')
    imin = np.argmin(d12)
    i1, i2 = np.unravel_index(imin, d12.shape)
    v1 = x1[i1,:] # closest point in c1
    v2 = x2[i2,:] # closest point in c2
    disp = v1 - v2 # displacement
    lend = np.linalg.norm(disp)
    if lend > 0:
        disp = disp / lend
    B = np.outer(disp, disp)
    out = B.flatten()
    out = np.append(out, lend)
    return out


def cluster_edge_dirs(data, clusts, edge_index, cuda=True, device=None):
    """
    Cluster edge features
    returned as a pytorch tensor of size (n_edges, 9)
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    e = torch.tensor([cluster_edge_dir(data, clusts[edge_index[0,k]], clusts[edge_index[1,k]]) for k in range(edge_index.shape[1])], dtype=torch.float, requires_grad=False)
    if not device is None:
        e = e.to(device)
    elif cuda:
        e = e.cuda()
    return e


def cluster_edge_feature(data, c1, c2):
    """
    feature that includes closest points, displacement, and length of edge
    """
    x1 = get_cluster_voxels(data, c1)
    x2 = get_cluster_voxels(data, c2)
    d12 = sp.spatial.distance.cdist(x1, x2,'euclidean')
    imin = np.argmin(d12)
    i1, i2 = np.unravel_index(imin, d12.shape)
    v1 = x1[i1,:] # closest point in c1
    v2 = x2[i2,:] # closest point in c2
    out = np.append(v1, v2)
    disp = v1 - v2 # displacement
    out = np.append(out, disp)
    lend = np.linalg.norm(disp) # length of displacement
    out = np.append(out, lend)
    if lend > 0:
        disp = disp / lend
    out = np.append(out, np.outer(disp, disp).flatten())
    return out


def cluster_edge_features(data, clusts, edge_index, cuda=True, device=None):
    """
    Cluster edge features
    returned as a pytorch tensor of size (n_edges, 10)
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    e = torch.tensor([cluster_edge_feature(data, clusts[edge_index[0,k]], clusts[edge_index[1,k]]) for k in range(edge_index.shape[1])], dtype=torch.float, requires_grad=False)
    if not device is None:
        e = e.to(device)
    elif cuda:
        e = e.cuda()
    return e

def edge_feature(data, i, j):
    """
    12-dimensional edge feature based on displacement between two voxels
    """
    xi = data[i,:3].flatten()
    xj = data[j,:3].flatten()
    disp = xj - xi
    out = np.outer(disp, disp).flatten()
    out = np.append(out, disp)
    return out


def edge_features(data, edge_index, cuda=True, device=None):
    """
    produce features for edges between single voxels
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    e = torch.tensor([edge_feature(data, edge_index[0,k], edge_index[1,k]) for k in range(edge_index.shape[1])], dtype=torch.float, requires_grad=False)
    if not device is None:
        e = e.to(device)
    elif cuda:
        e = e.cuda()
    return e

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
    return output_edge_features
