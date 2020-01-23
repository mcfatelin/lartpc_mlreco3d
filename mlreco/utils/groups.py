# utility function to reconcile groups data with energy deposition and 5-types data:
# problem: parse_cluster3d and parse_sparse3d will not output voxels in same order
# additionally, some voxels in groups data do not deposit energy, so do not appear in images
# also, some voxels have more than one group.
# plan is to put in a function to:
# 1) lexicographically sort group data (images are lexicographically sorted)
# 2) remove voxels from group data that are not in image
# 3) choose only one group per voxel (by lexicographic order)
# WARNING: (3) is certainly not a canonical choice

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode
import torch

def get_group_types(particle_v, meta, point_type="3d"):
    """
    Gets particle classes for voxel groups
    """
    if point_type not in ["3d", "xy", "yz", "zx"]:
        raise Exception("Point type not supported in PPN I/O.")
    # from larcv import larcv
    gt_types = []
    for particle in particle_v:
        pdg_code = abs(particle.pdg_code())
        prc = particle.creation_process()

        # Determine point type
        if (pdg_code == 2212):
            gt_type = 0 # proton
        elif pdg_code != 22 and pdg_code != 11:
            gt_type = 1
        elif pdg_code == 22:
            gt_type = 2
        else:
            if prc == "primary" or prc == "nCapture" or prc == "conv":
                gt_type = 2 # em shower
            elif prc == "muIoni" or prc == "hIoni":
                gt_type = 3 # delta
            elif prc == "muMinusCaptureAtRest" or prc == "muPlusCaptureAtRest" or prc == "Decay":
                gt_type = 4 # michel
            else:
                gt_type = -1 # not well defined

        gt_types.append(gt_type)

    return np.array(gt_types)


def filter_duplicate_voxels(data, usebatch=True):
    """
    return array that will filter out duplicate voxels
    Only first instance of voxel will appear
    Assume data[:4] = [x,y,z,batchid]
    Assumes data is lexicographically sorted in x,y,z,batch order
    """
    # set number of cols to look at
    if usebatch:
        k = 4
    else:
        k = 3
    n = data.shape[0]
    ret = np.empty(n, dtype=np.bool)
    ret[0] = True
    for i in range(n-1):
        if np.all(data[i,:k] == data[i+1,:k]):
            # duplicate voxel
            ret[i+1] = False
        else:
            # new voxel
            ret[i+1] = True
    return ret


def filter_nonimg_voxels(data_grp, data_img, usebatch=True):
    """
    return array that will filter out voxels in data_grp that are not in data_img
    ASSUME: data_grp and data_img are lexicographically sorted in x,y,z,batch order
    ASSUME: all points in data_img are also in data_grp
    ASSUME: all voxels in data are unique
    """
    # set number of cols to look at
    if usebatch:
        k = 4
    else:
        k = 3
    ngrp = data_grp.shape[0]
    nimg = data_img.shape[0]
    igrp = 0
    iimg = 0
    ret = np.empty(ngrp, dtype=np.bool) # return array
    while igrp < ngrp and iimg < nimg:
        if np.all(data_grp[igrp,:k] == data_img[iimg,:k]):
            # voxel is in both data
            ret[igrp] = True
            igrp += 1
            iimg += 1
        else:
            # voxel is in data_grp, but not data_img
            ret[igrp] = False
            igrp += 1
    # need to go through rest of data_grp if any left
    while igrp < ngrp:
        ret[igrp] = False
        igrp += 1
    return ret


def filter_group_data(data_grp, data_img):
    """
    return return array that will permute and filter out voxels so that data_grp and data_img have same voxel locations
    1) lexicographically sort group data (images are lexicographically sorted)
    2) remove voxels from group data that are not in image
    3) choose only one group per voxel (by lexicographic order)
    WARNING: (3) is certainly not a canonical choice
    """
    # step 1: lexicographically sort group data
    perm = np.lexsort(data_grp[:,:-1:].T)
    data_grp = data_grp[perm,:]

    # step 2: remove duplicates
    sel1 = filter_duplicate_voxels(data_grp)
    inds1 = np.where(sel1)[0]
    data_grp = data_grp[inds1,:]

    # step 3: remove voxels not in image
    sel2 = filter_nonimg_voxels(data_grp, data_img)
    inds2 = np.where(sel2)[0]

    return perm[inds1[inds2]]


def process_group_data(data_grp, data_img):
    """
    return processed group data
    1) lexicographically sort group data (images are lexicographically sorted)
    2) remove voxels from group data that are not in image
    3) choose only one group per voxel (by lexicographic order)
    WARNING: (3) is certainly not a canonical choice
    """
    data_grp_np = data_grp.cpu().detach().numpy()
    data_img_np = data_img.cpu().detach().numpy()

    inds = filter_group_data(data_grp_np, data_img_np)

    return data_grp[inds,:]


def get_interaction_id(particle_v, np_features,):
    '''
    A function to sort out interaction ids.
    Note that this assumes cluster_id==particle_id.
    Inputs:
        - particle_v vector: larcv::EventParticle.as_vector()
        - np_features: a numpy array with the shape (n,4) where 4 is voxel value,
        cluster id, group id, and semantic type respectively
    Outputs:
        - interaction_ids: a numpy array with the shape (n,1) where 1 is the interaction ids
    '''
    # initiate the interaction_ids, setting all ids to -1 (as unknown) by default
    interaction_ids = (-1.)*np.ones(np_features.shape[0])
    ##########################################################################
    # sort out the interaction ids using the information of ancestor vtx info
    ##########################################################################
    # get the particle ancestor vtx array first
    ancestor_vtxs = []
    for particle in particle_v:
        ancestor_vtx = [
            particle.ancestor_x(),
            particle.ancestor_y(),
            particle.ancestor_z(),
        ]
        ancestor_vtxs.append(ancestor_vtx)
    ancestor_vtxs = np.asarray(ancestor_vtxs)
    # get the list of unique interaction vertexes
    interaction_vtx_list = np.unique(
        ancestor_vtxs,
        axis=0,
    ).tolist()
    # loop over clust_ids
    for clust_id in range(particle_v.size()):
        # get the interaction id from the unique list (index is the id)
        interaction_id = interaction_vtx_list.index(
            ancestor_vtxs[clust_id].tolist()
        )
        # update the interaction_ids array
        clust_inds = np.where(np_features[:,1]==clust_id)[0]
        interaction_ids[clust_inds] = interaction_id
    # reshape the output
    interaction_ids = np.reshape(
        interaction_ids,
        (-1,1)
    )
    return interaction_ids


def form_groups(data):
    '''
    A function to get the indexes of the data tensor which belongs to same group in same batch
    input:
        data: tensor with each element in format of (x, y, z, batch_id, voxel_val, cluster_id, group_id, sem_type, interaction_id, .....)
    output:
        list of index list
    '''
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    groups = [] # main output list
    # loop over batches
    for batch_id in np.unique(data[:, 3]):
        # select the indexes that belong to the same batch
        inds_batch = np.where(data[:,3]==batch_id)[0]
        # loop over the group_ids within the batch
        for group_id in np.unique(data[inds_batch,6]):
            # select the indexes which have the group id
            inds_group = np.where(data[inds_batch,6]==group_id)[0]
            # append the indexes of the same batch-group voxels to output
            groups.append(inds_batch[inds_group])
    return np.asarray(groups)


def get_major_label(data_label, branch_index, index_list):
    '''
    This is a general function for extracting labels from based on a list of indexes corresponding to grouping/clustering/whatever
        - data_label: (N, xx) tensor as in format of (x, y, z, batch_id, voxel_val, cluster_id, group_id, sem_type, interaction_id, .....)
        - branch_index: the label index for extraction
        - index_list: list of indexes corresponding to certain grouping/clustering/or whatever criteria
    output:
        label_list. Note the same grouping may contain multiple extracted labels. We selected the major one.

    It is a general function for cluster.get_cluster_label, cluster.get_cluster_group, cluster.get_cluster_batch,
    '''
    if isinstance(data_label, torch.Tensor):
        data_label = data_label.cpu().detach().numpy()
    labels = []
    # loop over the grouping/clustering list
    for indexes in index_list:
        labels.append(
            mode(data_label[indexes,branch_index])[0][0]
        )
    return labels


def assign_clustered_groups(
        edge_index,
        edge_pred,
        device=None,
        cuda=True
):
    '''
    Function for assigning each cluster a reconstructed group id
    Inputs:
        - edge_index: edge indexes with shape of (2, N'). Rows are in [cluster_index_1, cluster_index_2]. Index correspond to index of batch_ids, clust_ids and group_ids. NOTE: First row of edge_index shall be ascending!
        - edge_pred: prediction of edge weights in shape of (N', 2) with column ( score for unconnection, score for connectivity). Score range from 0 to 1 or -1 to 1 depending on the predictor type.
    Output:
        - group_ids_pred: predicted group ids
    ##### Notice!
    Here we use clusters and groups as the sub-level segmentations and grouped segmentations, respectively, in a standard clustering case! It also applies to a specific case where we want cluster "groups" into "interactions"
    '''
    # get the size of clusters
    size_of_clusters = torch.max(edge_index)+1
    # get the index of predicted, which is also equivalent to whether this edge connect (0 for no, 1 for yes) under current data structure
    _, pred_inds = torch.max(edge_pred, 1)
    selected_inds = torch.nonzero(pred_inds, dtype=torch.long)
    ##############################
    ## Based on pred_inds group the clusters, give them pseudo group ids
    ##############################
    group_ids_pred = np.ones(size_of_clusters, dtype=np.int) * (-1)
    group_ids_pred_unique_list = []
    for clust_index_1, clust_index_2 in edge_index.t()[selected_inds]:
        if group_ids_pred[clust_index_1]>0 and group_ids_pred[clust_index_2]>0:
            pass
        elif group_ids_pred[clust_index_1]>0:
            group_ids_pred[clust_index_2]=group_ids_pred[clust_index_1]
        elif group_ids_pred[clust_index_2]>0:
            group_ids_pred[clust_index_1]=group_ids_pred[clust_index_2]
        else:
            group_ids_pred[clust_index_1]=len(group_ids_pred_unique_list)
            group_ids_pred[clust_index_2]=len(group_ids_pred_unique_list)
            group_ids_pred_unique_list.append(len(group_ids_pred_unique_list))
    # return
    group_ids_pred = torch.tensor(group_ids_pred, dtype=torch.long, requires_grad=False)
    if not device is None:
        group_ids_pred = group_ids_pred.to(device)
    elif cuda:
        group_ids_pred = group_ids_pred.cuda()
    return group_ids_pred
