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


# Qing's function. Remove after debugging
def get_interaction_id(
        particle_v,
        np_features,
        **kwargs
):
    '''
    A function to sort out interaction ids.
    Note that this assumes cluster_id==particle_id.
    Inputs:
        - particle_v vector: larcv::EventParticle.as_vector()
        - np_features: a numpy array with the shape (n,4) where 4 is voxel value,
        cluster id, group id, and semantic type respectively
        - kwargs: customer defined hyperparameters:
            - start_point_tol: maximum distance between two start points that can be classified as vertex
            - to-be-added
    Outputs:
        - interaction_ids: a numpy array with the shape (n,1) where 1 is the interaction ids
    '''
    # initiate the interaction_ids, setting all ids to -1 (as unknown) by default
    interaction_ids = (-1.)*np.ones(np_features.shape[0])
    # sort out interaction ids for the primary clusters
    interaction_ids, primary_ids = sort_out_interactions_primary(particle_v, np_features, interaction_ids, **kwargs)
    # sort out interaction ids for the daughter particles from hidden particles (such as pi0)
    # interaction_ids = sort_out_interactions_hidden_mother(particle_v, np_features, interaction_ids, **kwargs)
    # sort out the rest of the interactions
    # interaction_ids = sort_out_interactions_remain(particle_v, np_features, interaction_ids, **kwargs)
    # # reshape the output
    interaction_ids = np.reshape(
        interaction_ids,
        (-1,1)
    )
    # return interaction_ids
    primary_ids = np.reshape(
        primary_ids,
        (-1,1)
    )
    # debug, remove after using
    return  np.concatenate((interaction_ids,primary_ids), axis=1)


# sub-functionalities for get_interaction_id
def sort_out_interactions_primary(
        particle_v,
        np_features,
        input_interaction_ids,
        **kwargs
):
    '''
    A function to assign the primaries same interaction id
    if their start_points are very close to each other (distance below certain limit)
    Inputs:
        Basically same as get_interaction_id, but have one more:
        - input_interaction_ids: assigned interaction ids from last step
    Outputs:
        - output_interaction_ids: assigned interaction ids after this step
    '''
    output_interaction_ids = input_interaction_ids
    #debug
    primary_ids = (-1)*np.ones(input_interaction_ids.shape[0])
    # loop over group_ids
    # record start point [x, y, z] of primary clusters
    # the recordered vertex index is the interaction_id
    vertexes = np.zeros((0, 3))
    for group_id in np.unique(np_features[:, 3]):
        # select voxels with same group_id
        inds_same_group_id = np.where(np_features[:, 3]==group_id)[0]
        # loop over the clusters within one group
        for clust_id in np.unique(np_features[inds_same_group_id, 2]):
            # continue if the cluster is not from primary
            prc = particle_v[int(clust_id)].creation_process()
            if prc!='primary':
                continue
            selection = np.logical_and(
                np_features[:, 3] == group_id,
                np_features[:, 2] == clust_id,
            )
            selected_inds = np.where(selection)[0]
            x = np.asarray([[
                particle_v[int(clust_id)].first_step().x(),
                particle_v[int(clust_id)].first_step().y(),
                particle_v[int(clust_id)].first_step().z(),
            ]])
            # debug remove after using
            primary_ids[selected_inds]=1.
            # if the cluster is from primary
            # check if its start point is close to existing vertexes within a tolerance
            # first if vertexes contains no found interaction yet
            if vertexes.shape[0]==0:
                # update the found vertexes list
                vertexes = np.concatenate((vertexes,x), axis=0)
                # update the interaction ids
                output_interaction_ids[selected_inds]=0
                continue
            # if not, then check all the distances between found vertexes and this cluster's start point
            # if the minimum of it is below the tolerance, then this cluster belong to the interaction
            d = cdist(vertexes, x, 'euclidean')
            d = np.reshape(d, d.shape[0]) # flatten
            min_d_index = np.argmin(d) # this index is the interaction id if tolerance requirement met
            if d[min_d_index]<=kwargs.get('start_point_tol', 1): # 1 cm
                # update the interaction ids
                output_interaction_ids[selected_inds] = min_d_index
            else:
                # update the interaction ids
                output_interaction_ids[selected_inds] = vertexes.shape[0]
                # update found vertexes
                vertexes = np.concatenate((vertexes, x), axis=0)
    # return output_interaction_ids
    # debug, remove after using
    return output_interaction_ids, primary_ids

def sort_out_interaction_hidden_mother(
        particle_v,
        np_features,
        input_interaction_ids,
        **kwargs
):
    '''
    UNFINISHED
    A function to assign the groups same interaction id
    if their parents are same and parent is hidden, like pi0.
    And based on certain criteria we find them originated from the known vertexes
    Inputs:
        Basically same as get_interaction_id, but have one more:
        - input_interaction_ids: assigned interaction ids from last step
    Outputs:
        - output_interaction_ids: assigned interaction ids after this step
    '''
    output_interaction_ids = input_interaction_ids
    #
    return output_interaction_ids


def sort_out_interaction_remain(
        particle_v,
        np_features,
        input_interaction_ids,
        **kwargs
):
    '''
    UNFINISHED
    A function to assign the groups same interaction id for rest of groups after primary and hidden-mother groups
    Inputs:
        Basically same as get_interaction_id, but have one more:
        - input_interaction_ids: assigned interaction ids from last step
    Outputs:
        - output_interaction_ids: assigned interaction ids after this step
    '''
    output_interaction_ids = input_interaction_ids
    #
    return output_interaction_ids