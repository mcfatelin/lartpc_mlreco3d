import torch
import numpy as np
from scipy.spatial.distance import cdist


def filling_empty(data):
    '''
    Inputs: data [tensor] (N, >=5) [x,y,z,batchid,value]
    Returns: same
    Function for filling in data for batch that has empty image
    randomly using the existing image
    '''
    # check if the data has empty batches
    batch_ids = data[:,3].unique().view(-1)
    if batch_ids.size()[0]==batch_ids.max()+1:
        return data
    # for convenience turn batch_ids to numpy
    batch_ids = batch_ids.cpu().detach().numpy()
    # expected batch ids
    full_batch_ids = np.linspace(0, np.max(batch_ids), np.max(batch_ids)+1)
    # complement batch ids for filling
    compl_batch_ids = np.setdiff1d(batch_ids, full_batch_ids)
    # filling batch ids
    fill_batch_ids  = np.random.choice(batch_ids, size=len(compl_batch_ids))
    # Loop over
    datasets = [data]
    for compl_batch_id, fill_batch_id in zip(
        compl_batch_ids,
        fill_batch_ids,
    ):
        # get indexes of certain batch
        selection = data[:,3]==fill_batch_id
        inds = selection.nonzero().view(-1)
        # get the new data piece
        batch_data = data[inds].clone()
        batch_data[:,3] = compl_batch_id
        datasets.append(batch_data)
    return torch.cat(datasets, 0)


def shuffle_data(data):
    '''
    Inputs: data [tensor] (N, >=5) [x,y,z,batchid,value]
    Returns: same
    Function for shuffling data (replacing the batch ids)
    '''
    # get the batch ids
    batch_ids = data[:,3].unique().view(-1).cpu().detach().numpy()
    # shuffle
    shuffled_batch_ids = batch_ids.copy()
    np.random.shuffle(shuffled_batch_ids)
    # Loop over and replace the batch ids
    for batch_id, shuffled_batch_id in zip(
        batch_ids,
        shuffled_batch_ids,
    ):
        selection = data[:,3]==batch_id
        inds = selection.nonzero().view(-1)
        data[inds,3] = shuffled_batch_id
    return data


def form_batches(data):
    '''
    Function to for list of indexes (batches)
    Input:
        - data: (tensor) (N, >=5) [x, y, z, batchid, value]
    Output:
        - batches: (list) of indexes (torch.tensor)
    '''
    batches = []
    for batch_id in data[:,3].unique():
        batches.append(
            torch.nonzero(
                data[:,3]==batch_id,
                device=data.device,
                type=torch.long
            ).view(-1)
        )
    return batches


def form_voxel_index(voxel1, voxel2):
    """
    Function for finding indexes of shared voxels, unique voxels.
    Input:
        - voxel1: (numpy) [N, 3] (x,y,z)
        - voxel2: (numpy) [N, 3] (x,y,z)
        - device: torch device
    Output:
        - inds_shared_1:    (tensor) (N') indexes on tensor1 with value shared by two tensor
        - inds_shared_2:    (tensor) (N') indexes on tensor2 with value shared by two tensor
        - inds_only_on_1: (tensor) (M)  indexes not shared but unique for tensor1
        - inds_only_on_2: (tensor) (M')  indexes not shared but unique for tensor2
    """
    # Calculate the distance metrix
    d12 = cdist(voxel1, voxel2, 'euclidean')
    # Get the shared indexes
    inds_shared_1, inds_shared_2 = np.where(d12==0)
    # Get the unshared indexes
    inds_only_on_1 = np.where(np.min(d12,axis=1)>0)[0]
    inds_only_on_2 = np.where(np.min(d12,axis=0)>0)[0]
    return inds_shared_1, inds_shared_2, inds_only_on_1, inds_only_on_2


def image_difference_score(raw_data, gen_data, image_size):
    '''
    Function for giving the score for how much different the gen_data is compared to raw_data. Batching is allowed. Output score will be the average of batches.

    Inputs:
        - raw_data: (tensor) (N, >=5) [x, y, z, batchid, value]
        - gen_data: (tensor) (N, >=5) [x, y, z, batchid, value]
        - image_size: (int) size of image
    Outputs:
        - score: (float)
    '''
    # form batches
    raw_batches = form_batches(raw_data)
    gen_batches = form_batches(gen_data)
    # Loop over batch to get scores
    score = torch.tensor(0.,dtype=torch.float)
    for raw_b, gen_b in zip(raw_batches, gen_batches):
        # Get the maximum of raw
        raw_max = raw_data[raw_b,4].max()
        # change to numpy
        batch_raw_data_voxels = raw_data[raw_b,:3].clone().cpu().detach().numpy()
        batch_gen_data_voxels = gen_data[gen_b,:3].clone().cpu().detach().numpy()
        # Get voxel indexes where raw and gen share
        inds_raw_share, inds_gen_share, inds_raw_only, inds_gen_only = form_voxel_index(batch_raw_data_voxels, batch_gen_data_voxels)
        # Get shared voxels score
        score += (raw_data[raw_b[inds_raw_share],4] - gen_data[gen_b[inds_gen_share],4])**2 / raw_max
        # Get unshared voxels score
        score += raw_data[raw_b[inds_raw_only],4]**2 / raw_max
        score += gen_data[gen_b[inds_gen_only],4]**2 / raw_max
    # normalize the score by total voxels
    score /= image_size**3
    return score


