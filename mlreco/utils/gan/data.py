import torch
import numpy as np


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
    shuffled_batch_ids = np.random.shuffle(batch_ids)
    # Loop over and replace the batch ids
    for batch_id, shuffled_batch_id in zip(
        batch_ids,
        shuffle_batch_ids,
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
    score = []
    for raw_b, gen_b in zip(raw_batches, gen_batches):
        # Get the maximum of raw
        raw_max = raw_data[raw_b,4].max()
        # change to numpy
        batch_raw_data = raw_data[raw_b,:].clone().cpu().detach().numpy()
        batch_gen_data = gen_data[gen_b,:].clone().cpu().detach().numpy()
        # Get voxel indexes where raw and gen share
        inds_share = np.

