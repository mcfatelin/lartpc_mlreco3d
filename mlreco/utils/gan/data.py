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


