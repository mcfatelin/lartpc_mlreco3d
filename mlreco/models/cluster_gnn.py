# GNN that attempts to put clusters together into groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from .gnn import edge_model_construct, node_encoder_construct, edge_encoder_construct
from .layers.dbscan import DBScanClusts2
from mlreco.utils.gnn.data import regulate_to_data
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_label, get_cluster_batch, get_cluster_group, get_start_points
from mlreco.utils.gnn.network import complete_graph, delaunay_graph, mst_graph, bipartite_graph, inter_cluster_distance, get_fragment_edges
from mlreco.utils.gnn.evaluation import edge_assignment, edge_assignment_from_graph
from mlreco.utils import local_cdist
from mlreco.utils.groups import reassign_id, merge_batch, merge_batch_based_on_list
import random

class ClustEdgeGNN(torch.nn.Module):
    """
    Driver class for cluster edge prediction, assumed to be a GNN model.
    This class mostly acts as a wrapper that will hand the graph data to another model.
    If DBSCAN is used, use the semantic label tensor as an input.

    For use in config:
    model:
      name: cluster_gnn
      modules:
        chain:
          node_type       : <semantic class to group (all classes if -1, default 0, i.e. EM)>
          node_min_size   : <minimum number of voxels inside a cluster to be considered (default -1)>
          network         : <type of network: 'complete', 'delaunay', 'mst' or 'bipartite' (default 'complete')>
          edge_max_dist   : <maximal edge Euclidean length (default -1)>
          edge_dist_method: <edge length evaluation method: 'centroid' or 'set' (default 'set')>
          merge_batch     : <flag for whether to merge batches, default False>
          merge_batch_mode: <mode of batch merging, 'const' or 'fluc'; 'const' use a fixed size of batch for merging, 'fluc' takes the input size a mean and sample based on it>
          merge_batch_size: <size of batch merging>
          edge_dist_numpy : <use numpy to compute inter cluster distance (default False)>
        dbscan:
          <dictionary of dbscan parameters>
        node_encoder:
          name: <name of the node encoder>
          <dictionary of arguments to pass to the encoder>
          model_path      : <path to the encoder weights>
        edge_encoder:
          name: <name of the edge encoder>
          <dictionary of arguments to pass to the encoder>
          model_path      : <path to the encoder weights>
        edge_model:
          name: <name of the edge model>
          <dictionary of arguments to pass to the model>
          model_path      : <path to the model weights>
    ################
    In case of "mix" type node (or edge) encoder, config need to be like:
        node_encoder:
              geo_encoder:
                  <dictionary of arguments to pass to the encoder>
                  model_path      : <path to the encoder weights>
              cnn_encoder:
                  <dictionary of arguments to pass to the encoder>
                  model_path      : <path to the encoder weights>
    """
    def __init__(self, cfg):
        super(ClustEdgeGNN, self).__init__()

        # Get the chain input parameters
        chain_config = cfg['chain']

        # Choose what type of node to use
        self.node_type = chain_config.get('node_type', 0)
        self.node_min_size = chain_config.get('node_min_size', -1)

        # Choose what type of network to use
        self.network = chain_config.get('network', 'complete')
        self.edge_max_dist = chain_config.get('edge_max_dist', -1)
        self.edge_dist_metric = chain_config.get('edge_dist_metric','set')
        self.edge_dist_numpy = chain_config.get('edge_dist_numpy',False)

        # Extra flags for merging events in batch
        self.merge_batch = chain_config.get('merge_batch', False)
        self.merge_batch_mode = chain_config.get('merge_batch_mode', 'const')
        self.merge_batch_size = chain_config.get('merge_batch_size', 2)
        self.add_start_point = chain_config.get('add_start_point', False) # whether add start point into the node features

        # Hidden flag for shuffling cluster
        self.shuffle_clusters = chain_config.get('shuffle_clusters', False)

        # If requested, use DBSCAN to form clusters from semantics
        self.do_dbscan = False
        if 'dbscan' in cfg:
            self.do_dbscan = True
            self.dbscan = DBScanClusts2(cfg)

        # Initialize encoders
        self.node_encoder = node_encoder_construct(cfg)
        self.edge_encoder = edge_encoder_construct(cfg)

        # Construct the model
        self.edge_predictor = edge_model_construct(cfg)



    def forward(self, data):
        """
        Prepares particle clusters and feed them to the GNN model.

        Args:
            data ([torch.tensor]): (N,5-6) [x, y, z, batchid, (value,) id]
        Returns:
            dict:
                'edge_pred' (torch.tensor): (E,2) Two-channel edge predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
                'edge_index' (np.ndarray) : (E,2) Incidence matrix
        """
        # Find index of points that belong to the same clusters
        # If a specific semantic class is required, apply mask
        # Here the specified size selection is applied
        particles = None
        if len(data)>1:
            particles = data[1]
        data = data[0]
        device = data.device
        if self.do_dbscan:
            clusts = self.dbscan(data, onehot=False)
            if self.node_type > -1:
                clusts = clusts[self.node_type]
            else:
                clusts = np.concatenate(clusts).tolist()
        else:
            if self.node_type > -1:
                mask = torch.nonzero(data[:,-1] == self.node_type).flatten()
                clusts = form_clusters(data[mask], self.node_min_size)
                clusts = [mask[c].cpu().numpy() for c in clusts]
            else:
                clusts = form_clusters(data, self.node_min_size)
                clusts = [c.cpu().numpy() for c in clusts]

        if not len(clusts):
            return {}

        # if shuffle the clusters
        if self.shuffle_clusters:
            random.shuffle(clusts)

        # if merge_batch set all batch id to zero
        # and also reassign ids and group ids
        if self.merge_batch:
            # It needs to regulate between data and particles before merging batches
            # bc particles can have more group_id depending on how
            # data (clusters) were obtained
            particles = regulate_to_data(data, particles)
            if self.merge_batch_mode=='fluc':
                data, merging_batch_list = merge_batch(data, self.merge_batch_size, whether_fluctuate=True)
            else:
                data, merging_batch_list = merge_batch(data, self.merge_batch_size, whether_fluctuate=False)
            particles, _ = merge_batch_based_on_list(
                particles,
                merging_batch_list,
                data_type='particle'
            )

        # Get the batch id for each cluster
        batch_ids = get_cluster_batch(data, clusts)

        # Compute the cluster distance matrix, if necessary
        dist_mat = None
        if self.edge_max_dist > 0 or self.network == 'mst' or self.network == 'knn':
            dist_mat = inter_cluster_distance(data[:,:3], clusts, batch_ids, self.edge_dist_metric, self.edge_dist_numpy)

        # Form the requested network
        if len(clusts) == 1:
            edge_index = np.empty((2,0))
        elif self.network == 'complete':
            edge_index = complete_graph(batch_ids, dist_mat, self.edge_max_dist)
        elif self.network == 'delaunay':
            edge_index = delaunay_graph(data, clusts, dist_mat, self.edge_max_dist)
        elif self.network == 'mst':
            edge_index = mst_graph(batch_ids, dist_mat, self.edge_max_dist)
        elif self.network == 'knn':
            edge_index = knn_graph(batch_ids, dist_mat, k=5)
        elif self.network == 'bipartite':
            primary_ids = [i for i, c in enumerate(clusts) if (data[c,-3] == data[c,-2]).any()]
            edge_index = bipartite_graph(batch_ids, primary_ids, dist_mat, self.edge_max_dist)
        else:
            raise ValueError('Network type not recognized: '+self.network)

        # Skip if there is less than two edges (fails batchnorm)
        if edge_index.shape[1] < 2:
            return {}

        # Obtain node and edge features
        x = self.node_encoder(data, clusts)
        e = self.edge_encoder(data, clusts, edge_index)

        # See if need add start point to node features
        if self.add_start_point:
            x = torch.cat([x, get_start_points(particles, data, clusts)], dim=1)

        # Bring edge_index and batch_ids to device
        index = torch.tensor(edge_index, device=device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=device)

        # Pass through the model, get output (long edge_index)
        out = self.edge_predictor(x, index, e, xbatch)
        edge_pred = out['edge_pred'][0]

        # Divide the output out into different arrays (one per batch)
        _, counts = torch.unique(data[:,3], return_counts=True)
        vids = np.concatenate([np.arange(n.item()) for n in counts])
        cids = np.concatenate([np.arange(n) for n in np.unique(batch_ids, return_counts=True)[1]])
        bcids = [np.where(batch_ids == b)[0] for b in range(len(counts))]
        beids = [np.where(batch_ids[edge_index[0]] == b)[0] for b in range(len(counts))]

        edge_pred = [edge_pred[b] for b in beids]
        edge_index = [cids[edge_index[:,b]].T for b in beids]
        clusts = [np.array([vids[c] for c in np.array(clusts)[b]]) for b in bcids]

        # debug use, temporary
        output_x = [x[b].cpu().detach().numpy() for b in bcids]
        output_e = [e[b].cpu().detach().numpy() for b in beids]

        return {
            'edge_pred': [edge_pred],
            'edge_index': [edge_index],
            'clusts': [clusts],
            'x': [output_x],
            'e': [output_e],
        }


class EdgeChannelLoss(torch.nn.Module):
    """
    Takes the output of EdgeModel and computes the channel-loss.

    For use in config:
    model:
      name: cluster_gnn
      modules:
        chain:
          loss            : <loss function: 'CE' or 'MM' (default 'CE')>
          reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
          balance_classes : <balance loss per class: True or False (default False)>
          target          : <basis to form target adjacency matrix:'group', 'group_max', 'photon' (default 'group')>
          high_purity     : <only penalize loss on groups with a primary (default False)>
    """
    def __init__(self, cfg):
        super(EdgeChannelLoss, self).__init__()

        # Get the chain input parameters
        chain_config = cfg['chain']

        # Set the loss
        self.loss = chain_config.get('loss', 'CE')
        self.reduction = chain_config.get('reduction', 'sum')
        self.balance_classes = chain_config.get('balance_classes', False)
        self.target = chain_config.get('target', 'group')
        self.high_purity = chain_config.get('high_purity', False)

        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = chain_config.get('p', 1)
            margin = chain_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('Loss not recognized: ' + self.loss)

    def forward(self, out, clusters, graph):
        """
        Applies the requested loss on the edge prediction.

        Args:
            out (dict):
                'edge_pred' (torch.tensor): (E,2) Two-channel edge predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
                'edge_index' (np.ndarray) : (E,2) Incidence matrix
            clusters ([torch.tensor])     : (N,8) [x, y, z, batchid, value, id, groupid, shape]
            graph ([torch.tensor])        : (N,3) True edges
        Returns:
            double: loss, accuracy, clustering metrics
        """
        total_loss, total_acc = 0., 0.
        n_edges = 0
        for i in range(len(clusters)):

            # If this batch did not have any node, proceed
            if 'edge_pred' not in out:
                continue

            # Get the list of batch ids, loop over individual batches
            batches = clusters[i][:,3]
            nbatches = len(batches.unique())
            for j in range(nbatches):

                # Narrow down the tensor to the rows in the batch
                labels = clusters[i][batches == j]

                # Use group information or particle tree to determine the true edge assigment
                edge_pred = out['edge_pred'][i][j]
                if not edge_pred.shape[0]:
                    continue
                edge_index = out['edge_index'][i][j]
                clusts = out['clusts'][i][j]
                group_ids = get_cluster_group(labels, clusts)

                # If high purity is requested, remove edges in poorly defined groups from the loss
                if self.high_purity:
                    clust_ids   = np.array([labels[c[0],5].item() for c in clusts])
                    purity_mask = np.ones(len(edge_index), dtype=bool)
                    for g in np.unique(group_ids):
                        group_mask = np.where(group_ids == g)[0]
                        if g not in clust_ids[group_mask]:
                            edge_mask = [(e[0] in group_mask) & (e[1] in group_mask) for e in edge_index]
                            purity_mask[edge_mask] = np.zeros(np.sum(edge_mask))
                    edge_index = edge_index[purity_mask]
                    edge_pred = edge_pred[np.where(purity_mask)[0]]
                    if not len(edge_index):
                        continue

                if self.target == 'group':
                    edge_assn = edge_assignment(edge_index, group_ids)
                elif self.target == 'group_mst':
                    # For each group, find the most likely spanning tree, label the edges in the
                    # tree as 1. For all other edges, apply loss only if in separate group.
                    # If undirected, also assign symmetric path to 1.
                    from scipy.sparse.csgraph import minimum_spanning_tree
                    edge_assn     = edge_assignment(edge_index, group_ids)
                    off_scores    = torch.softmax(edge_pred, dim=1)[:,0].detach().cpu().numpy()
                    score_mat     = np.full((len(clusts), len(clusts)), 2.0)
                    score_mat[tuple(edge_index.T)] = off_scores
                    new_edges = np.empty((0,2))
                    for g in np.unique(group_ids):
                        clust_ids = np.where(group_ids == g)[0]
                        if len(clust_ids) < 2:
                            continue

                        mst_mat = minimum_spanning_tree(score_mat[np.ix_(clust_ids,clust_ids)]).toarray().astype(float)
                        inds = np.where(mst_mat.flatten() > 0.)[0]
                        ind_pairs = np.array(np.unravel_index(inds, mst_mat.shape)).T
                        edges = np.array([[clust_ids[i], clust_ids[j]] for i, j in ind_pairs])
                        edges = np.concatenate((edges, np.flip(edges, axis=1))) # reciprocal connections
                        new_edges = np.concatenate((new_edges, edges))

                    edge_assn_max = np.zeros(len(edge_assn))
                    for e in new_edges:
                        edge_id = np.where([(e == ei).all() for ei in edge_index])[0]
                        edge_assn_max[edge_id] = 1.

                    max_mask = edge_assn == edge_assn_max
                    edge_assn = edge_assn_max[max_mask]
                    edge_pred = edge_pred[np.where(max_mask)[0]]
                    if not len(edge_pred):
                        continue
                elif 'photon' in self.target:
                    clust_ids = get_cluster_label(labels, clusts)
                    subgraph = graph[i][graph[i][:,-1] == j, :2]
                    true_edge_index = get_fragment_edges(subgraph, clust_ids)
                    edge_assn = edge_assignment_from_graph(edge_index, true_edge_index)
                    if self.target == 'photon_relax':
                        adjacency  = [group_ids[e[0]] == group_ids[e[1]] for e in edge_index]
                        relax_mask = adjacency == edge_assn
                        edge_assn  = edge_assn[relax_mask]
                        edge_pred  = edge_pred[np.where(relax_mask)[0]]
                        if not len(edge_pred):
                            continue
                else:
                    raise ValueError('Prediction target not recognized:', self.target)

                edge_assn = torch.tensor(edge_assn, device=edge_pred.device, dtype=torch.long, requires_grad=False).view(-1)

                # Increment the loss, balance classes if requested
                if self.balance_classes:
                    vals, counts = torch.unique(edge_assn, return_counts=True)
                    weights = np.array([float(counts[k])/len(edge_assn) for k in range(len(vals))])
                    for k, v in enumerate(vals):
                        total_loss += (1./weights[k])*self.lossfn(edge_pred[edge_assn==v], edge_assn[edge_assn==v])
                else:
                    total_loss += self.lossfn(edge_pred, edge_assn)

                # Compute accuracy of assignment (fraction of correctly assigned edges)
                total_acc += torch.sum(torch.argmax(edge_pred, dim=1) == edge_assn).float()

                # Increment the number of events
                n_edges += len(edge_pred)

        # Handle the case where no cluster/edge were found
        if not n_edges:
            return {
                'accuracy': 0.,
                'loss': torch.tensor(0., requires_grad=True, device=clusters[0].device),
                'n_edges': n_edges
            }

        return {
            'accuracy': total_acc/n_edges,
            'loss': total_loss/n_edges,
            'n_edges': n_edges
        }
