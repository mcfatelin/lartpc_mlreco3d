# GNN attempts to clustering groups into interaction
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features, edge_assignment, get_input_node_features, get_input_edge_features
from mlreco.utils.groups import form_groups, get_major_label, assign_clustered_groups
from mlreco.utils.gnn.compton import filter_compton
from mlreco.utils.gnn.network import group_bipartite
from mlreco.utils.metrics import SBD, AMI, ARI, purity_efficiency
from .gnn import edge_model_construct


class InteractionModel(torch.nn.Module):
    """
        Driver for interaction clustering, assumed to be with PyTorch GNN model.
        This class mostly inherit the structure from edge_gnn

        for use in config
        model:
            modules:
                interaction_model:
                    name: <name of edge model>
                    model_cfg:
                        <dictionary of arguments to pass to model>
                    remove_compton: <True/False to remove compton groups> (default True)
                    compton_threshold: Minimum number of voxels
                    balance_classes: <True/False for loss computation> (default False)
                    loss: 'CE' or 'MM' (default 'CE')
                    node_feature_input: <True/False for using input feature arrays instead of human-supervised feature extraction> (default True)
                    edge_feature_input: <True/False for using input feature arrays instead of human-supervised feature extraction> (default True, it cannot be True when node_feature_input is False)
    """
    def __init__(self, cfg):
        super(InteractionModel, self).__init__()

        # Get the model input parameters
        if 'modules' in cfg:
            self.model_config = cfg['modules']['interaction_model']
        else:
            self.model_config = cfg

        # Extract the model to use
        model = edge_model_construct(self.model_config.get('name', 'edge_only'))

        self.remove_compton = self.model_config.get('remove_compton', True)
        self.compton_thresh = self.model_config.get('compton_thresh', 30)
        self.node_feature_input = self.model_config.get('node_feature_input', True)
        self.edge_feature_input = self.model_config.get('edge_feature_input', True)

        # Assert
        # edge_feature_input cannot be True if node_feature_input is False
        if (not self.node_feature_input) and self.edge_feature_input:
            raise ValueError(
                'edge_feature_input cannot be True if node_feature_input==False'
            )

        # Construct the model
        self.edge_predictor = model(self.model_config.get('model_cfg', {}))
        return

    @staticmethod
    def default_return(device):
        """
        Default forward return if the graph is empty (no node)
        """
        xg = torch.tensor([], requires_grad=True)
        x  = torch.tensor([])
        x.to(device)
        return {'edge_pred':[xg], 'group_ids':[x], 'batch_ids':[x], 'interaction_ids':[x], 'edge_index':[x]}

    def forward(self, data):
        '''
        data:
            inputs data:
            Use the same voxel-level structure as other models so that it can be more compatible
            However the input data structure can change depending on whether the flag 'feature_input' is turned on.
            data[0]: (N, 9) tensor with row (x, y, z, batch_id, voxel_val, cluster_id, group_id, sem_type, interaction_id)
            Optional only when node_feature_input==True (data[1]) and edge_feature_input==True (data[2]):
            data[1]: (G, 2+K) tensor with batch_id, group id, and input node features (size K), G is number of groups
            data[2]: (G*(G+1)/2, 3+E) tensor with batch_id, group_id_1, group_id_2 (basically these two form a edge_id), and input edge features (size E)
        output:
            dictionary, with
                'edge_pred' - torch.tensor with edge prediction weights
        '''
        # Get device
        label = data[0]
        device = label.device

        # get the feature inputs
        node_feat_data = None
        edge_feat_data = None
        if self.node_feature_input:
            node_feat_data = data[1]
            if node_feat_data.device!=device:
                raise ValueError('Multiple devices (node) are defined for InteractionModel inputs!')
        if self.edge_feature_input:
            edge_feat_data = data[2]
            if edge_feat_data.device!=device:
                raise ValueError('Multiple devices (edge) are defined for InteractionModel inputs!')

        # Find index of points that belong to the same group
        groups = form_groups(label)

        # If requested, remove groups below a certain size threshold
        if self.remove_compton:
            selection = np.where(filter_compton(groups, self.compton_thresh))[0]
            if not len(selection):
                return self.default_return(device)
            groups = groups[selection]

        # Get the group, batch, and interaction ids
        group_ids       = get_major_label(label, 6, groups)
        batch_ids       = get_major_label(label, 3, groups)
        interaction_ids = get_major_label(label, 8, groups)

        # Get the edge_id, node features and edge features
        edge_index = None # edge index which is supposed to be (2, -1), 2 shall be the [node_index_1, node_index_2] and node_index_2 > node_index_1. Note index is not id!
        x = None # node feature array in shape of (N, F_n) F_n is the number of node features
        e = None # edge feature array in shape of (N, F_e) F_e is the number of edge features

        if (not self.node_feature_input) and (not self.edge_feature_input):
            # both node and edge features are extracted by human-supervised functions
            edge_index = group_bipartite(batch_ids, group_ids, device=device)
            x = cluster_vtx_features(label, groups, device=device)
            e = cluster_edge_features(label, groups, edge_index, device=device)
        elif (self.node_feature_input) and (not self.edge_feature_input):
            # node features are extracted by CNN encoder
            # edge features are still extracted by human-supervised function
            edge_index = group_bipartite(batch_ids, group_ids, device=device)
            x = get_input_node_features(node_feat_data, batch_ids, group_ids, device=device)
            e = cluster_edge_features(label, groups, edge_index, device=device)
        elif (self.node_feature_input) and (self.edge_feature_input):
            # both node and edge features are from CNN encoder extraction
            x = get_input_node_features(node_feat_data, batch_ids, group_ids, device=device)
            e, edge_index = get_input_edge_features(edge_feat_data, batch_ids, group_ids, device=device)

        if not edge_index.shape[1]:
            return self.default_return(device)

        # Convert the the batch IDs to a torch tensor to pass to torch
        batch_ids = torch.tensor(batch_ids).to(device)

        # Pass through the model, get output
        out = self.edge_predictor(x, edge_index, e, batch_ids)

        return {
            **out,
            'group_ids': [torch.tensor(group_ids).to(device)],
            'batch_ids': [batch_ids],
            'interaction_ids': [torch.tensor(interaction_ids).to(device)],
            'edge_index': [edge_index]
        }



class InteractionClusteringLoss(torch.nn.Module):
    '''
    Interaction clustering loss
    '''
    def __init__(self, cfg):
        super(InteractionClusteringLoss, self).__init__()

        # Get the model loss parameters
        if 'modules' in cfg:
            self.model_config = cfg['modules']['interaction_model']
        else:
            self.model_config = cfg

        self.reduction = self.model_config.get('reduction', 'mean')
        self.loss = self.model_config.get('loss', 'CE')
        self.balance_classes = self.model_config.get('balance_classes', False)

        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = self.model_config.get('p', 1)
            margin = self.model_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('unrecognized loss: ' + self.loss)

    def forward(self, output, labels):
        """
        output:
            dictionary output from the DataParallel gather function
            out['edge_pred'] - n_gpus tensors of predicted edge weights from model forward
        data:
            labels: (N, 9) tensor with row (x, y, z, batch_id, voxel_val, cluster_id, group_id, sem_type, interaction_id)
        """
        edge_ct = 0
        total_loss, total_acc, edge_pur = 0., 0., 0.
        ari, ami, sbd, pur, eff = 0., 0., 0., 0., 0.
        ngpus = len(labels)
        for i in range(len(labels)):

            # Get the necessary data products
            edge_pred = output['edge_pred'][i]
            group_ids = output['group_ids'][i]
            batch_ids = output['batch_ids'][i]
            interaction_ids = output['interaction_ids'][i]
            edge_index = output['edge_index'][i]
            device = edge_pred.device
            if not len(group_ids):
                if ngpus > 1:
                    ngpus -= 1
                continue

            # Use interaction information to determine the true edge assigment
            edge_assn = edge_assignment(edge_index, batch_ids, interaction_ids, device=device, dtype=torch.long)
            edge_assn = edge_assn.view(-1)

            # Increment the loss, balance classes if requested (TODO)
            # during train the summed loss of batch is the one fed to optimzing function
            total_loss += self.lossfn(edge_pred, edge_assn)

            # Compute accuracy of assignment
            # in this model we use the edge prediction efficiency for accuracy
            _, pred_inds = torch.max(edge_pred, 1)
            total_acc += (pred_inds == edge_assn).sum().float() / edge_assn.size()[0]

            #################################
            # standard voxel-level evaluation
            #################################
            # first make the interaction id predictions based on edge_pred
            interaction_ids_pred = assign_clustered_groups(
                edge_index,
                edge_pred,
                cuda=False,
            )

            interaction_ids = interaction_ids.cpu().detach().numpy()
            # calculate the scores, efficiencies and purities
            ari += ARI(interaction_ids_pred, interaction_ids)
            ami += AMI(interaction_ids_pred, interaction_ids)
            sbd += SBD(interaction_ids_pred, interaction_ids)

            pur0, eff0 = purity_efficiency(interaction_ids_pred, interaction_ids)
            pur += pur0
            eff += eff0

            edge_ct += edge_index.shape[1]

        return {
            'ARI': ari/ngpus,
            'AMI': ami/ngpus,
            'SBD': sbd/ngpus,
            'purity': pur/ngpus,
            'efficiency': eff/ngpus,
            'accuracy': total_acc/ngpus,
            'loss': total_loss/ngpus,
            'edge_count': edge_ct
        }