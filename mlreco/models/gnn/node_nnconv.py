# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d
from torch_geometric.nn import MetaLayer, NNConv
from torch_scatter import scatter_mean
import torch.nn.functional as F

from .edge_pred import EdgeModel, BilinEdgeModel

class NodeNNConvModel(torch.nn.Module):
    """
    Simple GNN with several edge convolutions, followed by MetaLayer for nodes prections

    for use in config
    model:
        modules:
            node_model:
              name: nnconv
    """
    def __init__(self, cfg):
        super(NodeNNConvModel, self).__init__()


        if 'modules' in cfg:
            self.model_config = cfg['modules']['edge_nnconv']
        else:
            self.model_config = cfg


        self.node_in = self.model_config.get('node_feats', 16)
        self.edge_in = self.model_config.get('edge_feats', 19)

        self.aggr = self.model_config.get('aggr', 'add')
        self.leak = self.model_config.get('leak', 0.1)

        # perform batch normalization
        self.bn_node = BatchNorm1d(self.node_in)
        self.bn_edge = BatchNorm1d(self.edge_in)

        self.num_mp = self.model_config.get('num_mp', 3)

        self.nn = torch.nn.ModuleList()
        self.layer = torch.nn.ModuleList()
        ninput = self.node_in
        noutput = max(2*self.node_in, 32)
        for i in range(self.num_mp):
            self.nn.append(
                Seq(
                    Lin(self.edge_in, ninput),
                    LeakyReLU(self.leak),
                    Lin(ninput, ninput*noutput),
                    LeakyReLU(self.leak)
                )
            )
            self.layer.append(
                NNConv(ninput, noutput, self.nn[i], aggr=self.aggr)
            )
            ninput = noutput

        class NodeModel(torch.nn.Module):
            def __init__(self):
                super(NodeModel, self).__init__()

                #self.node_mlp_1 = Seq(Lin(34, 64), LeakyReLU(0.12), Lin(64, 32))
                self.node_mlp_2 = Seq(Lin(34, 16), LeakyReLU(0.12), Lin(16, 2))
                #self.node_mlp = Seq(Lin(64, 32), LeakyReLU(0.12), Lin(32, 16), LeakyReLU(0.12), Lin(32, 2))

            def forward(self, x, edge_index, edge_attr, u, batch):
                row, col = edge_index
                out = torch.cat([x[col], edge_attr], dim=1)
                #out = self.node_mlp_1(out)
                out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
                return self.node_mlp_2(out)

        # final prediction layer
        pred_cfg = self.model_config.get('pred_model', 'basic')
        if pred_cfg == 'basic':
            self.predictor = MetaLayer(EdgeModel(noutput, self.edge_in, self.leak), NodeModel())
        elif pred_cfg == 'bilin':
            self.predictor = MetaLayer(BilinEdgeModel(noutput, self.edge_in, self.leak))
        else:
            raise Exception('unrecognized prediction model: ' + pred_cfg)
    def forward(self, x, edge_index, e, xbatch):
        """
        inputs data:
            x - vertex features
            edge_index - graph edge list
            e - edge features
            xbatch - node batchid
        """

        x = x.view(-1,self.node_in)
        e = e.view(-1,self.edge_in)
        if self.edge_in > 1:
            e = self.bn_edge(e)
        if self.node_in > 1:
            x = self.bn_node(x)

        # go through layers
        for i in range(self.num_mp):
            x = self.layer[i](x, edge_index, e)

        x, e, u = self.predictor(x, edge_index, e, u=None, batch=xbatch)
        x = F.log_softmax(x, dim=1)

        return {'node_pred':[x]}
