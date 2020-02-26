########################################
## GAN for generating fake label 1 image from label-0 event
########################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch


class GAN(torch.nn.Module):
    """
    Class for GAN.
    Generator model using UResNet+MLP
    Discriminator model using Autoencoder+MLP

    for use in config:
    model:
        name: gan
        modules:
            generator:
                <uresnet config>
            discriminator:
                <cnn_encoder config>
    """
    def __init__(self, cfg):
        super(GAN, self).__init__()

        # Initialize the generator & discriminator
        self.generator     = gan_generator_construct(cfg)
        self.discriminator = gan_discriminator_construct(cfg)

    def forward(self, input):
        """
        Args:
            data ([torch.tensor]): (N, >=5) [x, y, z, batchid, value].
            sim ([torch.tensor]): (M, >=5) [x, y, z, batchid, value].
            Note: Total number of batches between data and sim shall be the same. We want to mimic data using sim.
        Returns:
            dict:
                'generated_data': (N, >=5) [x, y, z, batchid, value]
                'pred_data': predicted scores for data
                'batch_id_raw': The corresponding batch ids for label_pred_raw
                'label_pred_gen': predicted label by discriminator for generated data.
                'batch_id_gen': The corresponding batch ids for label_pred_gen. There can be batch ids missing, meaning they are simulations.
        """
        # Get the input
        data = input[0]
        device = data.device

        # Get the indexes for data
        inds_data = data[:,5].nonzero().view(-1)

        # Get the indexes for simulation
        inds_sim  = (data[:,5]==1).nonzero().view(-1)

        # Get the generated data
        generated_data = self.generator(data[inds_data,:])

        # Get the prediction of label
        label_pred_raw, batch_id_raw = self.discriminator(data)
        label_pred_gen, batch_id_gen = self.discriminator(generated_data)

        # concatenate the data
        generated_data = torch.concat((data[inds_sim,:], generated_data),dim=0)

        return {
            'generated_data': [generated_data],
            'label_pred_raw': [label_pred_raw],
            'label_pred_gen': [label_pred_gen],
            'batch_id_raw':   [batch_id_raw],
            'batch_id_gen':   [batch_id_gen],
        }


class GANLoss(torch.nn.Module):
    """
    Take the output of GAN and computes the loss

    for use in config:
    model:
        name: gan
        modules:
            loss            : <loss function: 'CE' or 'MM' (default 'CE')>
            reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
            gen_loss_factor : <factor before the pixel-wise loss for generator, default None>
            generator:
                <uresnet config>
            discriminator:
                <cnn_encoder config>
    """
    def __init__(self, cfg):
        super(GANLoss, self).__init__()

        # Get the chain input parameters
        chain_config = cfg['chain']

        # Set the loss
        self.loss = chain_config.get('loss', 'CE')
        self.reduction = chain_config.get('reduction', 'mean')

        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = chain_config.get('p', 1)
            margin = chain_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('Loss not recognized: ' + self.loss)


    def forward(self, out, clusters):
        """
        Applies the requested loss for gan
        The goal is to discriminate between data and simulation as much as possible
        In the meantime can label the generated data as data as much as possible

        Args:
            out (dict):
                'generated_data': (N, >=6) [x, y, z, batchid, value, label (binary)] if label=0, return the same data as input
                'label_pred_raw': predicted label by discriminator for original data.
                'batch_id_raw': The corresponding batch ids for label_pred_raw
                'label_pred_gen': predicted label by discriminator for generated data.
                'batch_id_gen': The corresponding batch ids for label_pred_gen. There can be batch ids missing, meaning they are simulations.
            clusters ([torch.tensor])     : (N,>=6) [x, y, z, batchid, value, label]
        Returns:
            double: loss, accuracy
        """