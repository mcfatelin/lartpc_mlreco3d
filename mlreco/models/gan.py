########################################
## GAN for generating fake label 1 image from label-0 event
########################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from mlreco.utils.gan.data import filling_empty, shuffle_data

class GAN(torch.nn.Module):
    """
    Class for GAN.
    Generator model using UResNet+MLP
    Discriminator model using Autoencoder+MLP

    for use in config:
    model:
        name: gan
        modules:
            shuffle_pairing: shuffle the pairing between data and sim in each batch, default False.
            filling_empty:   whether to fill the empty data and sim using random data and sim in the batch. Default True for training.
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

        # extra flags
        self.shuffle_pairing    = cfg.get('shuffle_pairing', False)
        self.filling_empty      = cfg.get('filling_empty', True)

    def forward(self, input):
        """
        Args:
            data ([torch.tensor]): (N, >=5) [x, y, z, batchid, value].
            sim ([torch.tensor]): (M, >=5) [x, y, z, batchid, value].
            Note: Total number of batches between data and sim shall be the same. We want to mimic data using sim.
        Returns:
            dict:
                'raw_data':   (N, >=5) [x, y, z, batchid, value]
                'gen_data':   (N, >=5) [x, y, z, batchid, value]
                'pred_data':  predicted scores for data, ordered by batch ids
                'pred_gen':   predicted scores for generated data
        """
        # Get the input
        data = input[0]
        sim  = input[1]
        device = data.device
        if sim.device!=device:
            raise ValueError('Input devices not consistent!')

        # Filling empty
        if self.filling_empty:
            data = filling_empty(data)
            sim  = filling_empty(sim)

        # shuffle if needed
        if self.shuffle_pairing:
            sim = shuffle_data(sim)

        # Get the generated data
        gen_data = self.generator(data)

        # Get the predictions
        pred_data = self.discriminator(data)
        pred_gen  = self.discriminator(gen_data)

        return {
            'raw_data':       [data],
            'gen_data':       [gen_data],
            'pred_data':      [pred_data],
            'pred_gen':       [pred_gen],
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