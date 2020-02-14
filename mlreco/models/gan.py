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

    def forward(self, data):
        """
        Args:
            data ([torch.tensor]): (N, >=6) [x, y, z, batchid, value, label (binary)]
        Returns:
            dict:
                'generated_data': (N, >=6) [x, y, z, batchid, value, label (binary)] if label=0, return the same data as input
                'label_pred': predicted label by discriminator
        """

