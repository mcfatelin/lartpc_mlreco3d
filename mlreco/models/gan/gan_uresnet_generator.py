# Generator of GAN
# based on UResNet
import torch
from mlreco.models.layers.uresnt import UResNet

class GANUResNetGenerator(torch.nn.Module):
    """
    Use a UResNet as the generator
    """
    def __init__(self, model_config):
        super(GANUResNetGenerator, self).__init__()

        # copy the config
        cfg = model_config.copy()

        # Initialize uresnet
        self.uresnet = UResNet(cfg)

    def forward(self, input):