# Discriminator of GAN
# based on encoder
import torch
from mlreco.models.layers.cnn_encoder import EncoderModel

class GANEncoderDiscriminator(torch.nn.Module):
    """
    Use a CNN encoder as the discriminator
    """
    def __init__(self, model_config):
        super(GANEncoderDiscriminator, self).__init__()

        # copy the config
        cfg = model_config.copy()
        # mandatorily change some config
        cfg['linear_layer'] = True # enable linear layer
        cfg['out_feat_num'] = 2    # 2 features come out for binary classification

        # Initialize the CNN
        self.encoder = EncoderModel(model_config)

        # Initialize the softmax
        self.log_softmax = torch.nn.LogSoftmax(dim=1)


    def forward(self, data)
        """
        Input:
            - data: (tensor) (N,5) [x,y,z,batch_id,value]
        Output:
            - score: (tensor) (N, 2) 
        """
        # extract features from encoder
        feats = self.encoder(data)
        # get score through softmax
        log_scores = self.log_softmax(feats)
        return torch.exp(log_scores)
