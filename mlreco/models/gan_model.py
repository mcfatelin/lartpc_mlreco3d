########################################
## GAN for generating fake label 1 image from label-0 event
########################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from mlreco.utils.gan.data import filling_empty, shuffle_data, image_difference_score
from mlreco.models.gan.factories import gan_construct

class GAN(torch.nn.Module):
    """
    Class for GAN.
    Generator model using UResNet
    Discriminator model using Autoencoder+MLP

    for use in config:
    model:
        name: gan
        modules:
            shuffle_pairing: shuffle the pairing between data and sim in each batch, default False.
            filling_empty:   whether to fill the empty data and sim using random data and sim in the batch. Default True for training.
            generator:
                name: 'uresnet'
                <uresnet config>
            discriminator:
                name: 'cnn'
                <cnn_encoder config>
    """
    def __init__(self, cfg):
        super(GAN, self).__init__()

        # Initialize the generator & discriminator
        self.generator     = gan_construct(cfg['generator'])
        self.discriminator = gan_construct(cfg['discriminator'])

        # extra flags
        self.shuffle_pairing    = cfg['chain'].get('shuffle_pairing', False)
        self.filling_empty      = cfg['chain'].get('filling_empty', True)

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
        gen_data = self.generator(sim)

        # Get the predictions
        pred_data = self.discriminator(data)
        pred_gen  = self.discriminator(gen_data)

        # Order the predictions
        # first get batch ids for raw and gen data
        batch_ids_raw = data[:,3].unique(sorted=False).flip(0)
        batch_ids_gen = gen_data[:,3].unique(sorted=False).flip(0)

        # Secondly, rearrange in ascending order
        pred_data = pred_data[batch_ids_raw.argsort()]
        pred_gen  = pred_gen[batch_ids_gen.argsort()]

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
            loss_aggr     : <way of aggregating the two losses in gan, default "concat", optional "sum">
            generator:
                <uresnet config>
            discriminator:
                <cnn_encoder config>
    """
    def __init__(self, cfg):
        super(GANLoss, self).__init__()

        # Set the loss
        self.loss               = cfg['chain'].get('loss', 'CE')
        self.reduction          = cfg['chain'].get('reduction', 'mean')
        self.gen_loss_factor    = cfg['chain'].get('gen_loss_factor', None)
        self.loss_aggr          = cfg['chain'].get('loss_aggr', 'concat')
        self.image_size         = cfg['chain'].get('image_size', 1024)

        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = chain_config.get('p', 1)
            margin = chain_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('Loss not recognized: ' + self.loss)

        if self.loss_aggr not in ['concat', 'sum']:
            raise Exception('Not supported loss aggregation method! Shall be either \'concat\' or \'sum\' ')


    def forward(self, out, inputs):
        """
        Applies the requested loss for gan
        The goal is to discriminate between data and simulation as much as possible
        In the meantime can label the generated data as data as much as possible

        Args:
            out (dict):
                'raw_data':   (N, >=5) [x, y, z, batchid, value]
                'gen_data':   (N, >=5) [x, y, z, batchid, value]
                'pred_data':  predicted scores for data, ordered by batch ids
                'pred_gen':   predicted scores for generated data
        Returns:
            loss:     tensor (2) or double
            accuracy: double
        NOTE: Currently it doesn't support multi-GPU
        """
        device = inputs[0][0].device
        total_loss, total_acc = 0, 0.
        if self.loss_aggr=='concat':
            total_loss = torch.tensor([0., 0.], requires_grad=True, device=device)
        for i in range(len(inputs)):
            # Get the predictions
            pred_raw = out['pred_data'][i]
            pred_gen = out['pred_gen'][i]

            # Loss 1: discriminator loss
            loss1 = -(pred_raw[:,1].log() + torch.log(1-pred_gen[:,1])).mean()

            # Loss 2: generator loss
            loss2 = (torch.log(1-pred_gen[:,1])).mean()
            if self.gen_loss_factor!=None:
                # add another loss to minimize the generator feature
                loss2 += self.gen_loss_factor * image_difference_score(
                    out['raw_data'][i],
                    out['gen_data'][i],
                    self.image_size
                )

            # Accuracy
            acc = len(pred_raw[pred_raw[:,1]>0.5,:])/len(pred_raw) + len(pred_gen[pred_gen[:,1]<0.5,:])/len(pred_gen)

            # aggregate to total
            total_acc += acc
            if self.loss_aggr=='concat':
                total_loss += torch.cat([loss1,loss2])
                total_loss.retain_grad()
            else:
                total_loss += loss1 + loss2
        return {
            'accuracy': total_acc/len(inputs),
            'loss':     total_loss/len(inputs),
        }



