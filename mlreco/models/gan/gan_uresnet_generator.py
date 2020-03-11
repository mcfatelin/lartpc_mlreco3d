# Generator of GAN
# based on UResNet (dense)
import torch
from mlreco.models.layers.uresnet_dense import UResNet

class GANUResNetGenerator(torch.nn.Module):
    """
    Use a UResNet as the generator
    """
    def __init__(self, model_config):
        super(GANUResNetGenerator, self).__init__()

        # get the batch size
        self.batch_size = model_config.get('batch_size', 32)
        self.image_size = model_config.get('spatial_size', 768)

        # Initialize the input_tensor
        self.input_tensor = torch.zeros(
            self.batch_size,
            1, # in current framework only 1 depth
            self.image_size,
            self.image_size,
            self.image_size,
            dtype=torch.float
        )

        # Initialize uresnet
        self.uresnet = UResNet(model_config)

    def forward(self, input):
        """
        Input is the same as sparse Uresnet
        Need to turn it to dense one her
        Return an image with the same size as an input
        """
        # debug use
        print("input device = "+str(input.device))
        print("internal tensor device = "+str(self.input_tensor.device))
        print("input type = "+str(input.dtype))
        print("internal tensor type = "+str(self.input_tensor.dtype))
        # check if the device is the same between the input and internal tensor
        if input.device!=self.input_tensor.device:
            self.input_tensor = self.input_tensor.to(input.device)
        # make self.input_tensor zeros
        self.input_tensor[:,:,:,:,:]=0
        # fill the value
        self.fill_input_tensor(input)

        # May need an adaptor
        net =  self.uresnet(self.input_tensor)

        # Make sparse to dense
        return self.dense_to_sparse(net)

    def fill_input_tensor(self, input):
        """
        Function for filling internal input_tensor which is the dense tensor to be fed into uresnet_dense
        input - (tensor) (N, 5) [x,y,z,batch_id,value]
        """
        self.input_tensor[
            input[:,3].long(),
            0,
            input[:,0].long(),
            input[:,1].long(),
            input[:,2].long(),
        ] = input[:,4].float()
        return

    def sparse_to_dense(self, net):
        """
        Function to transfer dense tensor into sparse format
        5D tensor -> 2D (N,5)
        """
        # get non zero value indexes (N, 5) tensor
        inds = (net>0).nonzero()
        # initialize a 2d tensor for output
        sparse_tensor = torch.zeros(
            inds.size()[0],
            5,
            dtype=net.dtype,
            device=net.device,
        )
        # fill the sparse tensor
        sparse_tensor[:,:3] = inds[:,2:].type(net.dtype)
        sparse_tensor[:,3]  = inds[:,0].type(net.dtype)
        sparse_tensor[:,4]  = self.input_tensor[
            inds[:,0],
            inds[:,1],
            inds[:,2],
            inds[:,3],
            inds[:,4]
        ]
        return sparse_tensor


