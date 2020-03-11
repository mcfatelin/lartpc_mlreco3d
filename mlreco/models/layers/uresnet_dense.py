###########################################
## Dense UResNet borrowed from
## https://github.com/Temigo/uresnet_pytorch/blob/master/uresnet/models/uresnet_dense.py
## Make a bit modification to be compatible with the lartpc framework
############################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

# Accelerate *if all input sizes are same*
# torch.backends.cudnn.benchmark = True


def get_conv(is_3d):
    if is_3d:
        return nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d
    else:
        return nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d


def padding(kernel, stride, input_size):
    if input_size[-1] % stride == 0:
        p = max(kernel - stride, 0)
    else:
        p = max(kernel - (input_size[-1] % stride), 0)
    p1 = int(p // 2)
    p2 = p - p1
    return (p1, p2,) * (len(input_size) - 2)


class ResNetModule(nn.Module):
    def __init__(self, is_3d, num_inputs, num_outputs, kernel=3, stride=1, bn_momentum=0.9):
        super(ResNetModule, self).__init__()
        fn_conv, fn_conv_transpose, batch_norm = get_conv(is_3d)
        self.kernel, self.stride = kernel, stride

        # Shortcut path
        self.use_shortcut = (num_outputs != num_inputs or stride != 1)
        self.shortcut = torch.nn.Sequential(
            fn_conv(
                in_channels = num_inputs,
                out_channels = num_outputs,
                kernel_size = 1,
                stride      = stride,
                padding     = 0
            ),
            batch_norm(num_features = num_outputs, momentum=bn_momentum, track_running_stats=False)
        )

        # residual path
        self.residual1 = torch.nn.Sequential(
            fn_conv(
                in_channels = num_inputs,
                out_channels = num_outputs,
                kernel_size = kernel,
                stride      = stride,
                padding     = 0
            ),
            batch_norm(num_features = num_outputs, momentum=bn_momentum, track_running_stats=False)
        )

        self.residual2 = torch.nn.Sequential(
            fn_conv(
                in_channels = num_outputs,
                out_channels = num_outputs,
                kernel_size = kernel,
                stride      = 1,
                padding     = 0
            ),
            batch_norm(num_features = num_outputs, momentum=bn_momentum, track_running_stats=False)
        )

    def forward(self, input_tensor):
        if not self.use_shortcut:
            shortcut = input_tensor
        else:
            shortcut = F.pad(input_tensor, padding(self.shortcut[0].kernel_size[0], self.shortcut[0].stride[0], input_tensor.size()), mode='replicate')
            shortcut = self.shortcut(shortcut)
        # FIXME padding value
        residual = F.pad(input_tensor, padding(self.residual1[0].kernel_size[0], self.residual1[0].stride[0], input_tensor.size()), mode='replicate')
        residual = self.residual1(residual)
        residual = F.pad(residual, padding(self.residual2[0].kernel_size[0], self.residual2[0].stride[0], residual.size()), mode='replicate')
        residual = self.residual2(residual)
        # print(self.shortcut[1].running_mean, self.shortcut[1].running_var)
        return F.relu(shortcut + residual)


class DoubleResnet(nn.Module):
    def __init__(self, is_3d, num_inputs, num_outputs, kernel=3, stride=1, bn_momentum=0.9):
        super(DoubleResnet, self).__init__()

        self.resnet1 = ResNetModule(
            is_3d = is_3d,
            num_inputs = num_inputs,
            num_outputs = num_outputs,
            kernel = kernel,
            stride = stride,
            bn_momentum = bn_momentum
        )
        self.resnet2 = ResNetModule(
            is_3d = is_3d,
            num_inputs = num_outputs,
            num_outputs = num_outputs,
            kernel = kernel,
            stride = 1,
            bn_momentum = bn_momentum
        )

    def forward(self, input_tensor):
        resnet = self.resnet1(input_tensor)
        resnet = self.resnet2(resnet)
        return resnet


class UResNet(nn.Module):
    """
    Configs can be

    """
    def __init__(self, cfg):
        super(UResNet, self).__init__()
        self._model_config = cfg

        # Parameters
        self.is_3d = self._model_config.get('data_dim', 3) == 3
        fn_conv, fn_conv_transpose, batch_norm = get_conv(self.is_3d)
        self.base_num_outputs = self._model_config.get('filters', 16)
        self.num_strides = self._model_config.get('num_strides', 5)
        self.num_inputs = 1  # number of channels of input image
        self.image_size = self._model_config.get('spatial_size', 768)
        self.num_classes = self._model_config.get('num_classes', 5)
        self.bn_momentum = self._model_config.get('bn_momentum', 0.9)

        # Define layers
        self.conv1 = torch.nn.Sequential(
            fn_conv(
                in_channels = self.num_inputs,
                out_channels = self.base_num_outputs,
                kernel_size = 3,
                stride = 1,
                padding = 0 # FIXME 'same' in tensorflow
            ),
            batch_norm(num_features=self.base_num_outputs, momentum=self.bn_momentum, track_running_stats=False),
            torch.nn.ReLU()
        )
        # Encoding steps
        self.double_resnet = nn.ModuleList()
        current_num_outputs = self.base_num_outputs
        for step in range(self.num_strides):
            self.double_resnet.append(DoubleResnet(
                is_3d = self.is_3d,
                num_inputs = current_num_outputs,
                num_outputs = current_num_outputs * 2,
                kernel = 3,
                stride = 2,
                bn_momentum = self.bn_momentum
            ))
            current_num_outputs *= 2

        # Decoding steps
        self.decode_conv = nn.ModuleList()
        self.decode_double_resnet = nn.ModuleList()
        for step in range(self.num_strides):
            self.decode_double_resnet.append(DoubleResnet(
                is_3d = self.is_3d,
                num_inputs = current_num_outputs,
                num_outputs = int(current_num_outputs / 2),
                kernel = 3,
                stride = 1,
                bn_momentum = self.bn_momentum
            ))
            self.decode_conv.append(torch.nn.Sequential(
                fn_conv_transpose(
                    in_channels = current_num_outputs,
                    out_channels = int(current_num_outputs / 2),
                    kernel_size = 3,
                    stride = 2,
                    padding=1,
                    output_padding=1
                ),
                batch_norm(num_features=int(current_num_outputs / 2), momentum=self.bn_momentum, track_running_stats=False),
                torch.nn.ReLU()
            ))
            current_num_outputs = int(current_num_outputs / 2)

        self.conv2 = torch.nn.Sequential(
            fn_conv(
                in_channels = current_num_outputs,
                out_channels = self.base_num_outputs,
                padding = 0,
                kernel_size = 3,
                stride = 1
            ),
            batch_norm(num_features=current_num_outputs, momentum=self.bn_momentum, track_running_stats=False),
            torch.nn.ReLU()
        )

        self.conv3 = torch.nn.Sequential(
            fn_conv(
                in_channels = self.base_num_outputs,
                out_channels = self.num_classes,
                padding = 0,
                kernel_size = 3,
                stride = 1
            ),
            batch_norm(num_features=self.num_classes, momentum=self.bn_momentum, track_running_stats=False)
        )

    def forward(self, input):
        """
        Can be 2D or 3D. Supports batch processing.
        input size: B, C, (N,) * dim
        """
        conv_feature_map = {}
        #net = input.view(-1,self.num_inputs,self.image_size,self.image_size,self.image_size)
        net = F.pad(input, padding(self.conv1[0].kernel_size[0], self.conv1[0].stride[0], input.size()), mode='replicate')
        net = self.conv1(net)
        conv_feature_map[net.size()[1]] = net
        # Encoding steps
        for step in range(self.num_strides):
            net = self.double_resnet[step](net)
            conv_feature_map[net.size()[1]] = net
        # Decoding steps
        for step in range(self.num_strides):
            # num_outputs = net.size()[1] / 2
            net = self.decode_conv[step](net)
            net = torch.cat((net, conv_feature_map[net.size()[1]]), dim=1)
            net = self.decode_double_resnet[step](net)
        # Final conv layers
        net = F.pad(net, padding(self.conv2[0].kernel_size[0], self.conv2[0].stride[0], net.size()), mode='replicate')
        net = self.conv2(net)
        net = F.pad(net, padding(self.conv3[0].kernel_size[0], self.conv3[0].stride[0], net.size()), mode='replicate')
        net = self.conv3(net)
        return net
