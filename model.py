# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Callable, Any, Optional

import torch
from torch import Tensor #import the Tensor module from torch
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation#import the Conv2dNormActivation module from torchvision.ops.misc

__all__ = [ # define a list that specifies the names of the classes and functions that should be considered public and accessible from outside the module
    "MobileNetV1",
    "DepthWiseSeparableConv2d",
    "mobilenet_v1",
]


class MobileNetV1(nn.Module):# create a class that defines the MobileNetV1 model

    def __init__(
            self,
            num_classes: int = 1000,#set the number of classes to 1000
    ) -> None:
        super(MobileNetV1, self).__init__()#call the parent MobileNetV1 initialization method
        self.features = nn.Sequential( #module named self.features is created and initialized with a sequence of layers. 
            Conv2dNormActivation(3,#set number of input channels to 3
                                 32,#set number of output channels to 32
                                 kernel_size=3,#set kernel size to 3
                                 stride=2,#set the stride to 2
                                 padding=1,# set the padding to 1
                                 norm_layer=nn.BatchNorm2d,#set the normalization layer tko BatchNorm2d
                                 activation_layer=nn.ReLU,#set the activation layer to ReLu
                                 inplace=True,#inplace activation
                                 bias=False,#there is no bias
                                 ),

            DepthWiseSeparableConv2d(32, 64, 1),#create a depth-wise separable convolutional layer with 32 input channels, 64 output channels, stride of 1 
            DepthWiseSeparableConv2d(64, 128, 2),#create a depth-wise separable convolutional layer with 64 input channels, 128 output channels, stride of 2  
            DepthWiseSeparableConv2d(128, 128, 1),#create a depth-wise separable convolutional layer with 128 input channels, 128 output channels, stride of 1 
            DepthWiseSeparableConv2d(128, 256, 2),#create a depth-wise separable convolutional layer with 128 input channels, 256 output channels, stride of 2 
            DepthWiseSeparableConv2d(256, 256, 1),#create a depth-wise separable convolutional layer with 256 input channels, 256 output channels, stride of 1 
            DepthWiseSeparableConv2d(256, 512, 2),#create a depth-wise separable convolutional layer with 256 input channels, 512 output channels, stride of 2 
            DepthWiseSeparableConv2d(512, 512, 1),#create a depth-wise separable convolutional layer with 512 input channels, 512 output channels, stride of 1
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 1024, 2),
            DepthWiseSeparableConv2d(1024, 1024, 1),
        )

        self.avgpool = nn.AvgPool2d((7, 7))# create an average pooling layer with a kernel size of 7x7

        self.classifier = nn.Linear(1024, num_classes) # create a linear layer that maps the 1024-dimensional input features to the specified number of classes

        # Initialize neural network weights
        self._initialize_weights() # call the `_initialize_weights()` method to initialize the weights of the model's module

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x) # perform the actual forward pass computation of the MobileNetV1 model on tensor x

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:#implementation of the forward pass for the MobileNetV1 model
        out = self.features(x)#compute the intermediate output
        out = self.avgpool(out) # apply average pooling to the intermediate output
        out = torch.flatten(out, 1)# flatten the tensor `out` along the second dimension (1) to create a 1D representation of the features
        out = self.classifier(out)#perform linear mapping to the specified number of classes and generate the output

        return out

    def _initialize_weights(self) -> None:##undefined
        for module in self.modules():# iterate over all modules in the MobileNetV1 model, including the sub-modules
            if isinstance(module, nn.Conv2d): # check if the current module is an instance of the `nn.Conv2d` class
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu") # initialize the weights of the convolutional layer using the Kaiming normal initialization, which takes into account the nonlinearity "ReLU" for fan-out mode
            if module.bias is not None:  # Check if the convolutional layer has a bias parameter
                nn.init.zeros_(module.bias)# initialize the bias parameter to zeros
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)): # check if the current module is an instance of either `nn.BatchNorm2d` or `nn.GroupNorm`
                nn.init.ones_(module.weight)  # initialize the weight parameter of the normalization layer to ones
                nn.init.zeros_(module.bias)  # initialize the bias parameter of the normalization layer to zeros
            elif isinstance(module, nn.Linear):# check if the current module is an instance of the `nn.Linear` class
                nn.init.normal_(module.weight, 0, 0.01)# Initialize the weights of the linear layer using the normal distribution with mean 0 and standard deviation 0.01
                nn.init.zeros_(module.bias)# Initialize the bias parameter of the linear layer to zeros


class DepthWiseSeparableConv2d(nn.Module):#class that inherits from nn.Module and represents a depthwise separable convolutional layer
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None #the object of this class is initialized as having an optional normalization layer
    ) -> None:
        super(DepthWiseSeparableConv2d, self).__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:  # check if the normalization layer is not specified
            norm_layer = nn.BatchNorm2d # use `nn.BatchNorm2d` as the default normalization layer

        self.conv = nn.Sequential( # define a sequential container to hold the convolutional layers
            Conv2dNormActivation(in_channels, # apply a convolutional layer with specified parameters:
                                 in_channels, #input channels same as input
                                 kernel_size=3, #kernel size of 3
                                 stride=stride,
                                 padding=1,
                                 groups=in_channels, # number of groups same as input channels (depthwise convolution)
                                 norm_layer=norm_layer, # normalization layer
                                 activation_layer=nn.ReLU, # activation layer
                                 inplace=True, # inplace activation
                                 bias=False, #no bias
                                 ),
            Conv2dNormActivation(in_channels,# apply another convolutional layer with specified parameters
                                 out_channels,# output channels
                                 kernel_size=1,# kernel size of 1
                                 stride=1,#stride of 1
                                 padding=0,#padding of 0
                                 norm_layer=norm_layer,#normalization layer
                                 activation_layer=nn.ReLU,#activation layer
                                 inplace=True,#inplace activation
                                 bias=False,#no bias
                                 ),

        )

    def forward(self, x: Tensor) -> Tensor: #forward pass of the DepthWiseSeparableConv2d
        out = self.conv(x)##undefined

        return out


def mobilenet_v1(**kwargs: Any) -> MobileNetV1: # function that constructs and returns an instance of the MobileNetV1 class
    model = MobileNetV1(**kwargs) #a new instance of class MobileNetV1 is created, and it allows passing any number of keyword arguments

    return model#return the model instance
