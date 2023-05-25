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
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0) # set the random seed for reproducibility to 0
torch.manual_seed(0) # set the random seed for torch operations to 0
np.random.seed(0) # set the random seed for numpy operations to 0

# Use GPU for training by default
device = torch.device("cuda", 0) # set the device for training to GPU (cuda:0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True # Enable the cuDNN benchmark mode for faster training
# Model arch name
model_arch_name = "mobilenet_v1"  # set the name of the model architecture (MobileNetV1)
# Model normalization parameters
model_mean_parameters = [0.485, 0.456, 0.406] # define the mean values for normalization of input images
model_std_parameters = [0.229, 0.224, 0.225]  # define the standard deviation values for normalization of input images

# Model number class
model_num_classes = 1000 # set the number of classes in the model to 0
# Current configuration parameter method
mode = "train" # set the current mode of the configuration to training mode
# Experiment name, easy to save weights and log files
exp_name = f"{model_arch_name}-ImageNet_1K" # set the experiment name based on model architecture and dataset

if mode == "train":
    # Dataset address
    train_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_train"
    valid_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val"

    image_size = 224 # set the size of the input images (224x224)
    batch_size = 128 # set the batch size for training to 128
    num_workers = 4 # define the number of workers for data loading (4)

    # The address to load the pretrained model
    pretrained_model_weights_path = "" # path to load pre-trained model weights (initially empty for no pre-training)

    # Incremental training and migration training
    resume = "" # path to resume training from a saved checkpoint (empty for starting from scratch)

    # Total num epochs
    epochs = 600 # set the total number of training epochs to 600

    # Loss parameters
    loss_label_smoothing = 0.1 # set the label smoothing factor for the loss function to 0.1
    loss_weights = 1.0 # set the weight for the loss function to 1

    # Optimizer parameter
    model_lr = 0.1 # set the learning rate for the optimizer to 0.1
    model_momentum = 0.9  # set the momentum factor for the optimizer to 0.9
    model_weight_decay = 2e-05  # set the weight decay factor for the optimizer
    model_ema_decay = 0.99998  # set the exponential moving average decay factor for the optimizer

    # Learning rate scheduler parameter
    lr_scheduler_T_0 = epochs // 4  # define the number of iterations to decrease learning rate (T_0)
    lr_scheduler_T_mult = 1 # multiply T_0 by this factor after each restart (T_mult)
    lr_scheduler_eta_min = 5e-5 # set the minimum learning rate (eta_min) for the scheduler

    # How many iterations to print the training/validate result
    train_print_frequency = 200 # set the frequency of printing training results to 200
    valid_print_frequency = 20 # set the frequency of printing validation results to 20

if mode == "test": # if the model enters the testing mode
    # Test data address
    test_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val"  # get the directory path for test dataset

    # Test dataloader parameters
    image_size = 224 # set the size of the input images (224x224)
    batch_size = 256 # set the batch size for testing to 256
    num_workers = 4 # define the number of workers for data loading

    # How many iterations to print the testing result
    test_print_frequency = 20 # define the frequency of printing test results and set it to 20
    model_weights_path = "results/pretrained_models/Xception-ImageNet_1K-a0b40234.pth.tar" # get the path to the pretrained model weights file
