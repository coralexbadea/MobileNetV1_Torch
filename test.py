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
import os
import time

import config
import model
import torch
from dataset import CUDAPrefetcher, ImageDataset
from torch import nn
from torch.utils.data import DataLoader
from utils import load_state_dict, accuracy, Summary, AverageMeter, ProgressMeter # import utility functions and classes for loading state dict, 
                                                                                  # calculating accuracy, summary, and average meter, and displaying progress

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def build_model() -> nn.Module: # method that builds the MobileNet V1 model
    mobilenet_v1_model = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes) # create an instance of the MobileNet V1 model by accessing 
                                                                                                      #the model architecture specified by `config.model_arch_name`
                                                                                                      # `num_classes` parameter is set to `config.model_num_classes`
    mobilenet_v1_model = mobilenet_v1_model.to(device=config.device, memory_format=torch.channels_last) # Move the model to the specified device (`config.device`) for computation
    # The `memory_format` is set to `torch.channels_last`, which optimizes memory layout for better performance.
    
    return mobilenet_v1_model


def load_dataset() -> CUDAPrefetcher: # method that loads the test dataset
    test_dataset = ImageDataset(config.test_image_dir,# testing dataset image directory
                                config.image_size,# image size
                                config.model_mean_parameters,# mean parameters
                                config.model_std_parameters,# standard deviation parameters
                                "Test")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return test_prefetcher


def main() -> None:
    # Initialize the model
    mobilenet_v1_model = build_model()
    print(f"Build `{config.model_arch_name}` model successfully.")

    # Load model weights
    mobilenet_v1_model, _, _, _, _, _ = load_state_dict(mobilenet_v1_model, config.model_weights_path)# load the model weights into the MobileNet V1 model using the `load_state_dict` function
                                                                                                      # the loaded weights are assigned to the `mobilenet_v1_model` variable
                                                                                                      # the additional variables (indicated by "_") are placeholders for potential 
                                                                                                      # additional information returned by the `load_state_dict` function
    print(f"Load `{config.model_arch_name}` "
          f"model weights `{os.path.abspath(config.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    mobilenet_v1_model.eval()

    # Load test dataloader
    test_prefetcher = load_dataset()

    # Calculate how many batches of data are in each Epoch
    batches = len(test_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE) # measure the time taken per batch
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE) # measure the top-1 accuracy
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE) # measure the top-5 accuracy
    progress = ProgressMeter(batches, [batch_time, acc1, acc5], prefix=f"Test: ")

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    test_prefetcher.reset()
    batch_data = test_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data["image"].to(device=config.device, non_blocking=True)# transfer the image data to the specified CUDA device
            target = batch_data["target"].to(device=config.device, non_blocking=True)# transfer the target data to the specified CUDA device
                                                                                     # the `non_blocking=True` flag enables asynchronous data transfer

            # Get batch size
            batch_size = images.size(0)# determine the size of the batch by accessing the size of the images tensor along the 0th dimension

            # Inference
            output = mobilenet_v1_model(images)# perform inference on the MobileNet V1 model using the input images and store the output

            # measure accuracy and record loss
            top1, top5 = accuracy(output, target, topk=(1, 5))# calculate the top-1 and top-5 accuracies by comparing the model's predictions (`output`) with the target labels (`target`)
            acc1.update(top1[0].item(), batch_size)# update the `acc1` AverageMeter with the top-1 accuracy for the current batch
            acc5.update(top5[0].item(), batch_size)# update the `acc5` AverageMeter with the top-5 accuracy for the current batch

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end) # update the `batch_time` AverageMeter with the time it took to process the current batch
            end = time.time()# computes the training end time

            # Write the data during training to the training log file
            if batch_index % config.test_print_frequency == 0:# check if the current batch index is a multiple of `config.test_print_frequency`
                progress.display(batch_index + 1)#display progress and metrics

            # Preload the next batch of data
            batch_data = test_prefetcher.next() # load the next batch of data from the test dataset
            batch_index += 1#increment the batch index

    # print metrics
    progress.display_summary()#print the metrics summary


if __name__ == "__main__":
    main()
