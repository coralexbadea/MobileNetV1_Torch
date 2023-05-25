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
import queue # import the queue module for data prefetching
import sys
import threading # import the threading module for parallel execution
from glob import glob # import the glob module for file path matching

import cv2 # import OpenCV library for image processing
import torch
from PIL import Image # import the Image module from PIL library for image manipulation
from torch.utils.data import Dataset, DataLoader # import necessary classes for dataset loading and data loading
from torchvision import transforms # import transformations for data augmentation
from torchvision.datasets.folder import find_classes # import function to find classes in a dataset folder
from torchvision.transforms import TrivialAugmentWide # import custom augmentation transformation

import imgproc

__all__ = [
    "ImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")

# The delimiter is not the same between different platforms
if sys.platform == "win32":
    delimiter = "\\"
else:
    delimiter = "/"


class ImageDataset(Dataset): # define a class that handles the image dataset of the model
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): Image size.
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir: str, image_size: int, mean: list, std: list, mode: str) -> None:
        super(ImageDataset, self).__init__()
        # Iterate over all image paths
        self.image_file_paths = glob(f"{image_dir}/*/*") # get the paths of all image files in the dataset
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(image_dir) # form class-label pairs based on folder structure
        self.image_size = image_size # size of the images
        self.mode = mode # dataset loading mode
        self.delimiter = delimiter

        if self.mode == "Train": # if the model is in training mode
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                TrivialAugmentWide(), # custom augmentation transformation
                transforms.RandomRotation([0, 270]),# set the random rotation of the transformation 
                transforms.RandomHorizontalFlip(0.5), # set the random horizontal flip of the transformation
                transforms.RandomVerticalFlip(0.5), # set the random vertical flip of the transformation
            ])
        elif self.mode == "Valid" or self.mode == "Test":
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop([self.image_size, self.image_size]),
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"

        self.post_transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean, std)
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]: # define a getitem method for the ImageDataset class that takes a batch index, and returns a list containing a tensor and an int
        image_dir, image_name = self.image_file_paths[batch_index].split(self.delimiter)[-2:] # split image file path into directory and file name
        # Read a batch of image data
        if image_name.split(".")[-1].lower() in IMG_EXTENSIONS:# check if the image file extension is supported
            image = cv2.imread(self.image_file_paths[batch_index]) # read image using OpenCV
            target = self.class_to_idx[image_dir] # get the class label based on the folder name
        else:
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, "
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert image from BGR to RGB

        # OpenCV convert PIL
        image = Image.fromarray(image) # convert image from OpenCV format to PIL format

        # Data preprocess
        image = self.pre_transform(image) # apply pre-processing transformations

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = imgproc.image_to_tensor(image, False, False) # convert image to tensor

        # Data postprocess
        tensor = self.post_transform(tensor)  # apply post-processing transformations

        return {"image": tensor, "target": target} # return the image after the applied transformations into tensor stream format

    def __len__(self) -> int:
        return len(self.image_file_paths)


class PrefetchGenerator(threading.Thread): # define a data prefetch generator class
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None: # define an initialization method for the PrefetchGenerator class
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)  # create a data prefetch queue
        self.generator = generator # set the data prefetch generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item) # put the items from the generator into the data prefetch queue
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader): # define a fast data prefetch dataloader class
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None: # define the initialization method for the prefetch data loader
        self.num_data_prefetch_queue = num_data_prefetch_queue # store the number of data prefetch queues
        super(PrefetchDataLoader, self).__init__(**kwargs) 

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher: # define a CPU prefetcher class used to accelerate data reading
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler,
            and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher: # define a CUDA prefetcher class used to accelerate data reading
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device): # initialization method for the CUDA prefetcher
        self.batch_data = None # variable to store the batch data
        self.original_dataloader = dataloader # tore the original dataloader
        self.device = device  # store the specified device

        self.data = iter(dataloader) # create an iterator over the dataloader
        self.stream = torch.cuda.Stream() # create a CUDA stream
        self.preload() # preload the first batch of data

    def preload(self): # method that preloads batches of data
        try: # try to fetch the next batch of data
            self.batch_data = next(self.data) # get the next batch data
        except StopIteration: # if all branches have been fetched
            self.batch_data = None # set batch_data to None
            return None # return None

        with torch.cuda.stream(self.stream): # execute the following operations in the CUDA stream
            for k, v in self.batch_data.items(): # iterate over the items in the batch data
                if torch.is_tensor(v): # if the item is a tensor
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True) # move the tensor to the specified device

    def next(self):# method that returns the current batch data and preloads the next one
        torch.cuda.current_stream().wait_stream(self.stream)  # wait for the CUDA stream to finish
        batch_data = self.batch_data #store the current batch data
        self.preload() # preload the next batch of data
        return batch_data # return the current batch data

    def reset(self): # method for resetting the CUDA prefetcher
        self.data = iter(self.original_dataloader) # reset the data iterator to the beginning
        self.preload()  # preload the first batch of data

    def __len__(self) -> int:# method that returns the length of the original dataloader
        return len(self.original_dataloader)  # return the length of the original dataloader
