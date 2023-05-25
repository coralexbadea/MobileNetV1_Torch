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
from typing import Any
from torch import Tensor
from numpy import ndarray
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F_vision

__all__ = [
    "image_to_tensor", "tensor_to_image",
    "center_crop", "random_crop", "random_rotate", "random_vertically_flip", "random_horizontally_flip",
]


def image_to_tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor: #function that converts the image data type to the tensort data type
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (torch.Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("example_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, False, False)

    """
    # Convert image data type to Tensor data type
    tensor = F_vision.to_tensor(image) #convert image data type to Tensor data type

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm: #if range_norm is true
        tensor = tensor.mul(2.0).sub(1.0) #scale the tensor values from range [0, 1] to [-1, 1] by multiplying each element by 2.0 and subtracting 1.0

    # Convert torch.float32 image data type to torch.half image data type
    if half: #if half is true
        tensor = tensor.half() #convert the tensor's data type from torch.float32 to torch.half(lower precision floating point format)

    return tensor


def tensor_to_image(tensor: torch.Tensor, range_norm: bool, half: bool) -> Any: #function that converts the tensor data type to image data type
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (torch.Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_tensor = torch.randn([1,3, 256, 256], dtype=torch.float)
        >>> example_image = tensor_to_image(example_tensor, False, False)

    """
    # Scale the image data from [-1, 1] to [0, 1]
    if range_norm: #if range_norm is true
        tensor = tensor.add(1.0).div(2.0) #scale the tensor values from range [-1, 1] to [0, 1] by adding 1.0 to each element and then dividing it by 2.0
    # Convert torch.float32 image data type to torch.half image data type
    if half: #if half is true
        tensor = tensor.half() #convert the tensor's data type from torch.float32 to torch.half(lower precision floating point format)

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8") #perform a series of operations on a tensor to convert it back into an image represented as a numpy array of type uuint8

    return image


def center_crop( ##undefined
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        patch_size: int,
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"

    if input_type == "Tensor": #if the input is of type Tensor
        image_height, image_width = images[0].size()[-2:] # Get the height and width of the tensor image
    else:
        image_height, image_width = images[0].shape[0:2] # Get the height and width of the NumPy array image

    # Calculate the start indices of the crop
    top = (image_height - patch_size) // 2  # Calculate the top index for center cropping
    left = (image_width - patch_size) // 2  # Calculate the left index for center cropping

    # Crop lr image patch
    if input_type == "Tensor": #if the input is a Tensor
        images = [image[ #create a list of cropped images
                  :,
                  :,
                  top:top + patch_size,# Crop the tensor image with the calculated indices
                  left:left + patch_size] for image in images] #repeat for every image
    else:
        images = [image[ #create a list of cropped images
                  top:top + patch_size,  # Crop the NumPy array image with the calculated top indices
                  left:left + patch_size, # -||- with the calculated left indices
                  ...] for image in images] #do this for each image in the image list

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images


def random_crop( #function that performs random cropping on input images
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        patch_size: int,
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"

    if input_type == "Tensor":
        image_height, image_width = images[0].size()[-2:]
    else:
        image_height, image_width = images[0].shape[0:2]

    # Just need to find the top and left coordinates of the image
    top = random.randint(0, image_height - patch_size)
    left = random.randint(0, image_width - patch_size)

    # Crop lr image patch
    if input_type == "Tensor":
        images = [image[ #create list of cropped images
                  :,
                  :,
                  top:top + patch_size,
                  left:left + patch_size] for image in images]
    else:
        images = [image[ #create list of cropped images
                  top:top + patch_size,
                  left:left + patch_size,
                  ...] for image in images]

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images


def random_rotate( ##undefined
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        angles: list,
        center: tuple = None,
        rotate_scale_factor: float = 1.0
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    # Random select specific angle
    angle = random.choice(angles)

    if not isinstance(images, list): ##undefined
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy" #get the input image data type

    if input_type == "Tensor": #if the input type of the image is a Tensor
        image_height, image_width = images[0].size()[-2:] # Get the height and width of the tensor image
    else:
        image_height, image_width = images[0].shape[0:2] # Get the height and width of the numpy image

    # Rotate LR image
    if center is None: #if the center isn't provided
        center = (image_width // 2, image_height // 2) # use the center of the image

    matrix = cv2.getRotationMatrix2D(center, angle, rotate_scale_factor)# Generate a rotation matrix using the specified center, angle, and rotation scale factor

    if input_type == "Tensor": # if the input type is a Tensor
        images = [F_vision.rotate(image, angle, center=center) for image in images] # Rotate the tensor image using the specified angle and center
    else:
        images = [cv2.warpAffine(image, matrix, (image_width, image_height)) for image in images] # Apply the rotation matrix to the NumPy array image

    # When image number is 1
    if len(images) == 1: #if the images list contains only one image
        images = images[0]

    return images


def random_horizontally_flip( #function that performs random horizontal flipping on images
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        p: float = 0.5
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    # Get horizontal flip probability
    flip_prob = random.random()

    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            images = [F_vision.hflip(image) for image in images]
        else:
            images = [cv2.flip(image, 1) for image in images]

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images


def random_vertically_flip( #function that performs random vertical flipping on images
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        p: float = 0.5
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    # Get vertical flip probability
    flip_prob = random.random()

    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            images = [F_vision.vflip(image) for image in images]
        else:
            images = [cv2.flip(image, 0) for image in images]

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images
