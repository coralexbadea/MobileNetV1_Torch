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
import argparse# import the argparse module (used for parsing command-line arguments & options)
import json# import the json module (used for encoding/decoding JSON data)
import os

import cv2
import torch
from PIL import Image# import the Image module from PIL library for image manipulation
from torch import nn
from torchvision.transforms import Resize, ConvertImageDtype, Normalize

import imgproc
import model
from utils import load_state_dict

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def load_class_label(class_label_file: str, num_classes: int) -> list:#function that loads class labels from a JSON file and returns them as a list
    class_label = json.load(open(class_label_file))#open the JSON file and loads the corresponding class label
    class_label_list = [class_label[str(i)] for i in range(num_classes)]#generate a list of class labels by mapping the numeric
                                                                        # indicess to their corresponding labels using the class_label dictionary

    return class_label_list


def choice_device(device_type: str) -> torch.device:#function that chooses the device for model processing (CPU or CUDA)
    # Select model processing equipment type
    if device_type == "cuda":# checks if the device is of type 'cuda'
        device = torch.device("cuda", 0)# assign a CUDA device to the variable 'device'
    else:
        device = torch.device("cpu")
    return device


def build_model(model_arch_name: str, model_num_classes: int, device: torch.device) -> [nn.Module, nn.Module]:##undefined
    mobilenet_v1_model = model.__dict__[model_arch_name](num_classes=model_num_classes)# build the specified model architecture
    mobilenet_v1_model = mobilenet_v1_model.to(device=device, memory_format=torch.channels_last) # move the model to the specified device

    return mobilenet_v1_model


def preprocess_image(image_path: str, image_size: int, device: torch.device) -> torch.Tensor:##undefined
    image = cv2.imread(image_path)#read the image from the specified path

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#convert the color format of the image from BGR to RGB

    # OpenCV convert PIL
    image = Image.fromarray(image)

    # Resize to 224
    image = Resize([image_size, image_size])(image)# resize the image to the specified size
    # Convert image data to pytorch format data
    tensor = imgproc.image_to_tensor(image, False, False).unsqueeze_(0)# convert the image to a tensor
    # Convert a tensor image to the given ``dtype`` and scale the values accordingly
    tensor = ConvertImageDtype(torch.float)(tensor)
    # Normalize a tensor image with mean and standard deviation.
    tensor = Normalize(args.model_mean_parameters, args.model_std_parameters)(tensor)#normalize the tensor with mean and standard deviation

    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)# move the tensor to the specified device

    return tensor


def main():
    # Get the label name corresponding to the drawing
    class_label_map = load_class_label(args.class_label_file, args.model_num_classes) # load class labels using the specified arguments

    device = choice_device(args.device_type) # choose the device based on the specified device type

    # Initialize the model
    mobilenet_v1_model = build_model(args.model_arch_name, args.model_num_classes, device)# build the model using the specified arguments
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    mobilenet_v1_model, _, _, _, _, _ = load_state_dict(mobilenet_v1_model, args.model_weights_path)# load the model weights using the specified arguments
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    mobilenet_v1_model.eval()# set the model to evaluation mode

    tensor = preprocess_image(args.image_path, args.image_size, device)# preprocess the image using the specified arguments

    # Inference
    with torch.no_grad():# disable gradient calculation
        output = mobilenet_v1_model(tensor)# perform inference using the preprocessed image

    # Calculate the five categories with the highest classification probability
    prediction_class_index = torch.topk(output, k=5).indices.squeeze(0).tolist() # retrieve the top-k predicted class indices

    # Print classification results
    for class_index in prediction_class_index:#iterate over the predicted class indices
        prediction_class_label = class_label_map[class_index]# retrieve the corresponding class label
        prediction_class_prob = torch.softmax(output, dim=1)[0, class_index].item() # calculate the probability of the predicted class
        print(f"{prediction_class_label:<75} ({prediction_class_prob * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()# create an ArgumentParser object
    parser.add_argument("--model_arch_name", type=str, default="mobilenet_v1") # define an argument '--model_arch_name' with type 'str' and default value 'mobilenet_v1'
    parser.add_argument("--model_mean_parameters", type=list, default=[0.485, 0.456, 0.406])# define an argument '--model_mean_parameters' with type 'list' and default value [0.485, 0.456, 0.406]
    parser.add_argument("--model_std_parameters", type=list, default=[0.229, 0.224, 0.225]) # define an argument '--model_std_parameters' with type 'list' and default value [0.229, 0.224, 0.225]
    parser.add_argument("--class_label_file", type=str, default="./data/ImageNet_1K_labels_map.txt")# define an argument '--class_label_file' with type 'str' and default value './data/ImageNet_1K_labels_map.txt'
    parser.add_argument("--model_num_classes", type=int, default=1000) # define an argument '--model_num_classes' with type 'int' and default value 1000
    parser.add_argument("--model_weights_path", type=str, default="./results/pretrained_models/MobileNetV1-ImageNet_1K.pth.tar") # Define an argument '--model_weights_path' with type 'str' and default value './results/pretrained_models/MobileNetV1-ImageNet_1K.pth.tar'
    parser.add_argument("--image_path", type=str, default="./figure/n01440764_36.JPEG") # define an argument '--image_path' with type 'str' and default value './figure/n01440764_36.JPEG'
    parser.add_argument("--image_size", type=int, default=224) # define an argument '--image_size' with type 'int' and default value 224
    parser.add_argument("--device_type", type=str, default="cpu", choices=["cpu", "cuda"]) # define an argument '--device_type' with type 'str', default value 'cpu', and choices ['cpu', 'cuda']
    args = parser.parse_args()

    main()
