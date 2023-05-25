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
import csv # import the csv module for reading CSV files
import os

from PIL import Image

train_csv_path = "../data/MiniImageNet_1K/original/train.csv" # path to the train.csv file
valid_csv_path = "../data/MiniImageNet_1K/original/valid.csv" # path to the valid.csv file
test_csv_path = "../data/MiniImageNet_1K/original/test.csv" # path to the test.csv file

inputs_images_dir = "../data/MiniImageNet_1K/original/mini_imagenet/images"  # path to the directory containing the input images
output_images_dir = "../data/MiniImageNet_1K/"

train_label = {} # dictionary to store the training labels
val_label = {} # dictionary to store the validation labels
test_label = {} # dictionary to store the test labels

with open(train_csv_path) as csvfile: # open the training CSV file
    csv_reader = csv.reader(csvfile) # create a csv_reader object to read the file
    birth_header = next(csv_reader) # read the header row and skip it
    for row in csv_reader:  # iterate over each row of the csv file
        train_label[row[0]] = row[1] # assign the value in the second column to the train_label dictionary using the value in the first column as the key

with open(valid_csv_path) as csvfile: # open the validation CSV file
    csv_reader = csv.reader(csvfile) # create a csv_reader object to read the file
    birth_header = next(csv_reader) # read the header row and skip it
    for row in csv_reader: # iterate over each row of the csv file
        val_label[row[0]] = row[1] # Assign the value in the second column to the val_label dictionary using the value in the first column as the key

with open(test_csv_path) as csvfile: # open the test CSV file
    csv_reader = csv.reader(csvfile) # create a csv_reader object to read the file
    birth_header = next(csv_reader) # read the header row and skip it
    for row in csv_reader: # iterate over each row of the csv file
        test_label[row[0]] = row[1] # assign the value in the second column to the test_label dictionary using the value in the first column as the key

for png in os.listdir(inputs_images_dir): # iterate over each file in the input images directory
    path = inputs_images_dir + "/" + png # create the complete file path
    im = Image.open(path) # open the image and store it in the variable im
    if png in train_label.keys(): # check if the image filename exists as a key in the training label dictionary
        tmp = train_label[png] # get the corresponding label from the training label dictionary
        temp_path = output_images_dir + "/train" + "/" + tmp # create the directory path for saving the processed image
        if not os.path.exists(temp_path): # check if the directory path doesn't exist
            os.makedirs(temp_path) # if it doesn't, create the directory
        t = temp_path + "/" + png # create the complete file path for saving the processed image
        im.save(t)

    elif png in val_label.keys():  # check if the image filename exists as a key in the validation label dictionary
        tmp = val_label[png] # get the corresponding label from the validation label dictionary
        temp_path = output_images_dir + "/valid" + "/" + tmp  # create the directory path for saving the processed image
        if not os.path.exists(temp_path): # check if the directory path doesn't exist
            os.makedirs(temp_path) # if it doesn't, create the directory
        t = temp_path + "/" + png
        im.save(t) # save the processed image

    elif png in test_label.keys(): # check if the image filename exists as a key in the test label dictionary
        tmp = test_label[png] # get the corresponding label from the test label dictionary
        temp_path = output_images_dir + "/test" + "/" + tmp # create the directory path for saving the processed image
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        t = temp_path + "/" + png
        im.save(t)
