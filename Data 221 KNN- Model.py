# import statements

import kagglehub
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2
import numpy as np

# TODO: Load the dataset and define the subfolder path

download_dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", download_dataset_path)
train_dataset_path = os.path.join(download_dataset_path, "chest_xray", "train")
data_from_flattened_images_in_dataset= []
labels_from_flattened_images_in_dataset = []

dataset_image_resized = 100

# TODO: Change image into vector to work for KNN (RESIZE + FLATTEN)

# TODO: Convert to arrays

# TODO: Train test split

# TODO: Apply KNN