# import statements

import kagglehub
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2
import numpy as np

# TODO: Load the dataset and define the subfolder path

path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", path)
train_path = os.path.join(path, "chest_xray", "train")
data_from_images_in_dataset= []
labels_from_images_in_dataset = []

IMAGE_SIZE = 100

# TODO: Change image into vector to work for KNN (RESIZE + FLATTEN)

# TODO: Convert to arrays

# TODO: Train test split

# TODO: Apply KNN