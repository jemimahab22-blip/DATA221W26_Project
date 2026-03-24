# import statements
import kagglehub
from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np

# Download the latest version of the dateset
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)

# TODO: Change image into vector to work for KNN (RESIZE + FLATTEN)

# TODO: Convert to arrays

# TODO: Train test split

# TODO: Apply KNN