# import statements

import kagglehub
from PIL.ImageOps import grayscale
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
for image_category_name_from_dataset in ["NORMAL","PNEUMONIA"]:
    category_folder_directory = os.path.join(download_dataset_path, image_category_name_from_dataset)
    labels_for_category = 0 if image_category_name_from_dataset == "NORMAL" else 1

for image_file_name in os.listdir(category_folder_directory):

    full_image_path = os.path.join(category_folder_directory, image_file_name)
    if not os.path.isfile(full_image_path):
        continue

    image_array_for_grayscale = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
    if image_array_for_grayscale is None:
        continue

    resized_image_array_for_grayscale = cv2.resize(image_array_for_grayscale(dataset_image_resized,dataset_image_resized))
    flattened_image_vector_for_grayscale = resized_image_array_for_grayscale.flatten()
    data_from_flattened_images_in_dataset.append(flattened_image_vector_for_grayscale)
    labels_from_flattened_images_in_dataset.append(labels_for_category)

# TODO: Convert to arrays

X = np.array(data_from_flattened_images_in_dataset)
y = np.array(labels_from_flattened_images_in_dataset)
print("Shape of the data from flattened image: ", X.shape)

# TODO: Train test split


# TODO: Apply KNN