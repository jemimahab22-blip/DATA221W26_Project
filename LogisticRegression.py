import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)

# Download the dataset and configure it
download_dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", download_dataset_path)

dataset_image_resized = 100
data_from_flattened_images_in_dataset = []
labels_from_flattened_images_in_dataset = []

# Load data and preprocess
# This goes through the Kaggle dataset, resizes, and flattens the images.
def load_and_flatten_images(base_path, size):
    folders = ['train', 'test', 'val']
    categories = ['NORMAL', 'PNEUMONIA']

    dataset_root = os.path.join(base_path, 'chest_xray')

    for folder in folders:
        for category in categories:
            folder_path = os.path.join(dataset_root, folder, category)
            # 0 represents normal, 1 represents pneumonia
            class_number = categories.index(category)

            if not os.path.exists(folder_path):
                continue

            for image_name in os.listdir(folder_path):
                try:
                    image_path = os.path.join(folder_path, image_name)
                    # Now load it in as greyscale
                    image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image_array is None:
                        continue

                    # Resize image to 100x100
                    resized_array = cv2.resize(image_array, (size, size))

                    # Flatten image to 1D array
                    flattened_image = resized_array.flatten()

                    data_from_flattened_images_in_dataset.append(flattened_image)
                    labels_from_flattened_images_in_dataset.append(class_number)
                except Exception as e:
                    pass

print(f"\nProcessing images (Resizing to {dataset_image_resized}x{dataset_image_resized})...")
load_and_flatten_images(download_dataset_path, dataset_image_resized)

# Convert data to numerical arrays
pneumonia_feature_matrix = np.array(data_from_flattened_images_in_dataset)
pneumonia_label_vector = np.array(labels_from_flattened_images_in_dataset)

print(f"Total images loaded: {len(pneumonia_feature_matrix)}")
print(f"Pixels per image (features): {pneumonia_feature_matrix.shape[1]}")
