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
