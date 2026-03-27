# import statements
import kagglehub
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

# TODO: Load the dataset and define the subfolder path

download_dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", download_dataset_path)

dataset_image_resized = 100

data_from_flattened_images_in_dataset= []
labels_from_flattened_images_in_dataset = []
def load_and_flatten_image_for_knn(base_path, size):
    train_dataset_path = os.path.join(base_path, "chest_xray", "train")
    image_category_name_from_dataset = ["NORMAL", "PNEUMONIA"]

    for category in image_category_name_from_dataset:
        category_folder_directory = os.path.join(train_dataset_path, category)
        labels_for_category = 0 if category == "NORMAL" else 1

        if not(os.path.exists(category_folder_directory)):
            continue

        for image_file_name in os.listdir(category_folder_directory):
            full_image_path = os.path.join(category_folder_directory, image_file_name)

            if not os.path.isfile(full_image_path):
                continue

            image_array_for_grayscale = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
            if image_array_for_grayscale is None:
                continue
            resized_image_array_for_grayscale = cv2.resize(image_array_for_grayscale, (size, size))
            normalized_image_array_for_grayscale = resized_image_array_for_grayscale/255.0
            flattened_image_vector_for_grayscale = normalized_image_array_for_grayscale.flatten()
            data_from_flattened_images_in_dataset.append(flattened_image_vector_for_grayscale)
            labels_from_flattened_images_in_dataset.append(labels_for_category)



# TODO: Convert to arrays

X = np.array(data_from_flattened_images_in_dataset)
y = np.array(labels_from_flattened_images_in_dataset)
print("Shape of the data from flattened image: ", X.shape)

# TODO: Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO: Apply KNN
knn_for_dataset = KNeighborsClassifier()
knn_for_dataset.fit(X_train, y_train)
predictions_for_dataset = knn_for_dataset.predict(X_test)
accuracy_for_dataset = accuracy_score(y_test, predictions_for_dataset)
print("Accuracy of KNN: ", accuracy_for_dataset)