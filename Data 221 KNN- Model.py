# import statements
import kagglehub
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

# Downloading the latest version of the dataset
download_dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", download_dataset_path) # shows that dateset is being loaded

dataset_image_resized = 100 # resizing the image to 100

# where the data from the flattened images get appended to
data_from_flattened_images_in_dataset= []

# where the labels from the flattened images get appended to
labels_from_flattened_images_in_dataset = []

# created a function to load and flatten the images for KNN similar format used in other models as well
def load_and_flatten_image_for_knn(base_path, size):
    train_dataset_path = os.path.join(base_path, "chest_xray", "train")  # the dataset path
    image_category_name_from_dataset = ["NORMAL", "PNEUMONIA"] # what the category names are

# this for loop goes through the image categories
    for category in image_category_name_from_dataset:
        category_folder_directory = os.path.join(train_dataset_path, category)
        labels_for_category = 0 if category == "NORMAL" else 1

# checks if the path exists
        if not(os.path.exists(category_folder_directory)):
            continue

# for loop that checks for conditions in

        for image_file_name in os.listdir(category_folder_directory):
            full_image_path = os.path.join(category_folder_directory, image_file_name)

            if not os.path.isfile(full_image_path):
                continue

# loads and converts image to greyscale so 1 channel only and not all RGB just brightness
            image_array_for_grayscale = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
            if image_array_for_grayscale is None:
                continue

            # resizes image to 100 x 100
            resized_image_array_for_grayscale = cv2.resize(image_array_for_grayscale, (size, size))
            # normalizes image
            normalized_image_array_for_grayscale = resized_image_array_for_grayscale/255.0
            # flattens image to 1D array
            flattened_image_vector_for_grayscale = normalized_image_array_for_grayscale.flatten()
            # appending data to the files for the image and labels
            data_from_flattened_images_in_dataset.append(flattened_image_vector_for_grayscale)
            labels_from_flattened_images_in_dataset.append(labels_for_category)

# Function call
load_and_flatten_image_for_knn(download_dataset_path, dataset_image_resized)


# converts to Arrays
X = np.array(data_from_flattened_images_in_dataset)
y = np.array(labels_from_flattened_images_in_dataset)

print("Shape of the data from flattened image: ", X.shape)
print("Total number of images loaded: ", len(X))

# this test is for Data splitting and stratification (70/15/15)
# Splits the data into 3 distinct sets: training, testing, and validation

X_intermediate, X_test, y_intermediate, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

# Also a data splitting set , but now it is 70% training and 15% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_intermediate,
    y_intermediate,
    test_size=0.1765, # 0.1765 of 85% in ~15% of the original
    random_state=42,
    stratify=y_intermediate
)

# for calculations
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Training for KNN model")

# applying KNN logic
knn_model_for_dataset = KNeighborsClassifier(n_neighbors=5)
knn_model_for_dataset.fit(X_train_scaled, y_train)

predictions_for_dataset = knn_model_for_dataset.predict(X_val_scaled)
accuracy_for_dataset = accuracy_score(y_val, predictions_for_dataset)
print(f"The Validation Accuracy is : {accuracy_for_dataset}")

print("\nTesting KNN for test set...")
test_predictions_for_dataset = knn_model_for_dataset.predict(X_test_scaled)
test_probability_for_dataset = knn_model_for_dataset.predict_proba(X_test_scaled)[:, 1]

# Calculating Metrics for KNN
test_accuracy = accuracy_score(y_test, test_predictions_for_dataset)
test_recall = recall_score(y_test, test_predictions_for_dataset)
test_roc_auc_score = roc_auc_score(y_test, test_probability_for_dataset)
test_precision = precision_score(y_test, test_predictions_for_dataset)
test_f1_score = f1_score(y_test, test_predictions_for_dataset)
# Display Calculating Metrics for KNN
print('~' * 30)
print('KNN Performance:')
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Recall: {test_recall}")
print(f"Test Precision: {test_precision}")
print(f"Test F1-Score: {test_f1_score}")
print('ROC-AUC:', test_roc_auc_score)
print('~' * 30)