import os
import cv2
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
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

# Data splitting and stratification (70/15/15)
# Split the data into 3 distinct sets: training, testing, and validation

# 1st split (15% for final test set):
features_intermediate, features_test, labels_intermediate, labels_test = train_test_split(
    pneumonia_feature_matrix,
    pneumonia_label_vector,
    test_size=0.15,
    random_state=42,
    stratify=pneumonia_label_vector
)

# 2nd split (70% training and 15% validation):
features_train, features_val, labels_train, labels_val = train_test_split(
    features_intermediate,
    labels_intermediate,
    test_size=0.1765, # 0.1765 of 85% in ~15% of the original
    random_state=42,
    stratify=labels_intermediate
)

# Data scaling for leakage prevention
# Calculate mean and standard deviation only on the training set
pixel_scaler = StandardScaler()

features_train_standardized = pixel_scaler.fit_transform(features_train)
features_val_standardized = pixel_scaler.transform(features_val)
features_test_standardized = pixel_scaler.transform(features_test)

# Train the model
print("\nTraining Logistic Regression Baseline...")
logistic_baseline_model = LogisticRegression(max_iter=1000, random_state=42)  # 100x100 pixels need more iterations to find best fit, hence max_iter should be high
logistic_baseline_model.fit(features_train_standardized, labels_train)

# Evaluate model on test set
print("\nEvaluating Model on Test Set...")
test_predictions = logistic_baseline_model.predict(features_test_standardized)  #0 or 1 for confusion matrix and F1 score

# Probability scores
test_prediction_probabilities = logistic_baseline_model.predict_proba(features_test_standardized) [:, 1]

# Calculations of final metrics
test_accuracy = accuracy_score(labels_test, test_predictions)
test_recall = recall_score(labels_test, test_predictions)
test_roc_auc_score = roc_auc_score(labels_test, test_prediction_probabilities)
test_precision = precision_score(labels_test, test_predictions)
test_f1_score = f1_score(labels_test, test_predictions)

print('~' * 30)
print('LOGISTIC REGRESSION PERFORMANCE:')
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Recall: {test_recall}")
print(f"Test Precision: {test_precision}")
print(f"Test F1-Score: {test_f1_score}")
print('ROC-AUC:', test_roc_auc_score)
print('~' * 30)

# Visualization
# Confusion matrix to see where the model failed to correctly classify the images
pneumonia_confusion_matrix = confusion_matrix(labels_test, test_predictions)

plt.figure(figsize=(6, 4))
# Make confusion matrix readable
sns.heatmap(pneumonia_confusion_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])

plt.title('Logistic Regression (Baseline) Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
