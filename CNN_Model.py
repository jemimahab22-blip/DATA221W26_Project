import os
import cv2
import kagglehub
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve)

# Downloading the latest version of the dataset
download_dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", download_dataset_path) # shows that dateset is being loaded

dataset_image_resized = 100
data_from_images_in_dataset = []
labels_from_images_in_dataset = []

def load_image(base_path, size):
    train_dataset_path = os.path.join(base_path, "chest_xray", "train")  # the dataset path
    image_category_name_from_dataset = ["NORMAL", "PNEUMONIA"]  # what the category names are

    # this for loop goes through the image categories
    for category in image_category_name_from_dataset:
        category_folder_directory = os.path.join(train_dataset_path, category)
        labels_for_category = 0 if category == "NORMAL" else 1

        # checks if the path exists
        if not (os.path.exists(category_folder_directory)):
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
            normalized_image_array_for_grayscale = resized_image_array_for_grayscale / 255.0

            # appending data to the files for the image and labels
            data_from_images_in_dataset.append(normalized_image_array_for_grayscale)
            labels_from_images_in_dataset.append(labels_for_category)

# function call
load_image(download_dataset_path, dataset_image_resized)

# Converts to Arrays
X = np.array(data_from_images_in_dataset)
y = np.array(labels_from_images_in_dataset)

print("Shape of the data from image: ", X.shape)
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

# reshaping image specific for CNN because it needs to be 4D
X_train = X_train.reshape(-1,100,100,1)
X_val = X_val.reshape(-1,100,100,1)
X_test = X_test.reshape(-1,100,100,1)

print("Shape of CNN input: ", X_train.shape)

# Building the CNN
cnn_model_for_dataset = Sequential([Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)),
                        MaxPooling2D(pool_size=(2,2)),

                        Conv2D(64, (3,3), activation='relu'),
                        MaxPooling2D(pool_size=(2,2)),

                        Flatten(),

                        Dense(128, activation='relu'),
                        Dropout(0.5),

                        Dense(1, activation='sigmoid')
                    ])

# Compiling the CNN model
cnn_model_for_dataset.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# training the CNN model
training_cnn_model = cnn_model_for_dataset.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_val, y_val)
)

# TODO: Predictions
# Generate probabilities 0.0 - 1.0
y_prediction_probabilities = cnn_model_for_dataset.predict(X_test)
y_predictions = (y_prediction_probabilities > 0.5).astype('int32')

# TODO: Evaluation for CNN + Confusion Matrix
print('\n~~~ Model Evaluation ~~~')
test_accuracy = accuracy_score(y_test, y_predictions)
test_recall = recall_score(y_test, y_predictions)
test_precision = precision_score(y_test, y_predictions)
test_f1 = f1_score(y_test, y_predictions)
test_roc_auc = roc_auc_score(y_test, y_prediction_probabilities)
# TODO: Display of evaluations