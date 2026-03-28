import os

import cv2
import sklearn
from sklearn.model_selection import train_test_split

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