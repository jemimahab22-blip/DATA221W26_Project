import kagglehub
import tensorflow as tf
import os

# 1. Download latest version of the dataset
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", path)

# 2. Define the FULL subfolder paths explicitly
# The Kaggle dataset structure is: path / $chest_xray$ / [train, val, test]
train_directory_path = os.path.join(path, "chest_xray", "train")
validation_directory_path = os.path.join(path, "chest_xray", "val")
test_directory_path = os.path.join(path, "chest_xray", "test")

print(f"Loading training data from: {train_directory_path}")
print(f"Loading validation data from: {validation_directory_path}")
print(f"Loading test data from: {test_directory_path}")

# 3. Load the datasets using Keras
# We'll use 150x150 as our uniform dimension for the preprocessing pipeline
training_dataset = tf.keras.utils.image_dataset_from_directory(
    train_directory_path,
    image_size=(150, 150),
    batch_size=32,
    color_mode='grayscale',
    label_mode='binary'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_directory_path,
    image_size=(150, 150),
    batch_size=32,
    color_mode='grayscale',
    label_mode='binary'
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_directory_path,
    image_size=(150, 150),
    batch_size=32,
    color_mode='grayscale',
    label_mode='binary'
)

# 4. Normalization (Rescale pixel values from 0-255 to 0-1)
# our group's proposal prioritizes this for stable training!