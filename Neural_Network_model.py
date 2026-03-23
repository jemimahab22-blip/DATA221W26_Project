import kagglehub
import tensorflow as tf
import os

# 1. Download latest version of the dataset
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", path)

# 2. Define the FULL subfolder paths explicitly
# The Kaggle dataset structure is: path / chest_xray / [train, val, test]
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

normalization_layer = tf.keras.layers.Rescaling(1./255)
training_dataset = training_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

from tensorflow.keras import layers, models

# 1. Define the Multi-Layered Architecture
# We use 'Sequential' to stack layers one after another

model = models.Sequential([
    # Input Layer: Flattens the 150x150 grayscale image into a 1D vector of 22,500 pixels
    layers.Flatten(input_shape=(150, 150, 1)),

# Hidden Layer 1: Learns initial patterns with 128 neurons and ReLU activation
layers.Dense(128, activation='relu'),

    # Hidden Layer 2: Adds depth to capture more complex textures
layers.Dense(64, activation='relu'),

    # Hidden Layer 3: Final refinement layer
layers.Dense(32, activation='relu'),

    # Output Layer: Single neuron for Binary Classification (Normal vs Pneumonia)
    # We use 'sigmoid' to get a probability between 0 and 1
layers.Dense(1, activation='sigmoid')
])

#  Compile the Model
# We use binary_crossentropy because this is a binary task!
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

# Train the model
# We use 'epochs=10' to start, which means the model sees the data 10 times
# our proposal emphasizes a 'fair testing environment'

history = model.fit(
    training_dataset,
    validation_data =validation_dataset,
    epochs=10
)

# Final Evaluation on the Test Set
print("\n--- Final Test Set Evaluation ---")
model.evaluate(test_dataset)