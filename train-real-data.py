import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
import requests
import ndjson
from sklearn.model_selection import train_test_split
from PIL import Image
import io

# Load the labels
labels = [
  "apple",
  "star",
  "moon",
  "mountain",
  "cup",
  "tree",
  "house",
  "snowman",
  "hat",
  "camera",
  "sun",
  "cloud",
  "umbrella",
  "face",
  "banana",
  "car",
  "bicycle",
  "fish",
  "flower",
  "heart",
  "book",
  "pencil",
  "clock",
  "phone",
  "key",
]

print(f"Loaded {len(labels)} labels")

def load_quickdraw_dataset(labels, max_samples_per_class=1000):
    """
    Load Quick Draw dataset directly from Google Cloud Storage
    """
    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"

    X = []
    y = []

    for class_idx, label in enumerate(labels):
        # Convert label to filename format (spaces become nothing, special chars handled)
        filename = label.replace(" ", "").lower() + ".ndjson"
        url = base_url + filename

        print(f"Downloading {label} from {url}...")

        try:
            response = requests.get(url)
            response.raise_for_status()

            # Parse NDJSON data
            data = ndjson.loads(response.text)

            # Limit samples per class
            samples = data[:max_samples_per_class]

            processed_count = 0
            for item in samples:
                try:
                    # Extract drawing data
                    drawing_data = item['drawing']

                    # Convert drawing to 28x28 image
                    image = strokes_to_image(drawing_data)
                    X.append(image)
                    y.append(class_idx)
                    processed_count += 1

                except Exception as e:
                    print(f"Error processing sample for {label}: {e}")
                    continue

            print(f"Loaded {processed_count} samples for {label}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {label}: {e}")
            continue

    return np.array(X), np.array(y)

def strokes_to_image(strokes, size=28):
    """
    Convert Quick Draw stroke data to 28x28 grayscale image
    Each stroke is [x_coords, y_coords, timestamps]
    """
    # Create blank image
    image = np.zeros((size, size), dtype=np.float32)

    # Collect all coordinates to find bounding box
    all_x = []
    all_y = []

    for stroke in strokes:
        if len(stroke) >= 2:  # Ensure we have x and y coordinates
            x_coords = np.array(stroke[0], dtype=np.float32)
            y_coords = np.array(stroke[1], dtype=np.float32)
            all_x.extend(x_coords)
            all_y.extend(y_coords)

    if not all_x or not all_y:
        return image.reshape(size, size, 1)

    # Find bounding box
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Avoid division by zero
    width = max_x - min_x if max_x > min_x else 1
    height = max_y - min_y if max_y > min_y else 1

    # Draw each stroke
    for stroke in strokes:
        if len(stroke) >= 2:
            x_coords = np.array(stroke[0], dtype=np.float32)
            y_coords = np.array(stroke[1], dtype=np.float32)

            # Normalize to 0-27 range with some padding
            x_coords = ((x_coords - min_x) / width) * (size - 4) + 2
            y_coords = ((y_coords - min_y) / height) * (size - 4) + 2

            # Clip to image bounds
            x_coords = np.clip(x_coords, 0, size - 1)
            y_coords = np.clip(y_coords, 0, size - 1)

            # Draw lines between consecutive points with thickness
            for i in range(len(x_coords) - 1):
                x1, y1 = int(x_coords[i]), int(y_coords[i])
                x2, y2 = int(x_coords[i + 1]), int(y_coords[i + 1])

                # Draw thicker lines by setting neighboring pixels
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx1, ny1 = x1 + dx, y1 + dy
                        nx2, ny2 = x2 + dx, y2 + dy
                        if 0 <= nx1 < size and 0 <= ny1 < size:
                            image[ny1, nx1] = 1.0
                        if 0 <= nx2 < size and 0 <= ny2 < size:
                            image[ny2, nx2] = 1.0

    # Add channel dimension
    return image.reshape(size, size, 1)

# Load the Quick Draw dataset directly from Google Cloud Storage
print("Loading Quick Draw dataset from Google Cloud Storage...")
X, y = load_quickdraw_dataset(labels, max_samples_per_class=2000)

if len(X) == 0:
    print("Failed to load any data from Quick Draw dataset. Creating synthetic data for demonstration...")
    # Fallback to synthetic data if real dataset not available
    def create_synthetic_data(num_samples_per_class=500, image_size=28):
        X = []
        y = []

        for class_idx, label in enumerate(labels):
            print(f"Generating data for {label}...")

            for _ in range(num_samples_per_class):
                # Create a simple pattern based on the class
                image = np.zeros((image_size, image_size, 1), dtype=np.float32)

                # Generate different patterns for different classes
                if 'circle' in label.lower() or 'balloon' in label.lower():
                    center = np.random.randint(5, 23, 2)
                    radius = np.random.randint(3, 8)
                    y_coords, x_coords = np.ogrid[:image_size, :image_size]
                    mask = (x_coords - center[0])**2 + (y_coords - center[1])**2 <= radius**2
                    image[mask] = 1.0

                elif 'square' in label.lower() or 'house' in label.lower():
                    start = np.random.randint(2, 20, 2)
                    size = np.random.randint(5, 15, 2)
                    image[start[0]:start[0]+size[0], start[1]:start[1]+size[1]] = 1.0

                elif 'triangle' in label.lower():
                    height = np.random.randint(8, 20)
                    for i in range(height):
                        width = int((i / height) * 10) + 1
                        start_col = image_size // 2 - width // 2
                        image[image_size - 1 - i, start_col:start_col + width] = 1.0

                elif 'line' in label.lower():
                    if np.random.random() > 0.5:
                        row = np.random.randint(10, 18)
                        image[row, 5:23] = 1.0
                    else:
                        col = np.random.randint(10, 18)
                        image[5:23, col] = 1.0

                elif 'heart' in label.lower():
                    for i in range(image_size):
                        for j in range(image_size):
                            if ((i-14)**2 + (j-14)**2 - 25) * ((i-14)**2 + (j-18)**2 - 25) <= -100:
                                image[i, j] = 1.0

                elif 'star' in label.lower():
                    center = [14, 14]
                    for i in range(image_size):
                        for j in range(image_size):
                            angle = np.arctan2(i - center[0], j - center[1])
                            dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                            if dist < 8 and (np.sin(angle * 5) > 0.3 or dist < 3):
                                image[i, j] = 1.0

                else:
                    num_pixels = np.random.randint(20, 100)
                    indices = np.random.choice(image_size * image_size, num_pixels, replace=False)
                    image.flat[indices] = 1.0

                # Add some noise
                noise = np.random.normal(0, 0.1, image.shape)
                image = np.clip(image + noise, 0, 1)

                X.append(image)
                y.append(class_idx)

        return np.array(X), np.array(y)

    X, y = create_synthetic_data(num_samples_per_class=500)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=len(labels))
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=len(labels))

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='valid'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='valid'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train_onehot,
                    epochs=30,
                    batch_size=32,
                    validation_data=(X_test, y_test_onehot),
                    verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
print(".4f")

# Save the model in TensorFlow.js format
# First save as HDF5, then convert
model.save('trained_model.h5')

# Use tensorflowjs_converter command line tool
import subprocess
result = subprocess.run([
    'tensorflowjs_converter',
    '--input_format=keras',
    'trained_model.h5',
    'public/assets/model'
], capture_output=True, text=True)

if result.returncode == 0:
    print("Model saved in TensorFlow.js format!")
    print("You can now use the trained model in your web application.")
else:
    print("Error converting model:", result.stderr)
    # Fallback: save weights manually with correct layer names
    print("Falling back to manual weight saving...")

    # Get weights
    weights = model.get_weights()

    # Create weights manifest with correct layer names from topology
    manifest = {
        "weights": [
            {
                "name": "conv2d_1/kernel",
                "shape": list(weights[0].shape),
                "dtype": "float32"
            },
            {
                "name": "conv2d_1/bias",
                "shape": list(weights[1].shape),
                "dtype": "float32"
            },
            {
                "name": "conv2d_2/kernel",
                "shape": list(weights[2].shape),
                "dtype": "float32"
            },
            {
                "name": "conv2d_2/bias",
                "shape": list(weights[3].shape),
                "dtype": "float32"
            },
            {
                "name": "dense_1/kernel",
                "shape": list(weights[4].shape),
                "dtype": "float32"
            },
            {
                "name": "dense_1/bias",
                "shape": list(weights[5].shape),
                "dtype": "float32"
            },
            {
                "name": "dense_2/kernel",
                "shape": list(weights[6].shape),
                "dtype": "float32"
            },
            {
                "name": "dense_2/bias",
                "shape": list(weights[7].shape),
                "dtype": "float32"
            }
        ],
        "paths": ["./doodleModel.weights.bin"]
    }

    # Write weights to binary file
    with open('public/assets/model/doodleModel.weights.bin', 'wb') as f:
        for weight in weights:
            weight_data = weight.flatten().astype(np.float32)
            f.write(weight_data.tobytes())

    # Create a new model.json with correct topology and weights manifest
    # The issue is that the old model.json has units: 123 but we trained with 18 classes
    model_json = {
        "format": "layers-model",
        "generatedBy": "Quick Draw Training Script",
        "convertedBy": "Manual conversion",
        "modelTopology": {
            "keras_version": "2.1.6",
            "backend": "tensorflow",
            "model_config": {
                "class_name": "Sequential",
                "config": {
                    "name": "doodle_model",
                    "layers": [
                        {
                            "class_name": "Conv2D",
                            "config": {
                                "name": "conv2d_1",
                                "trainable": True,
                                "batch_input_shape": [None, 28, 28, 1],
                                "dtype": "float32",
                                "filters": 32,
                                "kernel_size": [3, 3],
                                "strides": [1, 1],
                                "padding": "valid",
                                "data_format": "channels_last",
                                "dilation_rate": [1, 1],
                                "activation": "relu",
                                "use_bias": True,
                                "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                                "bias_initializer": {"class_name": "Zeros", "config": {}}
                            }
                        },
                        {
                            "class_name": "MaxPooling2D",
                            "config": {
                                "name": "max_pooling2d_1",
                                "trainable": True,
                                "pool_size": [2, 2],
                                "padding": "valid",
                                "strides": [2, 2],
                                "data_format": "channels_last"
                            }
                        },
                        {
                            "class_name": "Conv2D",
                            "config": {
                                "name": "conv2d_2",
                                "trainable": True,
                                "filters": 64,
                                "kernel_size": [3, 3],
                                "strides": [1, 1],
                                "padding": "valid",
                                "data_format": "channels_last",
                                "dilation_rate": [1, 1],
                                "activation": "relu",
                                "use_bias": True,
                                "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                                "bias_initializer": {"class_name": "Zeros", "config": {}}
                            }
                        },
                        {
                            "class_name": "MaxPooling2D",
                            "config": {
                                "name": "max_pooling2d_2",
                                "trainable": True,
                                "pool_size": [2, 2],
                                "padding": "valid",
                                "strides": [2, 2],
                                "data_format": "channels_last"
                            }
                        },
                        {
                            "class_name": "Flatten",
                            "config": {
                                "name": "flatten_1",
                                "trainable": True
                            }
                        },
                        {
                            "class_name": "Dense",
                            "config": {
                                "name": "dense_1",
                                "trainable": True,
                                "units": 128,
                                "activation": "relu",
                                "use_bias": True,
                                "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                                "bias_initializer": {"class_name": "Zeros", "config": {}}
                            }
                        },
                        {
                            "class_name": "Dense",
                            "config": {
                                "name": "dense_2",
                                "trainable": True,
                                "units": len(labels),  # Use actual number of labels
                                "activation": "softmax",
                                "use_bias": True,
                                "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                                "bias_initializer": {"class_name": "Zeros", "config": {}}
                            }
                        }
                    ]
                }
            }
        },
        "weightsManifest": [manifest]
    }

    with open('public/assets/model/doodleModel.json', 'w') as f:
        json.dump(model_json, f, indent=2)

    print("Model weights saved manually with correct layer names and topology!")