import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import io

# Load the labels
labels = [
  "apple",
  "book",
  "bowtie",
  "candle",
  "cloud",
  "cup",
  "door",
  "envelope",
  "eyeglasses",
  "guitar",
  "ice cream",
  "line",
  "moon",
  "mountain",
  "mouse",
  "parachute",
  "pencil",
  "smiley face",
  "star",
  "sun",
  "t-shirt",
  "tree",
  "umbrella",
  "watch",
  "wheel",
  "heart",
  "house",
  "car",
  "bicycle",
  "airplane",
  "train",
  "boat",
  "fish",
  "cat",
  "dog",
  "flower",
  "balloon",
  "cake",
  "chair",
  "table",
  "lamp",
  "phone",
  "key",
  "lock",
  "ring",
  "shoe",
  "hat",
  "glasses",
  "camera",
  "clock",
  "map",
  "palm tree",
  "rocket",
  "snowman",
  "treehouse",
  "volcano",
  "waterfall",
  "windmill",
  "zebra",
  "elephant",
  "giraffe",
  "kangaroo",
  "koala",
  "lion",
  "monkey",
  "panda",
  "penguin",
  "rabbit",
  "tiger",
  "whale",
  "dolphin",
  "shark",
  "octopus",
  "crab",
  "lobster",
  "starfish",
  "seahorse",
  "coral",
  "jellyfish",
  "butterfly",
  "bee",
  "ant",
  "spider",
  "snail",
  "caterpillar",
  "dragonfly",
  "grasshopper",
  "ladybug",
  "worm",
  "bat",
  "owl",
  "eagle",
  "parrot",
  "swan",
  "peacock",
  "flamingo",
  "turkey",
  "chicken",
  "rooster",
  "duck",
  "goose",
  "frog",
  "toad",
  "lizard",
  "snake",
  "tortoise",
  "alligator",
  "crocodile",
  "hippopotamus",
  "rhinoceros",
  "buffalo",
  "bison",
  "camel",
  "donkey",
  "horse",
  "pig",
  "sheep",
  "goat",
  "cow",
  "chameleon",
  "iguana",
]

print(f"Loaded {len(labels)} labels")

def load_doodle_dataset(csv_path, max_samples_per_class=1000):
    """
    Load doodle dataset from CSV file
    Expected format: CSV with columns for drawing data and labels
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset with {len(df)} samples")

        X = []
        y = []

        # Group by label/class
        for class_idx, label in enumerate(labels):
            # Filter data for this class
            class_data = df[df['word'] == label]
            if len(class_data) == 0:
                print(f"Warning: No data found for class '{label}'")
                continue

            # Limit samples per class
            class_data = class_data.head(max_samples_per_class)

            for _, row in class_data.iterrows():
                try:
                    # Parse the drawing data (assuming it's stored as string representation of strokes)
                    # This will need to be adjusted based on the actual CSV format
                    drawing_data = json.loads(row['drawing'])

                    # Convert drawing to 28x28 image
                    image = strokes_to_image(drawing_data)
                    X.append(image)
                    y.append(class_idx)

                except Exception as e:
                    print(f"Error processing sample for {label}: {e}")
                    continue

            print(f"Loaded {len(class_data)} samples for {label}")

        return np.array(X), np.array(y)

    except FileNotFoundError:
        print(f"Dataset file not found: {csv_path}")
        print("Please download the dataset from Kaggle and place it in the correct location")
        return None, None

def strokes_to_image(strokes, size=28):
    """
    Convert stroke data to 28x28 grayscale image
    """
    # Create blank image
    image = np.zeros((size, size), dtype=np.float32)

    for stroke in strokes:
        x_coords = stroke[0]
        y_coords = stroke[1]

        # Normalize coordinates to 0-27 range
        x_coords = np.array(x_coords, dtype=np.float32)
        y_coords = np.array(y_coords, dtype=np.float32)

        x_coords = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min() + 1e-6) * (size - 1)
        y_coords = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min() + 1e-6) * (size - 1)

        # Draw lines between consecutive points
        for i in range(len(x_coords) - 1):
            x1, y1 = int(x_coords[i]), int(y_coords[i])
            x2, y2 = int(x_coords[i + 1]), int(y_coords[i + 1])

            # Simple line drawing
            image[y1, x1] = 1.0
            image[y2, x2] = 1.0

    # Add channel dimension
    return image.reshape(size, size, 1)

# Try to load the dataset
csv_path = "path/to/doodle/dataset.csv"  # Update this path
X, y = load_doodle_dataset(csv_path)

if X is None:
    print("Creating synthetic data for demonstration...")
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
    # Fallback: save weights manually
    print("Falling back to manual weight saving...")

    # Get weights
    weights = model.get_weights()

    # Create weights manifest
    manifest = {
        "weights": [],
        "paths": ["./doodleModel.weights.bin"]
    }

    # Write weights to binary file
    with open('public/assets/model/doodleModel.weights.bin', 'wb') as f:
        for i, weight in enumerate(weights):
            weight_data = weight.flatten().astype(np.float32)
            f.write(weight_data.tobytes())

            manifest["weights"].append({
                "name": f"layer_{i}",
                "shape": list(weight.shape),
                "dtype": "float32"
            })

    # Update the model.json with correct weights manifest
    with open('public/assets/model/doodleModel.json', 'r') as f:
        model_json = json.load(f)

    model_json["weightsManifest"] = [manifest]

    with open('public/assets/model/doodleModel.json', 'w') as f:
        json.dump(model_json, f, indent=2)

    print("Model weights saved manually!")