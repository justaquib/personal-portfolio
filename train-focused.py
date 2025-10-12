import tensorflow as tf
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split

# Focus on these key labels for better accuracy
focus_labels = [
  "apple", "book", "bowtie", "candle", "cloud", "cup", "door", "envelope",
  "eyeglasses", "guitar", "ice cream", "line", "moon", "mountain", "mouse",
  "parachute", "pencil", "smiley face", "star", "sun", "t-shirt", "tree",
  "umbrella", "watch", "wheel", "heart", "house", "car", "bicycle",
  "airplane", "train", "boat", "fish", "cat", "dog", "flower", "balloon",
  "cake", "chair", "table", "lamp", "phone", "key", "lock", "ring",
  "shoe", "hat", "glasses", "camera", "clock", "map", "palm tree",
  "rocket", "snowman", "treehouse", "volcano", "waterfall", "windmill"
]

print(f"Training on {len(focus_labels)} focused labels")

def create_high_quality_synthetic_data(num_samples_per_class=500, image_size=28):
    """
    Create high-quality synthetic doodle data with better patterns
    """
    X = []
    y = []

    for class_idx, label in enumerate(focus_labels):
        print(f"Generating {num_samples_per_class} samples for {label}...")

        for sample_idx in range(num_samples_per_class):
            # Create base image
            image = np.zeros((image_size, image_size, 1), dtype=np.float32)

            # Generate specific patterns for each class
            if label == "apple":
                # Draw apple shape
                center = [14, 14]
                for i in range(image_size):
                    for j in range(image_size):
                        dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                        if dist < 8 and dist > 2:
                            image[i, j] = 1.0
                # Add stem
                image[6:10, 14:15] = 1.0

            elif label == "book":
                # Draw book shape
                image[8:20, 6:22] = 1.0
                # Add pages
                image[10:18, 8:20] = 0.0
                image[10:18, 10:18] = 1.0

            elif label == "heart":
                # Draw heart shape
                for i in range(image_size):
                    for j in range(image_size):
                        # Heart formula
                        if ((i-14)**2 + (j-14)**2 - 25) * ((i-14)**2 + (j-18)**2 - 25) <= -100:
                            image[i, j] = 1.0

            elif label == "star":
                # Draw star
                center = [14, 14]
                for i in range(image_size):
                    for j in range(image_size):
                        angle = np.arctan2(i - center[0], j - center[1])
                        dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                        if dist < 10 and (np.sin(angle * 5) > 0.5 or dist < 4):
                            image[i, j] = 1.0

            elif label == "house":
                # Draw house
                image[12:22, 8:20] = 1.0  # base
                image[8:12, 12:16] = 1.0  # roof
                image[14:18, 10:12] = 0.0  # door
                image[14:16, 16:18] = 0.0  # window

            elif label == "car":
                # Draw car
                image[14:18, 6:22] = 1.0  # body
                image[12:14, 10:18] = 1.0  # roof
                image[16:18, 8:10] = 1.0   # wheel
                image[16:18, 18:20] = 1.0   # wheel

            elif label == "boat":
                # Draw boat
                image[16:18, 8:20] = 1.0  # hull
                image[14:16, 12:16] = 1.0  # cabin
                image[12:14, 14:15] = 1.0  # mast

            elif label == "watch":
                # Draw watch
                center = [14, 14]
                for i in range(image_size):
                    for j in range(image_size):
                        dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                        if 6 < dist < 10:
                            image[i, j] = 1.0
                # Hands
                image[14, 10:18] = 1.0
                image[10:18, 14] = 1.0

            elif label == "tree":
                # Draw tree
                image[16:22, 13:15] = 1.0  # trunk
                image[10:16, 10:18] = 1.0  # leaves

            elif label == "cat":
                # Draw cat
                image[14:18, 10:18] = 1.0  # body
                image[12:14, 12:16] = 1.0  # head
                image[10:12, 13:15] = 1.0  # ears
                image[16:18, 12:14] = 1.0  # tail

            elif label == "dog":
                # Draw dog
                image[14:18, 10:18] = 1.0  # body
                image[12:14, 12:16] = 1.0  # head
                image[16:18, 16:18] = 1.0  # tail
                image[14:16, 8:10] = 1.0   # legs

            elif label == "fish":
                # Draw fish
                image[14:16, 8:20] = 1.0  # body
                image[13:17, 18:20] = 1.0  # tail
                image[12:14, 8:10] = 1.0   # head

            elif label == "bird" in label.lower() or "parrot" in label or "eagle" in label:
                # Draw bird
                image[14:16, 10:18] = 1.0  # body
                image[12:14, 12:16] = 1.0  # head
                image[10:12, 14:16] = 1.0  # beak
                image[14:15, 8:10] = 1.0   # wing

            elif "circle" in label.lower() or "balloon" in label.lower():
                # Draw circle
                center = [14, 14]
                radius = np.random.randint(5, 9)
                y_coords, x_coords = np.ogrid[:image_size, :image_size]
                mask = (x_coords - center[0])**2 + (y_coords - center[1])**2 <= radius**2
                image[mask] = 1.0

            elif "square" in label.lower() or "chair" in label or "table" in label:
                # Draw rectangle/square
                start = np.random.randint(4, 10, 2)
                size = np.random.randint(8, 16, 2)
                image[start[0]:start[0]+size[0], start[1]:start[1]+size[1]] = 1.0

            elif "triangle" in label.lower():
                # Draw triangle
                height = np.random.randint(10, 18)
                for i in range(height):
                    width = int((i / height) * 12) + 2
                    start_col = image_size // 2 - width // 2
                    image[image_size - 1 - i, start_col:start_col + width] = 1.0

            elif "line" in label.lower():
                # Draw line
                if np.random.random() > 0.5:
                    row = np.random.randint(12, 16)
                    image[row, 4:24] = 1.0
                else:
                    col = np.random.randint(12, 16)
                    image[4:24, col] = 1.0

            else:
                # Generic pattern for other classes
                num_shapes = np.random.randint(2, 5)
                for _ in range(num_shapes):
                    shape_type = np.random.choice(['circle', 'line', 'dot'])
                    if shape_type == 'circle':
                        c = np.random.randint(5, 23, 2)
                        r = np.random.randint(2, 6)
                        y_coords, x_coords = np.ogrid[:image_size, :image_size]
                        mask = (x_coords - c[0])**2 + (y_coords - c[1])**2 <= r**2
                        image[mask] = 1.0
                    elif shape_type == 'line':
                        if np.random.random() > 0.5:
                            row = np.random.randint(5, 23)
                            start = np.random.randint(2, 10)
                            end = np.random.randint(18, 26)
                            image[row, start:end] = 1.0
                        else:
                            col = np.random.randint(5, 23)
                            start = np.random.randint(2, 10)
                            end = np.random.randint(18, 26)
                            image[start:end, col] = 1.0
                    else:  # dot
                        pos = np.random.randint(3, 25, 2)
                        image[pos[0], pos[1]] = 1.0

            # Add variations and noise
            # Random scaling
            scale = np.random.uniform(0.9, 1.1)
            if abs(scale - 1.0) > 0.01:
                from scipy.ndimage import zoom
                scaled = zoom(image.squeeze(), scale, order=1)
                # Ensure we maintain 28x28 size
                if scaled.shape[0] != 28 or scaled.shape[1] != 28:
                    # Resize to 28x28
                    from skimage.transform import resize
                    scaled = resize(scaled, (28, 28), mode='constant', anti_aliasing=True)
                image = scaled.reshape(28, 28, 1)

            # Random rotation
            angle = np.random.uniform(-15, 15)
            if abs(angle) > 1:
                from scipy.ndimage import rotate
                image = rotate(image.squeeze(), angle, reshape=False).reshape(image.shape)

            # Add noise
            noise = np.random.normal(0, 0.05, image.shape)
            image = np.clip(image + noise, 0, 1)

            # Random thickness variation
            thickness_factor = np.random.uniform(0.8, 1.5)
            image = np.clip(image * thickness_factor, 0, 1)

            X.append(image)
            y.append(class_idx)

    return np.array(X), np.array(y)

# Generate high-quality training data
print("Generating high-quality synthetic training data...")
X, y = create_high_quality_synthetic_data(num_samples_per_class=2000)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=len(focus_labels))
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=len(focus_labels))

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Create improved model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(focus_labels), activation='softmax')
])

# Compile with better optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Add callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
]

# Train the model
print("Training the improved model...")
history = model.fit(X_train, y_train_onehot,
                    epochs=100,
                    batch_size=64,
                    validation_data=(X_test, y_test_onehot),
                    callbacks=callbacks,
                    verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
print(".4f")

# Load best model
model = tf.keras.models.load_model('best_model.h5')

# Save in TensorFlow.js format
model.save('trained_model.h5')

# Convert to TensorFlow.js
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
    print("Model saved as HDF5. You'll need to convert it manually.")

# Clean up
os.remove('trained_model.h5')
os.remove('best_model.h5')