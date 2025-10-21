# Doodle Predictor Training Guide

## Overview
This doodle predictor uses a Convolutional Neural Network (CNN) trained on the Google Quick Draw dataset to recognize hand-drawn doodles in real-time.

## Current Model
- **Architecture**: CNN with 2 convolutional layers, max pooling, and dense layers
- **Input**: 28x28 grayscale images
- **Output**: 23 classes (apple, star, moon, mountain, cup, tree, house, snowman, hat, camera, sun, cloud, umbrella, face, banana, car, bicycle, fish, flower, book, pencil, clock, key)
- **Accuracy**: ~98.8% on validation set

## Training Data
The model is trained on real Quick Draw data from Google Cloud Storage:
- **Source**: `https://storage.googleapis.com/quickdraw_dataset/full/simplified/`
- **Format**: NDJSON files containing stroke-based vector drawings
- **Samples**: 2,000 samples per class (46,000 total training samples)

## How to Train the Model Manually

### Prerequisites
1. **Python Environment**: Python 3.8+
2. **Required Packages**:
   ```bash
   pip install tensorflow numpy pandas scikit-learn pillow requests ndjson
   ```

### Training Steps

1. **Set up the environment**:
   ```bash
   cd web-app
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install tensorflow numpy pandas scikit-learn pillow requests ndjson
   ```

2. **Run the training script**:
   ```bash
   python3 train-real-data.py
   ```

   This will:
   - Download Quick Draw datasets for all 13 classes
   - Convert stroke data to 28x28 images
   - Train the CNN model for 30 epochs
   - Save the model in TensorFlow.js format

3. **Monitor training progress**:
   - The script shows training accuracy and validation accuracy
   - Training typically takes 5-10 minutes depending on your hardware
   - Final accuracy should be >99%

### Model Architecture Details

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])
```

### Customization Options

1. **Change classes**: Edit the `labels` array in `train-real-data.py`
2. **Adjust samples per class**: Modify `max_samples_per_class` parameter
3. **Change model architecture**: Edit the model definition in the training script
4. **Training parameters**: Adjust epochs, batch size, learning rate

### Troubleshooting

- **Missing classes**: Some Quick Draw classes may not be available (heart, rocket, boat, phone, balloon)
- **Memory issues**: Reduce `max_samples_per_class` if you run out of RAM
- **Slow training**: The script uses CPU by default; GPU training requires TensorFlow-GPU

### Output Files
- `public/assets/model/doodleModel.json` - Model topology
- `public/assets/model/doodleModel.weights.bin` - Model weights
- `best_model.h5` - Best model checkpoint (temporary)

The trained model will be automatically loaded by the DoodlePredictor component in the web application.