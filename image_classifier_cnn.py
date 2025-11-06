# -------------------------------------------------------------
# PROJECT 2: FASHION MNIST IMAGE CLASSIFIER (CONVOLUTIONAL NEURAL NETWORK)
# -------------------------------------------------------------
# This script builds and trains a Convolutional Neural Network (CNN) 
# to classify 10 types of clothing items from the Fashion MNIST dataset.
# CNNs are the standard architecture for all image recognition tasks.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Global variables for image dimensions and classes
IMG_WIDTH = 28
IMG_HEIGHT = 28
NUM_CHANNELS = 1 # Grayscale images (1 color channel)
NUM_CLASSES = 10 

# Mapping of class indices to actual human-readable labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# -------------------------------------------------------------
# STEP 1: Load and Preprocess Data
# -------------------------------------------------------------

def load_and_prepare_data():
    """Loads Fashion MNIST data and preprocesses it for CNN training."""
    
    print("--- STEP 1: Loading and Preparing Data ---")
    
    # Load the Fashion MNIST dataset directly from Keras
    (X_train_raw, y_train), (X_test_raw, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    print(f"Original Training Data Shape: {X_train_raw.shape}")
    
    # CNNs expect the data to be in the shape (samples, height, width, channels).
    # Currently, it is (samples, 28, 28). We need to add the channel dimension (1 for grayscale).
    
    # Reshape: (60000, 28, 28) -> (60000, 28, 28, 1)
    X_train_reshaped = X_train_raw.reshape(-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)
    X_test_reshaped = X_test_raw.reshape(-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)
    
    # Normalize pixel values to be between 0 and 1 (from 0-255)
    # This is critical for faster learning, similar to MinMaxScaler.
    X_train_normalized = X_train_reshaped.astype("float32") / 255.0
    X_test_normalized = X_test_reshaped.astype("float32") / 255.0
    
    print(f"Normalized Training Data Shape (CNN-Ready): {X_train_normalized.shape}")
    print("Data loading and normalization complete.")
    
    return X_train_normalized, X_test_normalized, y_train, y_test

# -------------------------------------------------------------
# STEP 2: Build the Convolutional Neural Network (CNN)
# -------------------------------------------------------------

def build_cnn_model(input_shape):
    """Defines the CNN architecture for image classification."""
    
    print("\n--- STEP 2: Building CNN Architecture ---")
    
    model = Sequential([
        # 1. CONVOLUTIONAL LAYER: The 'Feature Extractor'
        # It scans the image with 32 small 3x3 filters (or kernels) to detect simple features like edges.
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, name='Conv_1'),
        
        # 2. MAX POOLING: The 'Downsampler'
        # It reduces the size of the feature map (image) by 50% (2x2) to speed up computation and reduce noise.
        MaxPooling2D(pool_size=(2, 2), name='MaxPool_1'),
        
        # 3. SECOND CONVOLUTIONAL BLOCK (For more complex features)
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='Conv_2'),
        MaxPooling2D(pool_size=(2, 2), name='MaxPool_2'),
        
        # 4. FLATTEN: The 'Transition Layer'
        # Converts the 2D feature map into a 1D vector so it can be fed into the standard Dense (Sequential) layers.
        Flatten(name='Flatten'),
        
        # 5. DENSE HIDDEN LAYER: The 'Decision Maker'
        # Processes the extracted features using 128 neurons.
        Dense(units=128, activation='relu', name='Dense_Hidden'),
        
        # 6. DENSE OUTPUT LAYER
        # Outputs 10 probabilities (one for each class). Softmax ensures probabilities sum to 1.
        Dense(units=NUM_CLASSES, activation='softmax', name='Output')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', # Different loss for non-one-hot-encoded targets
                  metrics=['accuracy'])
    
    model.summary()
    return model

# -------------------------------------------------------------
# STEP 3: Train and Evaluate the Model
# -------------------------------------------------------------

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """Trains the CNN and evaluates its performance."""
    
    print("\n--- STEP 3: Training the Model (10 Epochs) ---")
    
    # Training the model
    # We use fewer epochs here because CNNs learn much faster
    history = model.fit(X_train, y_train, 
                        epochs=10, 
                        batch_size=32, 
                        validation_data=(X_test, y_test),
                        verbose=1)

    # Evaluate the final performance on the test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n--- Evaluation Results ---")
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")
    
    # Make predictions for the classification report
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Print a detailed classification report
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return history, y_pred

# -------------------------------------------------------------
# STEP 4: Visualization of Training Curves
# -------------------------------------------------------------

def plot_training_history(history):
    """Plots the model's accuracy and loss over the epochs."""
    
    print("\n--- STEP 4: Generating Visualization Plot ---")
    
    plt.figure(figsize=(12, 5))

    # Subplot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('CNN Model Accuracy Over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Loss (Error)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('CNN Model Loss Over Epochs')
    plt.ylabel('Loss (Error)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle("Fashion MNIST CNN Training History", fontsize=16)
    plt.savefig('cnn_training_curves.png')
    plt.show()

# -------------------------------------------------------------
# MAIN EXECUTION BLOCK
# -------------------------------------------------------------

if __name__ == "__main__":
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Define input shape for the CNN
    input_shape = (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)

    # Build the model
    model = build_cnn_model(input_shape)

    # Train and evaluate
    history, y_pred = train_and_evaluate(model, X_train, X_test, y_train, y_test)

    # Visualization
    plot_training_history(history)

    print("\nProject 2 Complete: CNN Model Trained and Evaluated.")