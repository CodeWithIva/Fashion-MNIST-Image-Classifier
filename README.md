# **Image Classifier CNN (Fashion MNIST)**

This is an end-to-end Computer Vision project that uses a Convolutional Neural Network (CNN) to classify grayscale images of clothing items. The model was trained on the Fashion MNIST dataset, achieving an accuracy exceeding 90%.

**Project Goal:** To build, train, and validate a Convolutional Neural Network—the standard architecture for all image recognition tasks—that can accurately identify 10 distinct clothing categories from raw pixel data.

## **Results Summary**

| Metric | Value | Description |
| :---- | :---- | :---- |
| **Final Test Accuracy** | 91.27% | The model's ability to correctly classify unseen images. |
| **Model Architecture** | Convolutional Neural Network (CNN) | Utilizes Conv2D layers for efficient feature extraction from images. |
| **Key Challenge** | Distinguishing ambiguous items (e.g., Shirt vs. Pullover) | The classification report highlights areas where human-like ambiguity affects the model's certainty. |

### **Training Visualization**

The model demonstrated successful learning, with both training and validation accuracy curves converging, indicating stable training without significant overfitting.

*(The cnn\_training\_curves.png file shows the visual proof of this stable training performance.)*

## **Project Contents**

| File | Description |
| :---- | :---- |
| image\_classifier\_cnn.py | The main script contains the CNN architecture definition, data normalization, training, and plotting. |
| requirements.txt | Lists all necessary Python libraries for this specific virtual environment. |
| cnn\_training\_curves.png | A visual plot showing accuracy and loss over the training epochs. |
| fashion\_cnn\_model.keras | The trained model weights and architecture (saved after a successful run). |

## **How to Run the Project**

### **1\. Setup Environment**

Create a new virtual environment (highly recommended for complex libraries like TensorFlow):

python \-m venv .venv  
.\\.venv\\Scripts\\Activate.ps1

Install the dependencies:

pip install \-r requirements.txt

### **2\. Execute the Script**

Run the main file from your terminal:

python image\_classifier\_cnn.py

**Technologies Used:** Python, TensorFlow/Keras, NumPy, Matplotlib