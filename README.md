# CIFAR-10 Image Classification with OpenCV, SVM and CNN

## 1. Project Overview
This project builds a simple image classification system on the CIFAR-10 dataset using two approaches:

- Classical ML baseline with OpenCV preprocessing + SVM.
- Deep learning model using a small CNN in Keras/TensorFlow.

The goal is to practice basic image processing (grayscale, resizing, brightness/contrast, filtering) and compare a traditional ML model with a CNN.

## 2. Dataset

- **Dataset**: CIFAR-10 (from `tensorflow.keras.datasets.cifar10`).  
- **Size**: 60,000 color images of size 32×32.  
- **Classes (10)**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. 

## 3. Image Processing with OpenCV

For the classical ML baseline a subset of 10,000 training images is preprocessed using OpenCV:

1. Convert RGB → BGR and then to **grayscale**. 
2. **Resize** to 32×32 (kept same size, but via OpenCV).  
3. Adjust **brightness and contrast** using `cv2.convertScaleAbs` (alpha = 1.3, beta = 30). 
4. Apply a **Gaussian blur** filter (`cv2.GaussianBlur` with 3×3 kernel). 
5. Normalize pixel values to \([0, 1]\).

The notebook visualizes original vs preprocessed images side‑by‑side to show the effect of these operations.

## 4. Model 1 – SVM with OpenCV Features

- Input features: preprocessed grayscale 32×32 images flattened to 1,024‑dimensional vectors.  
- Train/validation split: 80% / 20% on the 10,000‑image subset.  
- Classifier: `sklearn.svm.SVC` with RBF kernel (`C=2.0`, `gamma="scale"`). 

**Results**

- Validation accuracy: ~**0.39** (39%).  
- Confusion matrix shows many confusions between visually similar classes such as cats/dogs and cars/trucks.  
- Sample validation images with predicted vs true labels are plotted in the notebook.

This serves as a baseline and demonstrates that simple grayscale + SVM is not enough for CIFAR‑10.

## 5. Model 2 – CNN (Keras)

For the CNN, the **full** CIFAR‑10 train/test split is used:

- Images are normalized to \([0, 1]\) and kept as 32×32×3 color tensors. 
- Labels are one‑hot encoded with `to_categorical`. 

**Architecture**

Implemented using `tensorflow.keras`:

- Conv2D(32, 3×3, ReLU, same)  
- Conv2D(32, 3×3, ReLU, same)  
- MaxPooling2D(2×2) + Dropout(0.25)  
- Conv2D(64, 3×3, ReLU, same)  
- Conv2D(64, 3×3, ReLU, same)  
- MaxPooling2D(2×2) + Dropout(0.25)  
- Flatten  
- Dense(256, ReLU) + Dropout(0.5)  
- Dense(10, softmax)

- Optimizer: Adam (learning rate = 1e‑3).  
- Loss: categorical cross‑entropy.  
- Metrics: accuracy.

**Training setup**

- Epochs: **25**  
- Batch size: 64  
- Validation split: 20% of the training set. 

**Results**

- Best validation accuracy: around **0.77**.  
- Final test accuracy: about **0.785** on the CIFAR‑10 test set.  
- The notebook plots random test images with **Pred / True** labels, showing many correct predictions for multiple classes.

Compared to the SVM baseline, the CNN significantly improves performance, which matches typical behavior on CIFAR‑10.
