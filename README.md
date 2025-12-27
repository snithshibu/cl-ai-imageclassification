# CIFAR-10 Image Classification with OpenCV and CNN

## Project Overview
Short description of task and CIFAR-10 dataset.

## Dataset
- CIFAR-10: 60k 32x32 color images, 10 classes. 

## Image Processing with OpenCV
- Grayscale conversion, resize, brightness/contrast, Gaussian blur.
- Side-by-side plots of original vs preprocessed images.

## Model 1: SVM with OpenCV Features
- Input: flattened 32x32 grayscale images (1024 features).
- Accuracy: ~0.39 on validation; confusion matrix + example predictions.

## Model 2: CNN (Keras)
- Architecture: Conv2D + MaxPooling + Dense + Dropout.
- Training: 25 epochs, batch size 64.
- Test accuracy: ~0.78 (or your final value), plus example predictions.

## How to Run
- `python3 -m venv venv`  
- `source venv/bin/activate`  
- `pip install -r requirements.txt`  
- Open notebook in VS Code and run cells.
