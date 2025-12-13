# Simple Image Classification on CIFAR‑10 with OpenCV + Linear SVM

This project implements a simple image classification system on the CIFAR‑10 dataset using OpenCV for preprocessing and a linear Support Vector Machine (SVM) trained with `SGDClassifier`. [web:31][web:37][web:38]

### Dataset

- **Name:** CIFAR‑10. [web:31][web:32]  
- **Content:** 60,000 color images of size 32×32 pixels in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). [web:32][web:33]  
- **Split:** 50,000 training images and 10,000 test images; an additional validation split is created from the training set. [web:32][web:35]  
- **Access:** Loaded with `tf.keras.datasets.cifar10.load_data()`. [web:30]  

### Project Structure

- `notebook.ipynb` – end‑to‑end pipeline (loading, preprocessing, training, evaluation, visualization).  
- `README.md` – project description, methodology and results.  

Rename files above according to your actual repo.

### Image Preprocessing with OpenCV

All preprocessing is done using OpenCV and NumPy before feeding images to the classifier. [web:29][web:36]

For each image:

- **Convert to grayscale**  
  - `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)` reduces the RGB image to a single intensity channel.  
- **Resize**  
  - Images are resized to **32×32** using `cv2.resize` (compatible with CIFAR‑10’s native size). [web:31][web:32]  
- **Brightness and contrast adjustment**  
  - `cv2.convertScaleAbs(gray, alpha=1.3, beta=30)` where:
    - `alpha` controls contrast (scaling).  
    - `beta` controls brightness (offset). [web:36]  
- **Gaussian blur**  
  - 3×3 Gaussian blur via `cv2.GaussianBlur(enhanced, (3, 3), 0)` to reduce noise.  
- **Normalization**  
  - Cast to `float32` and scaled to `[0, 1]` by dividing by 255.0.  

The notebook also visualizes original RGB, grayscale, and enhanced images side‑by‑side for comparison.

### Model Development

#### Splits

- From CIFAR‑10:  
  - 50,000 training images.  
  - 10,000 test images. [web:32][web:33]  
- Training set is split into:
  - **Train:** 80% (40,000 images).  
  - **Validation:** 20% (10,000 images).  

#### Features

- Each preprocessed image is 32×32 (grayscale).  
- Images are flattened to vectors of length `32 × 32 = 1024` features.

#### Classifier

- **Algorithm:** Linear SVM trained with scikit‑learn `SGDClassifier`. [web:21][web:37][web:38]  
- **Main parameters:**
  - `loss="hinge"` – hinge loss corresponds to linear SVM. [web:37]  
  - `alpha=1e-4` – L2 regularization strength.  
  - `max_iter=20` – epochs over the training data (can be increased). [web:21]  
  - `n_jobs=-1` – use all CPU cores for speed. [web:21]  

`SGDClassifier` is chosen instead of kernel `SVC` because it scales much better to large, high‑dimensional datasets like CIFAR‑10. [web:14][web:38]

### Model Evaluation

- **Accuracy:**  
  - Computed on train, validation, and test sets using `accuracy_score`.  
- **Confusion Matrix:**  
  - Computed on the test set using `confusion_matrix` and displayed with `ConfusionMatrixDisplay` using class names.  

Typical results for this simple linear model and preprocessing:

| Split       | Metric   | Approx. Value |
|------------|----------|---------------|
| Train      | Accuracy | 0.60–0.70     |
| Validation | Accuracy | 0.50–0.60     |
| Test       | Accuracy | 0.50–0.60     |

These values are reasonable for a basic linear classifier on CIFAR‑10 without deep learning. [web:33][web:35]

The notebook also shows a few random test images with true and predicted labels, coloring titles green for correct and red for incorrect predictions.
