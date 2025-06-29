# 🍐 Pear Leaf Disease Detection & Classification

This project implements a **hybrid deep learning system** for classifying visually similar pear leaf diseases using a combination of **Siamese Convolutional Neural Networks (SCNN)**, **Variational Autoencoders (VAE)**, and a **Support Vector Machine (SVM)** classifier. The system is designed for low-data, real-world agricultural settings and integrates contrastive learning with interpretable latent representations.

---

##  Project Highlights

-  **Siamese CNN** to learn visual similarity through positive/negative image pairs
-  **Variational Autoencoder (VAE)** for learning generative latent features
-  **Joint Embedding (enhanced_z)**: Concatenation of latent `z` and SCNN features
-  **SVM Classifier** trained on `enhanced_z` for disease classification
-  **Data Augmentation** to prevent overfitting (flips, rotation, noise)
-  **PCA and Heatmap Visualizations** for interpretability and latent space analysis

---

## 📁 Dataset Structure

Organize your dataset on Google Drive like this:

/MyDrive/pear/
├── DiseaseA/
│ ├── image1.jpg
│ └── image2.jpg
├── DiseaseB/
│ ├── image1.jpg
│ └── image2.jpg


Each folder must represent a unique disease class.

---

## 🚀 How to Run on Google Colab

> 📌 Make sure your dataset is uploaded to `/MyDrive/pear`

### ✅ Steps:

1. Open the `.ipynb` or script file in Google Colab  
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
3. DATASET_PATH = '/content/drive/MyDrive/pear'
4. Run all cells to:

Load and preprocess images

Build and train SCNN-VAE model

Generate latent features and train SVM

Evaluate performance and visualize results

## MODEL ARCHITECTURE

| Component             | Description                                                |
| --------------------- | ---------------------------------------------------------- |
|  Siamese CNN        | Extracts visual encodings from input pairs                 |
|  VAE                | Learns latent distribution using encoder-decoder framework |
|  Enhanced Embedding | Combines SCNN features with latent `z`                     |
|  SVM                | Trained on enhanced embeddings for final classification    |


---

## 📈 Results

- **Final Accuracy**: **76.31%**  
  Achieved using a hybrid SCNN feature extractor + VAE latent embeddings and SVM classifier on limited, imbalanced pear leaf disease dataset.

- **Precision / Recall / F1-score**:
  
  | Class     | Precision | Recall | F1-score |
  |-----------|-----------|--------|----------|
  | Curl      | 0.88      | 0.93   | 0.91     |
  | Healthy   | 0.74      | 0.73   | 0.73     |
  | Spot      | 0.59      | 0.40   | 0.48     |
  | Slug      | 0.78      | 0.88   | 0.83     |
  | **Overall** | **0.76** | **0.76** | **0.76** |

---

## 🖼️ Sample Visualizations

### 🔷 PCA Projection (SCNN Features)

![PCA Projection]

> Shows clustering of leaf disease classes in reduced 2D space. “Slug” class appears better separated; “Spot” and “Healthy” overlap slightly.

---

### 🔍 Sample Predictions

![Sample Predictions]

> True vs. Predicted labels from test set using SVM. Model correctly classifies many difficult cases; some confusion between "spot" and "slug" remains.

---

### 📊 Confusion Matrix

![Confusion Matrix]

> Majority of misclassifications occur between **spot ↔ slug**, a common challenge due to visual similarity. “Curl” class shows highest precision and recall.

---

