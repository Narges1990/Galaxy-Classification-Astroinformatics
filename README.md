# Galaxy Morphology Classification with CNN and Explainable AI

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)]()
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNN-brightgreen)]()
[![Explainable AI](https://img.shields.io/badge/XAI-Grad--CAM%20%7C%20IG%20%7C%20LIME-purple)]()

## Overview

This repository contains a 2025 Astroinformatics course project focused on **binary galaxy morphology classification** using a convolutional neural network (CNN). The goal of the project was to classify galaxy images into two morphological categories and to complement the predictive model with **explainable AI (XAI)** methods to inspect which image regions contributed to the model's decisions.

The project was completed as part of the **Astroinformatics** course taught by **Prof. Massio Brescia**.

## Project Motivation

Galaxy morphology is an important observational property in astronomy, as it reflects structural differences and can provide insight into galaxy formation and evolution. Manual galaxy classification can be time-consuming and subjective, especially for large astronomical surveys. Deep learning provides a scalable way to support automated classification, while explainable AI helps evaluate whether the model is relying on meaningful visual structures rather than irrelevant image artifacts.

## Main Objectives

- Build a CNN-based classifier for galaxy morphology recognition.
- Preprocess galaxy images and labels for supervised learning.
- Address class imbalance using balanced class weights.
- Improve generalization through image augmentation.
- Evaluate model performance using classification metrics and a confusion matrix.
- Interpret CNN decisions using Grad-CAM, Integrated Gradients, and LIME.

## Dataset

The project uses a binary galaxy classification dataset stored locally as `GALAXIES.zip`.

Expected internal structure:

```text
GALAXIES/
├── 2class_list.csv
└── DATA/
    ├── <CATAID>_giH.png
    ├── <CATAID>_giH.png
    └── ...
```

The label file contains:

- `CATAID`: galaxy image identifier
- `TARGET`: binary class label

Images are loaded from the `DATA/` directory, resized to **128 × 128 pixels**, normalized to the `[0, 1]` range, and split into training and validation sets using stratified sampling.

> Note: the dataset is not included in this repository because of size and/or distribution constraints. To reproduce the notebook, place `GALAXIES.zip` in the working directory or upload it in Google Colab.

## Methodology

### 1. Data Preprocessing

The images were loaded using OpenCV, resized to a fixed input size, normalized, and paired with their corresponding binary labels. A stratified train-validation split was used to preserve the class distribution across subsets.

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

### 2. Data Augmentation

To improve model generalization, the training images were augmented using random transformations:

- Rotation
- Width and height shifts
- Shear transformation
- Zoom
- Horizontal flipping

### 3. CNN Architecture

The model is a custom CNN built with TensorFlow/Keras. It includes:

- Three convolutional blocks
- Batch normalization
- Max pooling
- Dropout regularization
- Dense classification layers
- Sigmoid output for binary classification

Simplified architecture:

```text
Input: 128 × 128 × 3 image
↓
Conv2D(32) + BatchNorm + MaxPooling + Dropout
↓
Conv2D(64) + BatchNorm + MaxPooling + Dropout
↓
Conv2D(128, name="last_conv") + BatchNorm + MaxPooling + Dropout
↓
Flatten
↓
Dense(512) + BatchNorm + Dropout
↓
Dense(1, sigmoid)
```

The model was trained with:

- Optimizer: Adam
- Loss function: binary cross-entropy
- Batch size: 32
- Epochs: 20
- Class balancing: `class_weight`

## Results

The final validation evaluation showed the following performance:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Elliptical | 0.90 | 0.67 | 0.77 | 579 |
| Other | 0.75 | 0.93 | 0.83 | 601 |
| **Accuracy** |  |  | **0.80** | 1180 |
| **Macro average** | 0.82 | 0.80 | 0.80 | 1180 |
| **Weighted average** | 0.82 | 0.80 | 0.80 | 1180 |

The model achieved approximately **80% validation accuracy** in the final reported evaluation. During training, validation accuracy reached higher values in some epochs, but the final reported classification report should be considered the main reproducible performance summary.

## Explainable AI Analysis

To make the CNN predictions more interpretable, three XAI approaches were implemented.

### Grad-CAM

Grad-CAM was used to highlight image regions that strongly influenced the CNN prediction. The method uses gradients from the final convolutional layer named `last_conv`.

### Integrated Gradients

Integrated Gradients was used to estimate pixel-level attribution by comparing the prediction pathway from a baseline image to the original galaxy image.

### LIME

LIME was applied to generate local explanations by perturbing image segments and estimating their contribution to the model prediction.

Together, these methods provide complementary interpretability perspectives:

- Grad-CAM: coarse spatial localization
- Integrated Gradients: pixel-level attribution
- LIME: superpixel-based local explanation

## Repository Structure

```text
galaxy-classification-final/
├── Galaxy-Classification-Astroinformatics.ipynb   # Main notebook with outputs
├── Galaxy-Classification-Final.ipynb              # Clean/final notebook version
├── Galaxy Classification-Astroinformatics2.pdf    # Project presentation/report
├── README.md                                      # Project documentation
└── README-galaxy.md                               # Previous README draft
```

Recommended cleaned structure for a professional GitHub repository:

```text
galaxy-morphology-classification-xai/
├── README.md
├── notebooks/
│   └── Galaxy-Classification-Astroinformatics.ipynb
├── reports/
│   └── Galaxy Classification-Astroinformatics2.pdf
├── figures/
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   ├── confusion_matrix.png
│   ├── gradcam_example.png
│   ├── integrated_gradients_example.png
│   └── lime_output.png
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repository-name>.git
cd <your-repository-name>
```

Install the required packages:

```bash
pip install -r requirements.txt
```

If no `requirements.txt` file is available yet, the main dependencies are:

```bash
pip install numpy pandas opencv-python matplotlib scikit-learn tensorflow lime scikit-image tqdm
```

## How to Run

### Option 1: Google Colab

1. Open the notebook in Google Colab.
2. Upload `GALAXIES.zip` when prompted.
3. Run all cells sequentially.
4. The notebook will train the CNN, evaluate performance, and generate XAI visualizations.

### Option 2: Local Environment

1. Place `GALAXIES.zip` in the project directory.
2. Update the dataset path in the notebook if needed.
3. Run the notebook using Jupyter:

```bash
jupyter notebook
```

## Key Skills Demonstrated

This project demonstrates practical experience in:

- Astroinformatics
- Astronomical image classification
- Deep learning with TensorFlow/Keras
- Computer vision preprocessing
- CNN model design
- Class imbalance handling
- Model evaluation
- Explainable AI for image-based models
- Grad-CAM, Integrated Gradients, and LIME

## Limitations and Future Work

The current model was developed as a coursework project and can be improved in several ways:

- Use transfer learning with pretrained CNN architectures such as ResNet, EfficientNet, or VGG.
- Add independent test-set evaluation beyond the validation split.
- Perform hyperparameter optimization.
- Improve reproducibility by adding a fixed environment file.
- Compare CNN performance with classical machine learning baselines.
- Extend the task to multi-class galaxy morphology classification.
- Perform deeper error analysis on misclassified galaxies.

## Citation / Acknowledgment

This project was completed in 2025 as part of the **Astroinformatics** course taught by **Prof. Massio Brescia**.

## Author

**Narges Davoudi**  
MSc Data Science  
Astroinformatics coursework project, 2025

