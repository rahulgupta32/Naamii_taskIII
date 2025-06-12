# NAAMII Task III - Knee CT Feature Extraction Pipeline

This repository contains the complete pipeline for **feature extraction and comparison from a 3D knee CT scan**, as specified in NAAMII's Task III assignment.

## 📋 Overview

**Task Goal:**  
- Split a segmented 3D knee CT scan into Tibia, Femur, and Background regions.
- Use an inflated (2D→3D) pretrained DenseNet121 model to extract feature vectors from each region.
- Compute cosine similarity between regions at three specified convolutional layers.
- Output all similarity scores for each image in a single CSV.

## 🏗️ Pipeline Steps

1. **Segmentation-Based Splitting:**  
   The input segmentation mask is separated into:
   - Tibia region (label = 1)
   - Femur region (label = 2)
   - Background (label = 0)

2. **2D→3D DenseNet121 Model Inflation:**  
   - All Conv2d, BatchNorm2d, MaxPool2d, and AvgPool2d layers are replaced with their 3D counterparts.
   - Pretrained weights are inflated as per assignment instructions.

3. **Feature Extraction:**  
   - Each region is passed through the 3D DenseNet121 (up to the `features` block).
   - Feature maps are extracted from:
     - Last convolutional layer
     - Third-last convolutional layer
     - Fifth-last convolutional layer
   - Global average pooling is applied to produce a fixed-length feature vector for each.

4. **Feature Comparison:**  
   - Cosine similarity is computed for every pair of regions (Tibia vs Femur, Tibia vs Background, Femur vs Background), at each layer.

5. **Results Organization:**  
   - Results are saved in a single CSV (`knee_region_cosine_similarity.csv`) with one row per image and columns for each region-pair/layer combination.


## 🚀 How to Run

1. **Dependencies:**  
   - Python 3.x  
   - torch (PyTorch)  
   - torchvision  
   - nibabel  
   - numpy  
   - pandas  

   Install with:
   ```bash
   pip install torch torchvision nibabel numpy pandas
