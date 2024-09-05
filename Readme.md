🌊 Water Segmentation using Satellite Images 🛰️

Overview

This project focuses on water segmentation using satellite images. The goal is to identify and segment water bodies in satellite imagery, which is crucial for environmental monitoring and resource management. We use images with a resolution of 128x128 pixels and 12 band channels.


Dataset

Image Size: 128x128 pixels 📏
Channels: 12 band channels 🌈
Preprocessing
Normalization: 🧹 Each band channel is normalized to ensure that the data is on a consistent scale.

Preprocessing Steps:

Resizing: Ensuring all images are 128x128 pixels 🔄
Normalization: Applied to each band channel separately 🌟
Augmentation: Applied to enhance the robustness of the model (e.g., rotation, flipping) 🔄
Model Architecture
We utilize the U-Net architecture, which is well-suited for segmentation tasks due to its encoder-decoder structure and skip connections that retain spatial information. 🏗️

Evaluation Metrics

We use the following metrics to evaluate model performance:

Intersection over Union (IoU): Measures the overlap between predicted and ground truth masks 📊
F1 Score: Harmonic mean of precision and recall 🔍
Precision: Fraction of relevant instances retrieved by the model 📈
Recall: Fraction of relevant instances that were retrieved 📉
AUC Curve: Area under the receiver operating characteristic curve 📉
Confusion Matrix: Visualizes the performance of the classification algorithm 🧩
Learning Curve: Shows the model’s performance over epochs 📈
Results
The U-Net model achieved the following
 
accuracy scores:

Accuracy: 91% 🎯
Accuracy: 92% 🎯
Accuracy: 93% 🎯