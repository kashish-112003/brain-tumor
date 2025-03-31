# Abstract
The timely and precise diagnosis of brain tumors is essential for effective treatment planning, as they present a major healthcare concern. This Deep Learning project presents a deep learning system using Convolutional Neural Networks (CNNs) for brain tumor classification. Trained on MRI scans across four classes (glioma, meningioma, pituitary tumor, and no tumor), our model achieved high accuracy (over 95%), demonstrating its potential to support medical decision-making. Further research will involve dataset expansion, exploring the integration of multiple imaging modalities, and addressing ethical considerations for safe clinical adoption.


# 🧠 Brain Tumor Classification Using CNN
This project utilizes Convolutional Neural Networks (CNNs) for brain tumor detection using MRI images. The model classifies images into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor. The goal is to assist in early diagnosis and improve medical decision-making using AI-powered deep learning techniques


# Introduction
Brain tumors are life-threatening conditions requiring early and accurate diagnosis. Manual MRI analysis is time-consuming and prone to errors. This project automates tumor classification using deep learning, reducing the diagnostic burden on radiologists.

## ❌ Problem Statement  
Traditional methods for tumor diagnosis have limitations:  
- **Subjectivity**: Human errors in diagnosis  
- **Time Constraints**: Manual MRI analysis takes too long  
- **Inconsistencies**: Variability in radiologists' assessments  

This project tackles these challenges by implementing **CNN-based image classification**. 

##  Importance of This Project  
✅ **Early Detection**: Faster and more accurate **tumor diagnosis**  
✅ **Reduced Human Effort**: AI-powered automation for **quick decision-making**  
✅ **Improved Accuracy**: CNN-based classification ensures **high reliability**  

---


 # Methodology
 This section outlines the key technical steps involved in developing and implementing brain tumor detection and classification system.
Step1: Dataset Preparation

# Source: The dataset was download from Kaggle. It contains 2 folders 'Testing' and 'Training' and each folder contains 4 more folders for different categories of brain tumor each variant contains 115 - 74 images in them.
Dataset link:	https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor- mri-dataset

# plotting count
![Screenshot 2025-03-31 195551](https://github.com/user-attachments/assets/a2344f39-018c-4dc5-8e00-a06cd3431e02)

# Training loss Vs Epoch
![Screenshot 2025-03-31 230011](https://github.com/user-attachments/assets/5b88f76c-aa1e-4b10-9e54-3493924b7c93)

# Deployement with Prediction.
Designed a User Interface where user can provide their Brain MRI image and Submit it. The trained model will predict the Class of Brain Tumor present.
The system was design using Streamlit, Tensorflow and OpenCV. 
![Screenshot 2025-03-31 200816](https://github.com/user-attachments/assets/e19a9da0-528a-47df-8fc8-5e05194d3965)

# Conclusion and Future Scope
This project successfully developed a brain tumor classification system leveraging deep learning techniques. The Xception model architecture demonstrated promising results on the dataset, achieving an Accuracy of 95.10%, highest Precision of 97%, ighest Recall of 98% and highest F1-Score of 97%.

# Future Scope
-->Dataset Expansion: Collaborate with medical institutions to collect a larger and more diverse dataset of brain tumor images, with a focus on increasing the representation of less prevalent tumor types. This will boost the model's ability to handle real- world variations.

-->Interpretability: Implement Grad-CAM visualizations to highlight the regions within the MRI scans that contribute most to the model's classification decisions. This will help in understanding the model's reasoning and build trust with clinicians.





