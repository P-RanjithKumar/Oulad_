# **OULAD Dataset Analysis and Random Forest Classifier with RAPIDS**

## **🚀 Project Overview**
This project utilizes the **Open University Learning Analytics Dataset (OULAD)** for training a **GPU-accelerated Random Forest Classifier**. The primary goal is to preprocess, merge, and analyze the dataset to predict student success (final_result) efficiently using **RAPIDS cuDF and cuML**.

### Key Objectives:
- Efficient preprocessing of the OULAD dataset using RAPIDS.  
- Hyperparameter tuning for GPU-accelerated Random Forest Classifier.  
- Leveraging RAPIDS libraries for faster data handling and model training.  

---

## **🧑‍💻 Features**
- **Dataset Preprocessing**:  
  - Merging and cleaning OULAD data.
  - Handling missing values and ensuring data quality.
- **GPU-Accelerated Machine Learning**:  
  - Random Forest implementation using RAPIDS cuML.  
  - Hyperparameter tuning using a custom grid search approach.
- **Performance Metrics**:  
  - Evaluation using accuracy scores for model optimization.  

---

## **🛠️ Technologies Used**
- **GPU Libraries**:  
  - RAPIDS cuDF (GPU DataFrame library)  
  - RAPIDS cuML (GPU-accelerated ML library)  
- **Machine Learning Frameworks**:  
  - cuML RandomForestClassifier  
  - sklearn for Stratified K-Fold and GridSearchCV  
- **Programming Language**: Python 3.8 or later  

---

## **📋 Requirements**

### Software and Libraries:
- Python 3.8 or later  
- RAPIDS libraries (cuDF, cuML)  
- sklearn  
- cupy  

### Hardware:
- NVIDIA GPU (with CUDA support)  
- Compatible drivers and RAPIDS installation  

---

## **🏗️ Installation**

1. **Install RAPIDS**:  
   Clone the RAPIDS utilities repository and run the installation script:
   !git clone https://github.com/rapidsai/rapidsai-csp-utils.git
   !python rapidsai-csp-utils/colab/pip-install.py
   
2 . **Download and Prepare the Dataset:**
  -Obtain the OULAD dataset from OULAD Dataset.
  -Preprocess and merge the dataset using the preprocessing script provided in the project.
  
