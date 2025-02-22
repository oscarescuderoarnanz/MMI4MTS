# Multimodal Interpretable Data-Driven Models for Early Prediction of Antimicrobial Multidrug Resistance Using Multivariate Time-Series

Authors: Óscar Escudero-Arnanz, Sergio Martínez-Agüero, Paula Martín-Palomeque, Antonio G. Marques, Inmaculada Mora-Jimenez, Joaquín Álvarez-Rodríguez, Cristina Soguero-Ruiz

## Abstract
Electronic Health Records (EHRs) serve as a comprehensive repository of multimodal patient health data, combining static demographic attributes with dynamic, irregular Multivariate Time Series (MTS), characterized by varying lengths. While MTS provide critical insights for clinical predictions, their integration with static features enables a more nuanced understanding of patient health trajectories and enhances predictive accuracy. Deep Neural Networks (DNNs) have proven highly effective in capturing complex patterns in healthcare data, offering a framework for multimodal data fusion. However, their adoption in clinical practice is limited by a lack of interpretability, as transparency and explainability are essential for supporting informed medical decisions. This study presents interpretable multimodal DNN architectures for predicting and understanding the emergence of Multidrug Resistance (MDR) in Intensive Care Units (ICUs). The proposed models integrate static demographic data with temporal variables, providing a holistic view of baseline patient characteristics and health progression. To address predictive performance and interpretability challenges, we introduce a novel methodology combining feature selection techniques with attention mechanisms and post-hoc explainability tools. This approach not only reduces feature redundancy but also highlights key risk factors, thereby improving model accuracy and robustness. Experimental results demonstrate the effectiveness of the proposed framework, achieving a Receiver Operating Characteristic Area Under the Curve of 76.90 $\pm$ 3.10, a significant improvement over baseline models. Beyond MDR prediction, this methodology offers a scalable and interpretable framework for addressing various clinical challenges involving EHR data. By integrating predictive accuracy with explanatory insights—such as the identification of key risk factors—this work supports timely, evidence-based interventions to improve patient outcomes in ICU settings.

## Repository Overview

#### **`classification_architectures/`**
This folder contains implementations of multimodal and non-multimodal classification models:

- Multilayer Perceptron (MLP)
- Gated Recurrent Units (GRU)
- Transformer
- Graph Convolutional Neural Network (GCNN)
- Joint Heterogeneous Fusioner (JHF)
- First Hidden State Initializer (FHSI)
- Late Fusion Convex Optimization (LFCO)
- Late Fusion Logistic Regression (LFLR)
- Multimodal GCNN (MultGCNN)

#### **`experiments/`**
This folder contains all experiment setups and results for three main prediction tasks:
- **MDR/**: Experiments on private ICU data for MDR prediction
- **CIRCULATORY/**: Experiments using public HiRID data for circulatory failure prediction
- **SEPSIS/**: Experiments using public MIMIC-IV data for sepsis prediction

Each subfolder contains:
  - **considering_all_features/**: Models trained with the full feature set
  - **classical_fs/**: Models trained using features selected by classical FS methods
  - **PFI/**: Models trained using features selected by PFI-based methods

#### **`libraries/`**
Utility functions and dependencies required for running experiments, including: Loss function, performance metric calculations, and other helper functions for model training and evaluation

#### **`knowledge_extraction/`**
Feature selection and interpretability analyses for each task:
- Subfolders: `MDR/`, `CIRCULATORY/`, and `SEPSIS/`
- Includes:
  - Features selected by classical FS
  - Features selected using PFI
  - Features extracted through interpretability methods (NLHA, HAM, and Dynamask)

>  **Note**:  
> - **MDR Data**: The MDR dataset is private and cannot be shared due to confidentiality agreements 
> - **Circulatory Failure (HiRID)**: The HiRID dataset is publicly available but requires users to satisfy certain requirements. For more details and to request access, visit the [HiRID Dataset Website](https://physionet.org/content/hirid/1.1.1/)
> - **Sepsis (MIMIC-IV)**: The MIMIC-IV dataset is publicly available but requires users to satisfy certain requirements. For more details and access, visit the [MIMIC-IV Dataset Page](https://physionet.org/content/temporal-respiratory-support/1.0.0/)

### **`requirements.txt`**
File containing the dependencies required to run this project. It is recommended to install these dependencies in a virtual environment to avoid conflicts.
---
