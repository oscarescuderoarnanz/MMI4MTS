# Multimodal Interpretable Data-Driven Models for Early Prediction of Antimicrobial Multidrug Resistance Using Multivariate Time-Series
Authors: Óscar Escudero-Arnanz, Sergio Martínez-Agüero, Paula Martín-Palomeque, Antonio G. Marques, Inmaculada Mora-Jimenez, Joaquín Álvarez-Rodríguez, Cristina Soguero-Ruiz

## Abstract
Electronic Health Records (EHRs) serve as a comprehensive repository of multimodal patient health data, combining static demographic attributes with dynamic, irregular Multivariate Time Series (MTS), characterized by varying lengths. While MTS provide critical insights for clinical predictions, their integration with static features enables a more nuanced understanding of patient health trajectories and enhances predictive accuracy. Deep Neural Networks (DNNs) have proven highly effective in capturing complex patterns in healthcare data, offering a framework for multimodal data fusion. However, their adoption in clinical practice is limited by a lack of interpretability, as transparency and explainability are essential for supporting informed medical decisions. This study presents interpretable multimodal DNN architectures for predicting and understanding the emergence of Multidrug Resistance (MDR) in Intensive Care Units (ICUs). The proposed models integrate static demographic data with temporal variables, providing a holistic view of baseline patient characteristics and health progression. To address predictive performance and interpretability challenges, we introduce a novel methodology combining feature selection techniques with attention mechanisms and post-hoc explainability tools. This approach not only reduces feature redundancy but also highlights key risk factors, thereby improving model accuracy and robustness. Experimental results demonstrate the effectiveness of the proposed framework, achieving a Receiver Operating Characteristic Area Under the Curve of 76.90 $\pm$ 3.10, a significant improvement over baseline models. Beyond MDR prediction, this methodology offers a scalable and interpretable framework for addressing various clinical challenges involving EHR data. By integrating predictive accuracy with explanatory insights such as the identification of key risk factors—this work supports timely, evidence-based interventions to improve patient outcomes in ICU settings.

## Repository Overview
This repository includes all necessary code for building, training, and evaluating interpretable DNN models for MDR prediction. The repository is organized into three main directories:

- ORIGINAL_DATA: Scripts and data for generating data splits and applying Feature Selection (FS) and Permutation Feature Importance (PFI))
- Libraries: Required dependencies for running the experiments.
- EXPERIMENTS: Contains model scripts, experimental configurations, and results for different setups, including those using interpretability mechanisms, FS, PFI, and without dimensionality reduction.

### Detailed Directory Breakdown
ORIGINAL_DATA:
  - 0_CLASSICAL_FS: Scripts for traditional FS.
  - 0_PFI: Files for PFI-based feature selection.
  - 0_Results_FS_PFI: FS/PFI results and visual heatmap generation.
  - 1_Generate_splits: Scripts to create training, validation, and test splits.
  - splits_14_days: Folders with preprocessed data splits. These files cannot be uploaded due to privacy restrictions.
  - Note: This directory includes three private .csv files containing static and temporal patient data from the ICU of the University Hospital of Fuenlabrada. These files cannot be uploaded due to privacy restrictions. 

    
EXPERIMENTS:
  - 0_WITHOUT_FS: Models trained without FS
  - 1_INTERPRETABILITY: Experiments with interpretability methods (NLHA, HAM, and Dynamask).
  - 2_CLASSICAL_FS: Models trained using classical FS
  - 2_PFI: Models trained using PFI

Libraries:
Different .py files containing libraries and dependencies for running the experiments
