# Multimodal Interpretable Data-Driven Models for Early Prediction of Antimicrobial Multidrug Resistance Using Multivariate Time-Series
Authors: Óscar Escudero-Arnanz, Sergio Martínez-Agüero, Paula Martín-Palomeque, Antonio G. Marques, Inmaculada Mora-Jimenez, Joaquín Álvarez-Rodríguez, Cristina Soguero-Ruiz

## Abstract
Electronic health records (EHR) is an inherently multimodal register of the patient's health status characterized by static data and multivariate time series (MTS). While MTS are a valuable tool for clinical prediction, their fusion with other data modalities can possibly result in more thorough insights and more accurate results.  Deep neural networks (DNNs) have emerged as fundamental tools for identifying and defining underlying patterns in the healthcare domain.  However, fundamental improvements in interpretability are needed for DNN models to be widely used in the clinical setting.  In this study, we present an approach built on a collection of interpretable multimodal data-driven models that may anticipate and understand the emergence of antimicrobial multidrug resistance (MDR) germs in the intensive care unit (ICU) of the University Hospital of Fuenlabrada (Madrid, Spain). The profile and initial health status of the patient are modeled using static variables, while the evolution of the patient’s health status during the ICU stay is modeled using several MTS, including mechanical ventilation and antibiotics intake. The multimodal DNNs models proposed in this paper include interpretable principles in addition to being effective at predicting MDR and providing an explainable prediction support system for MDR in the ICU. Furthermore, our proposed methodology based on multimodal models and interpretability schemes can be leveraged in additional clinical problems dealing with EHR data, broadening the impact and applicability of our results.

## Repository Overview
This repository includes all necessary code for building, training, and evaluating interpretable DNN models for MDR prediction. The repository is organized into three main directories:

- ORIGINAL_DATA: Scripts and data for generating data splits and applying Feature Selection (FS) and Permutation Feature Importance (PFI))
- Libraries: Required dependencies for running the experiments.
- EXPERIMENTS: Contains model scripts, experimental configurations, and results for different setups, including those using interpretability mechanisms, FS, PFI, and without dimensionality reduction.

### Detailed Directory Breakdown
ORIGINAL_DATA
  - 0_CLASSICAL_FS: Scripts for traditional FS.
  - 0_PFI: Files for PFI-based feature selection.
  - 0_Results_FS_PFI: FS/PFI results and visual heatmap generation.
  - 1_Generate_splits: Scripts to create training, validation, and test splits.
  - splits_14_days: Folders with preprocessed data splits.
    
EXPERIMENTS
  - 0_WITHOUT_FS: Models trained without FS
  - 1_INTERPRETABILITY: Experiments with interpretability methods (NLHA, HAM, and Dynamask).
  - 2_CLASSICAL_FS: Models trained using classical FS
  - 2_PFI: Models trained using PFI

Libraries
Different .py files containing libraries and dependencies for running the experiments
