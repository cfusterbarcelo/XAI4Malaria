# XAI4Malaria
## Introduction

This repository contains the core code for our research on explainable AI applied to single-cell malaria diagnosis. At its heart is a full reproduction of the Soft-Attention Parallel CNN (SPCNN) from Ahamed et al. (2025) — https://www.nature.com/articles/s41598-025-90851-1 — including network architecture and training on the NIH malaria image dataset. Since the original authors did not release their code, we implemented and validated SPCNN end-to-end (noting small performance gaps likely due to unavailable hyperparameter details).

Building on that foundation, we integrate five complementary XAI techniques (Grad-CAM, Grad-CAM++, SHAP-Gradient, SHAP-Deep, and LIME) and provide interactive demo notebooks to generate, visualize, and compare their explanations. Our goal is to go beyond heatmaps: by engaging domain experts, we’ll assess each method’s clarity, usefulness, and trustworthiness in real-world diagnostic workflows.


## 📁 Project Structure  
- `configs/` • hyperparams & YAML files  
- `data/` • loader scripts and data transformations  
- `models/` • SPCNN model and model factory 
- `notebooks/` • demo notebook with demo dataset
- `explainability/` • Grad-CAM, Grad-CAM++, SHAP, LIME wrappers  
- `training/` • training pipelines for SPCNN
- `scripts/` • all scripts for running SPCNN and XAI
- `utils/` • helpers  


## 🎯 What’s Inside

- **Dataset**  
  We use the NLM-Falciparum-Thin-Cell-Images dataset (27 558 cropped RGB images of Giemsa-stained thin blood smear red blood cells), provided by the Lister Hill National Center for Biomedical Communications (LHNCBC), U.S. National Library of Medicine (2019), with expert annotations from the Mahidol Oxford Tropical Medicine Research Unit. Data is available at the NLM malaria datasheet: https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html :contentReference[oaicite:3]{index=3}

- **Baseline Model**  
  A faithful reimplementation of the Soft Attention Parallel Convolutional Neural Network (SPCNN) from Fuster-Barceló et al., Scientific Reports 15:6484 (2025), including all architectural details and hyperparameters :contentReference[oaicite:4]{index=4}.


- **Explainability Methods**  
  Integrated wrappers for Grad-CAM, Grad-CAM++, SHAP (Deep and Gradient variants), and LIME to generate both visual heatmaps and quantitative feature attributions.

## Demo Notebook

Try out our interactive demo to see all five XAI methods in action:

1. **Open** `notebooks/demo.ipynb` in your browser (e.g. upload to [Google Colab](https://colab.research.google.com/) for zero-install execution).  
2. **Run** each cell to load the pretrained SPCNN, generate Grad-CAM, Grad-CAM++, SHAP-Gradient, SHAP-Deep, and LIME explanations, and compare them.  

To run locally:

- Clone this repo  
- Install dependencies with  
  ```bash
  conda env create -f environment.yaml
  conda activate xai4malaria-demo
  ```
- Launch Jupyter and oipen `notebooks/demo.ipynb`

## Ownership & Collaborators

This project is the result of a joint effort between:

- **Universidad Carlos III de Madrid**, Neuroscience & Biomedical Sciences Department  
  — Prof. Arrate Muñoz-Barrutia
  — Dr. Caterina Fuster-Barceló  

- **Universitat de les Illes Balears**, Department of Mathematics & Computer Science  
  — Dr. Cristina Suemay Manresa Yee  
  — Dr. Silvia Ramis Guarinos  

<p float="left">
  <img src="utils/logos/Logo_UC3M.png" alt="UC3M logo" width="200" />
  <img src="utils/logos/Logo_UIB_2014.png" alt="UIB logo" width="200" />
</p>

