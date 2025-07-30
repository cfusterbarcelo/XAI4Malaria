# XAI4Malaria
## Introduction

Accurate and rapid diagnosis of malaria remains a cornerstone of global health efforts, yet manual examination of blood smears is time-consuming and prone to variability. **XAI4Malaria** bridges this gap by combining a robust convolutional neural network (CNN) for parasitized red blood cell detection with state-of-the-art explainability methods—such as Grad-CAM, SHAP, and LIME—to not only automate malaria screening, but also to make every prediction transparent and interpretable.

Our focus goes beyond raw performance: we conduct **user-centered evaluations** with biologists, clinicians, and laboratory technicians to understand which visual and quantitative explanations truly resonate with domain experts. By gathering direct feedback on clarity, trustworthiness, and usefulness, XAI4Malaria delivers practical guidance on selecting and deploying explainability techniques in real-world biomedical settings.

Whether you’re an AI researcher seeking best practices for model interpretability, or a healthcare professional curious about integrating explainable AI into diagnostic workflows, XAI4Malaria offers a hands-on framework, interactive demos, and survey-backed insights to empower stakeholders at every step of the pipeline.


## 📁 Project Structure  
- `configs/` • hyperparams & YAML files  
- `data/` • sample images & loader scripts  
- `models/` • CNN & SPCNN implementations  
- `explainability/` • Grad-CAM, SHAP, LIME wrappers  
- `training/` • training pipelines  
- `scripts/` • one-off utilities  
- `utils/` • helpers  


## 🎯 What’s Inside

- **Dataset**  
  We use the NLM-Falciparum-Thin-Cell-Images dataset (27 558 cropped RGB images of Giemsa-stained thin blood smear red blood cells), provided by the Lister Hill National Center for Biomedical Communications (LHNCBC), U.S. National Library of Medicine (2019), with expert annotations from the Mahidol Oxford Tropical Medicine Research Unit. Data is available at the NLM malaria datasheet: https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html :contentReference[oaicite:3]{index=3}

- **Baseline Model**  
  A faithful reimplementation of the Soft Attention Parallel Convolutional Neural Network (SPCNN) from Fuster-Barceló et al., Scientific Reports 15:6484 (2025), including all architectural details and hyperparameters :contentReference[oaicite:4]{index=4}.


- **Explainability Methods**  
  Integrated wrappers for Grad-CAM, Grad-CAM++, SHAP (Deep and Gradient variants), and LIME to generate both visual heatmaps and quantitative feature attributions.


## 🤝 Contributing
1. Fork & branch
2. Write tests
3. Open a PR

