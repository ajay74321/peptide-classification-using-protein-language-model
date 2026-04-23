# 🧬 Peptide Classification using Protein Language Models

<p align="center">
  <b>High-performance peptide classification using ESM & ProtT5 embeddings with AutoGluon ensemble learning</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/Models-ESM%20%7C%20ProtT5-green" />
  <img src="https://img.shields.io/badge/Framework-AutoGluon-orange" />
  <img src="https://img.shields.io/badge/Metric-ROC--AUC-red" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

---

## 📌 Overview

This project implements a machine learning pipeline for **peptide sequence classification** into binary classes (**+1 / -1**).

The approach leverages:
- Pretrained **protein language models (ESM, ProtT5)**  
- Feature fusion techniques  
- **AutoGluon ensemble learning (stacking + bagging)**  

to achieve strong predictive performance on unseen data.

---

## 🎯 Problem Statement

To develop a machine learning model that accurately predicts whether a given peptide sequence belongs to the **positive (+1)** or **negative (-1)** class.

---

## 🚀 Key Highlights

- Protein embeddings using:
  - **ESM**
  - **ProtT5**
- Feature fusion with sequence length augmentation  
- Automated model selection using **AutoGluon**  
- **10-fold bagging + stacked ensemble learning**  
- Multi-seed prediction averaging  
- Scalable and reproducible ML pipeline  

---

## 🧠 Methodology

### 🔬 Feature Engineering
- Generated embeddings using:
  - ESM  
  - ProtT5  
- Combined embeddings into a unified feature representation  
- Added **sequence length** as an additional feature  
- Final feature space: **~1345 features**

---

### ⚙️ Data Preprocessing
- Removed non-feature columns (e.g., sequence IDs)  
- Converted labels from **(-1, +1) → (0, 1)**  
- Standardized features using **StandardScaler**

---

### 🤖 Model Training
- Used **AutoGluon Tabular** for:
  - Model selection  
  - Hyperparameter tuning  
  - Ensemble construction  
- Techniques applied:
  - **10-fold bagging**
  - **Stacked ensemble learning**
- Evaluation metric: **ROC-AUC**

---

### 📊 Inference
- Trained multiple models with different random seeds  
- Averaged predictions for final output  

---

## 📈 Results

| Model                     | ROC-AUC |
|--------------------------|--------|
| LightGBMXT (Stacked)     | 0.9336 |
| LightGBM                 | 0.8521 |
| Random Forest            | 0.7995 |
| CatBoost                 | 0.8054 |

🏆 **Best Kaggle Score:** 0.87  

---

## 📂 Project Structure

```bash

├── input/
│   ├── train.csv
│   ├── test.csv
│   
├── outputs/
│   └── submission.csv
│
├── Code.py
├── requirements.txt
└── README.md
