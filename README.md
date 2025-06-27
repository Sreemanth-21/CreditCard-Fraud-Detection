# 💳 Credit Card Fraud Detection

This project detects fraudulent credit card transactions using machine learning. It combines anomaly detection and supervised learning (XGBoost) on a real-world dataset from Kaggle.

---

## 🚀 Project Overview

- **Objective:** Identify fraudulent transactions from credit card data.
- **Tech Stack:** Python, Scikit-learn, XGBoost, Streamlit
- **Models Used:**
  - Isolation Forest
  - Local Outlier Factor
  - XGBoost Classifier
- **UI:** Built using Streamlit for live transaction prediction.

---

## 📁 Dataset

Dataset used: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions
- 492 frauds (~0.17%)
- Features: PCA-transformed V1–V28, `Time`, `Amount`
- Target: `Class` (0 = legitimate, 1 = fraud)

> Note: Dataset is not included in this repo. Download manually from Kaggle.

---

## ⚙️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
