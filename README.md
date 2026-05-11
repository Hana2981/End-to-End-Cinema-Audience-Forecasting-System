# 🎬 End-to-End Cinema Audience Forecasting System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/ML-LightGBM%20%7C%20XGBoost%20%7C%20Random%20Forest-green)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

This project presents a complete end-to-end machine learning pipeline designed to **forecast daily cinema audience attendance** across multiple theater locations. By leveraging historical booking data, theater metadata, and temporal patterns, the system delivers accurate predictions of audience counts to help cinema operators make data-driven decisions.

> **Challenge:** Predict daily theatre audience counts across multiple locations using historical data and machine learning techniques.

---

## 🎯 Objectives

- Build a robust ML pipeline for daily audience count forecasting
- Perform in-depth Exploratory Data Analysis (EDA) to uncover patterns
- Engineer meaningful features from temporal, booking, and theater data
- Train and compare multiple regression models
- Apply hyperparameter tuning to maximize predictive performance
- Generate a submission-ready prediction file

---

## 📁 Dataset Description

The project uses data from two booking systems — **BookNow** (online) and **CinePOS** (point-of-sale) — along with supporting metadata:

| Dataset | Description |
|---|---|
| `booknow_visits` | Daily audience counts per theater *(Target Variable)* |
| `booknow_theaters` | Theater metadata from the BookNow platform |
| `cinepos_theaters` | Theater metadata from the CinePOS system |
| `theater_relation` | Mapping between BookNow and CinePOS theater IDs |
| `booknow_booking` | Historical online booking transactions |
| `cinepos_booking` | Historical point-of-sale booking transactions |
| `date_info` | Calendar and holiday information |
| `sample_submission` | Test set format for generating predictions |

---

## 🔄 Pipeline Architecture

```
Raw Data → EDA → Cleaning → Feature Engineering → Modeling → Tuning → Submission
```

### 1. 📊 Exploratory Data Analysis (EDA)
- **Target Analysis:** Distribution of `audience_count` revealed a right-skewed pattern; log1p transformation applied for better model stability
- **Temporal Analysis:** Clear seasonal patterns observed — peaks in April–May (summer vacation), drops during monsoon (July–October), and a dip in December
- **Day-of-Week Trends:** Weekends (Saturday & Sunday) consistently show higher audience counts
- **Theater Analysis:** Audience behavior varies significantly by theater type and area
- **Booking Patterns:** Online bookings average ~9 days lead time; POS bookings are mostly same-day

### 2. 🧹 Data Cleaning
- Removed rows with missing theater IDs
- Filled missing theater type and area with `'Unknown'`
- Dropped latitude/longitude columns due to excessive missing values
- Standardized all date columns to `datetime` format

### 3. 🛠️ Feature Engineering

| Feature Category | Features Created |
|---|---|
| **Date Features** | Year, Month, Day, Week of Year, Quarter, Is Weekend, Season |
| **Theater Features** | Theater type, area, missing info flag |
| **Online Booking** | Total tickets, booking count, avg/min/max lead days |
| **POS Booking** | Total tickets, transaction count, avg tickets per transaction |
| **Combined Booking** | Total tickets (all channels), online/POS ratio |
| **Lag Features** | Audience lag at 7, 14, 21, 28 days |
| **Rolling Features** | 7, 14, 28-day rolling mean and standard deviation |
| **Theater Aggregates** | Avg, std, median audience and total shows per theater |

### 4. ⚙️ Feature Scaling
Applied **RobustScaler** (using median and IQR) for models sensitive to outliers — making the pipeline robust against the extreme audience values common in cinema data.

### 5. 🤖 Model Training & Evaluation

Nine regression models were trained and evaluated using **R² score**:

| Model | Scaling Used |
|---|---|
| Linear Regression | ✅ Yes |
| SGD Regressor | ✅ Yes |
| Ridge Regression | ✅ Yes |
| Lasso Regression | ✅ Yes |
| ElasticNet | ✅ Yes |
| K-Nearest Neighbors | ✅ Yes |
| Random Forest | ❌ No |
| XGBoost | ❌ No |
| LightGBM | ❌ No |

### 6. 🔧 Hyperparameter Tuning
Applied **RandomizedSearchCV** (20 iterations, 3-fold cross-validation) on XGBoost and LightGBM:

**XGBoost parameters tuned:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`

**LightGBM parameters tuned:** `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `subsample`

### 7. 🏆 Final Model
The best-performing tuned model (**LightGBM**) was retrained on the full dataset (train + validation) and used to generate final predictions.

---

## 📈 Evaluation Metric

**R² Score (Coefficient of Determination)** — measures how well the model explains variance in daily audience counts. A higher R² indicates better predictive performance.

---

## 🗂️ Project Structure

```
End-to-End-Cinema-Audience-Forecasting-System/
│
├── 23ds3000121-notebook-t32025.ipynb   # Main Kaggle notebook
├── submission.csv                       # Final predictions (generated)
└── README.md                            # Project documentation
```

---

## 🚀 How to Run

1. Open the notebook on [Kaggle](https://www.kaggle.com/code/hanamuhammed93/23ds3000121-notebook-t32025)
2. Attach the **Cinema Audience Forecasting Challenge** dataset
3. Run all cells sequentially from top to bottom
4. The final `submission.csv` will be generated in `/kaggle/working/`

---

## 🧰 Libraries Used

```python
numpy, pandas, matplotlib, seaborn
scikit-learn (Pipeline, GridSearchCV, RandomizedSearchCV, RobustScaler)
xgboost, lightgbm
```

---

## 👩‍💻 Author

**Hana Mohammed**
- 🎓 Student ID: 23DS3000121
- 📧 23ds3000121@ds.study.iitm.ac.in
- 🔗 [GitHub](https://github.com/Hana2981) | [Kaggle](https://www.kaggle.com/hanamuhammed93)

---

## 📄 License

This project is submitted as part of an academic assignment. All rights reserved.
