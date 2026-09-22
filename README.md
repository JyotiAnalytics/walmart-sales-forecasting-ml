# 🛒 Walmart Sales Forecasting & Business Intelligence

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-RandomForest-f7931e?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

End-to-end sales forecasting project on Walmart's historical weekly sales dataset — covering EDA, outlier detection, time-based feature engineering, store-performance analysis, and predictive modeling using both Random Forest and a Deep Learning (ANN) model, with a GenAI-ready business summary generator.

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#️-dataset)
- [Workflow](#-workflow)
- [Results](#-results)
- [GenAI-Ready Insights](#-genai-ready-insights)
- [Tech Stack](#️-tech-stack)
- [Repository Structure](#-repository-structure)
- [How to Run](#️-how-to-run)
- [Future Improvements](#-future-improvements)

## 📌 Project Overview

Retailers need to understand *what* drives sales and *forecast* future sales accurately to plan inventory, staffing, and promotions. This project analyzes Walmart's weekly sales across 45 stores and builds two regression models — Random Forest and an Artificial Neural Network — to predict `Weekly_Sales`, along with a structured business-summary generator that's ready to feed into a GenAI report writer.

## 🗂️ Dataset

- **Source:** Walmart historical weekly sales dataset (45 stores, multiple years)
- **Key columns:** `Store`, `Date`, `Weekly_Sales`, `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`

## 🔧 Workflow

1. **Data Cleaning** — missing value imputation (median for numeric, mode for categorical), duplicate removal, date parsing
2. **Feature Engineering** — extracted `Year`, `Month`, `Week`, `Quarter`, `DayOfWeek`; built lag features (`Lag_1`, `Lag_2`, `Lag_3`, `Lag_4`, `Lag_12`) and rolling means (`Rolling_Mean_4`, `Rolling_Mean_12`) to capture sales trends and seasonality
3. **Exploratory Data Analysis** — store-level sales performance, outlier detection (IQR method), seasonal/monthly sales patterns
4. **Correlation Analysis** — Weekly Sales vs Unemployment, Temperature, and CPI, both overall and store-by-store
5. **Model Building**
   - **Random Forest Regressor** (300 trees, max depth 20)
   - **Artificial Neural Network** (Keras — 128→64→32 dense layers with Dropout, early stopping)
6. **Model Evaluation** — MAE, RMSE, R² for both models
7. **GenAI-Ready Reporting** — structured JSON business summary + a ready-to-use prompt template for generating a written business report from the numbers (provider-agnostic — works with any LLM)
8. **Model Saving** — trained models, scaler, and metadata saved for reuse

## 📊 Results

| Metric | Random Forest | ANN (Deep Learning) |
|---|---|---|
| MAE | 90,260.92 | 134,073.32 |
| RMSE | 128,106.95 | 167,755.70 |
| R² | 0.9493 | 0.9131 |

**Key findings:**
- **Top store (#20)** generated $123.7M in total sales vs. **worst store (#33)** at $15.2M — a **711% gap**, pointing to major store-level variation
- Weekly sales showed only **weak correlation** with Unemployment (−0.099), Temperature (−0.035), and CPI (−0.078) — none of these macro factors strongly drive weekly sales on their own (correlation, not causation)
- **Random Forest outperformed the ANN** on this tabular, feature-engineered dataset — a common pattern when strong lag/rolling features already do most of the work

## 🤖 GenAI-Ready Insights

The project outputs `walmart_ai_business_summary.json` (structured metrics) and `genai_prompt.txt` (a ready-made prompt) so the numeric results can be handed directly to any LLM to auto-generate a written business report — covering sales performance, store-level observations, seasonal patterns, and planning considerations, while explicitly distinguishing correlation from causation.

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `TensorFlow / Keras` · `Joblib`

## 📁 Repository Structure

```
├── WALMART.PY                       # Main script — full pipeline
├── README.md
├── Walmart_DataSet.csv              # Source dataset
├── random_forest_metadata.json      # RF model config & training info
├── ann_metadata.json                # ANN model config & training info
├── environment_info.json            # Library/environment versions used
├── walmart_ai_business_summary.json # Structured metrics for reporting
├── genai_prompt.txt                 # Ready-to-use LLM prompt template
└── *.joblib / *.keras                # Saved trained models & scaler (see note below)
```

> **Note:** `.joblib` / `.keras` model files and `settings.json` (a local editor config file) are excluded from version control — see [How to Run](#️-how-to-run) to regenerate them locally.

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib tensorflow
python WALMART.PY
```

Place `Walmart_DataSet.csv` in the same folder as the script before running.

## 🚀 Future Improvements

- Hyperparameter tuning (GridSearchCV / Optuna) for both models
- Add store-level forecasting dashboards (Power BI / Streamlit)
- Incorporate holiday-specific modeling given the `Holiday_Flag` feature
- Integrate the GenAI prompt with an actual LLM API call to auto-produce the business report end-to-end

## 👤 Author

*Jyoti Ranjan Bhanja.*
