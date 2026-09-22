# ============================================================
# WALMART SALES PROJECT
# EDA + OUTLIER + ML + DEEP LEARNING + JOBLIB + GENAI-READY
# ============================================================

# ============================================================
# 1. INSTALLATION
# ============================================================

# Run these commands in terminal only once:
#
# pip install pandas numpy matplotlib seaborn scikit-learn
# pip install joblib tensorflow
#
# Optional GenAI:
# pip install openai


# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import os
import sys
import json
import platform
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")


# ============================================================
# 3. VERSION INFORMATION
# ============================================================

print("=" * 60)
print("ENVIRONMENT INFORMATION")
print("=" * 60)

print("Python:", sys.version)
print("Platform:", platform.platform())
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", __import__("sklearn").__version__)
print("Joblib:", joblib.__version__)
print("TensorFlow:", tf.__version__)


# Save environment information

environment_info = {
    "python": sys.version,
    "platform": platform.platform(),
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "sklearn": __import__("sklearn").__version__,
    "joblib": joblib.__version__,
    "tensorflow": tf.__version__
}

with open(
    "environment_info.json",
    "w"
) as f:

    json.dump(
        environment_info,
        f,
        indent=4
    )


# ============================================================
# 4. LOAD DATA
# ============================================================

df = pd.read_csv("Walmart DataSet.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


numeric_columns = [
    "Weekly_Sales",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment"
]


for column in numeric_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].median()
        )


# Holiday flag

if df["Holiday_Flag"].isnull().sum() > 0:

    df["Holiday_Flag"] = df[
        "Holiday_Flag"
    ].fillna(
        df["Holiday_Flag"].mode()[0]
    )


# Store

if df["Store"].isnull().sum() > 0:

    df["Store"] = df[
        "Store"
    ].fillna(
        df["Store"].mode()[0]
    )


print("\nMissing Values After Treatment:")
print(df.isnull().sum())


# ============================================================
# 6. DUPLICATES
# ============================================================

print("\nDuplicates:")
print(df.duplicated().sum())

df = df.drop_duplicates()


# ============================================================
# 7. DATE CONVERSION
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.dropna(
    subset=["Date"]
)

df = df.sort_values(
    ["Store", "Date"]
).reset_index(drop=True)


# ============================================================
# 8. DATE FEATURES
# ============================================================

df["Year"] = df["Date"].dt.year

df["Month"] = df["Date"].dt.month

df["Week"] = (
    df["Date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

df["Quarter"] = df[
    "Date"
].dt.quarter

df["Day"] = df[
    "Date"
].dt.day

df["DayOfWeek"] = df[
    "Date"
].dt.dayofweek


# ============================================================
# 9. BASIC STATISTICAL ANALYSIS
# ============================================================

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 10. STORE PERFORMANCE
# ============================================================

store_summary = df.groupby(
    "Store"
)["Weekly_Sales"].agg(
    [
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "std"
    ]
)

store_summary = store_summary.sort_values(
    "sum",
    ascending=False
)


print("\nStore Performance:")
print(store_summary)


top_store = store_summary[
    "sum"
].idxmax()

worst_store = store_summary[
    "sum"
].idxmin()


top_sales = store_summary.loc[
    top_store,
    "sum"
]

worst_sales = store_summary.loc[
    worst_store,
    "sum"
]


difference = (
    top_sales -
    worst_sales
)


percentage_difference = (
    difference /
    worst_sales
) * 100


print("\nTop Store:", top_store)
print("Worst Store:", worst_store)
print("Sales Difference:", difference)
print(
    "Percentage Difference:",
    percentage_difference
)


# ============================================================
# 11. OUTLIER DETECTION
# ============================================================

Q1 = df[
    "Weekly_Sales"
].quantile(0.25)

Q3 = df[
    "Weekly_Sales"
].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - (
    1.5 * IQR
)

upper_limit = Q3 + (
    1.5 * IQR
)


outliers = df[
    (df["Weekly_Sales"] < lower_limit) |
    (df["Weekly_Sales"] > upper_limit)
]


print("\nNumber of Outliers:")
print(len(outliers))


# Don't remove genuine sales spikes
# because holiday/promotional sales
# can naturally be high.


# ============================================================
# 12. UNEMPLOYMENT ANALYSIS
# ============================================================

unemployment_corr = df[
    [
        "Weekly_Sales",
        "Unemployment"
    ]
].corr()


print(
    "\nUnemployment Correlation:"
)

print(unemployment_corr)


unemployment_store = (
    df.groupby("Store")
    .apply(
        lambda x:
        x["Weekly_Sales"].corr(
            x["Unemployment"]
        )
    )
    .sort_values()
)


print(
    "\nUnemployment Correlation by Store:"
)

print(unemployment_store)


# ============================================================
# 13. TEMPERATURE ANALYSIS
# ============================================================

temperature_corr = df[
    [
        "Weekly_Sales",
        "Temperature"
    ]
].corr()


print(
    "\nTemperature Correlation:"
)

print(temperature_corr)


temperature_store = (
    df.groupby("Store")
    .apply(
        lambda x:
        x["Weekly_Sales"].corr(
            x["Temperature"]
        )
    )
    .sort_values()
)


print(
    "\nTemperature Correlation by Store:"
)

print(temperature_store)


# ============================================================
# 14. CPI ANALYSIS
# ============================================================

cpi_corr = df[
    [
        "Weekly_Sales",
        "CPI"
    ]
].corr()


print("\nCPI Correlation:")

print(cpi_corr)


cpi_store = (
    df.groupby("Store")
    .apply(
        lambda x:
        x["Weekly_Sales"].corr(
            x["CPI"]
        )
    )
    .sort_values()
)


print(
    "\nCPI Correlation by Store:"
)

print(cpi_store)


# ============================================================
# 15. SEASONAL ANALYSIS
# ============================================================

monthly_sales = df.groupby(
    "Month"
)["Weekly_Sales"].mean()


print("\nAverage Monthly Sales:")

print(monthly_sales)


quarter_sales = df.groupby(
    "Quarter"
)["Weekly_Sales"].mean()


print("\nAverage Quarterly Sales:")

print(quarter_sales)


# ============================================================
# 16. HOLIDAY ANALYSIS
# ============================================================

holiday_sales = df.groupby(
    "Holiday_Flag"
)["Weekly_Sales"].mean()


print(
    "\nHoliday vs Non-Holiday Sales:"
)

print(holiday_sales)


# ============================================================
# 17. CORRELATION MATRIX
# ============================================================

corr_columns = [
    "Weekly_Sales",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Holiday_Flag"
]


correlation_matrix = df[
    corr_columns
].corr()


print("\nCorrelation Matrix:")

print(correlation_matrix)


plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f"
)

plt.title(
    "Walmart Correlation Matrix"
)

plt.show()


# ============================================================
# 18. VISUALIZATION
# ============================================================

plt.figure(figsize=(14, 6))

store_summary[
    "sum"
].plot(
    kind="bar"
)

plt.title(
    "Historical Sales by Store"
)

plt.xlabel("Store")

plt.ylabel(
    "Total Weekly Sales"
)

plt.tight_layout()

plt.show()


# ============================================================
# 19. TIME SERIES
# ============================================================

total_sales = df.groupby(
    "Date"
)["Weekly_Sales"].sum()


plt.figure(figsize=(14, 6))

plt.plot(
    total_sales.index,
    total_sales.values
)

plt.title(
    "Overall Walmart Weekly Sales"
)

plt.xlabel("Date")

plt.ylabel(
    "Weekly Sales"
)

plt.grid()

plt.show()


# ============================================================
# 20. MACHINE LEARNING DATASET
# ============================================================

model_df = df.copy()

model_df = model_df.sort_values(
    ["Store", "Date"]
)


# ============================================================
# 21. LAG FEATURES
# ============================================================

for lag in [
    1,
    2,
    3,
    4,
    12
]:

    model_df[
        f"Lag_{lag}"
    ] = (
        model_df
        .groupby("Store")[
            "Weekly_Sales"
        ]
        .shift(lag)
    )


# ============================================================
# 22. ROLLING FEATURES
# ============================================================

model_df[
    "Rolling_Mean_4"
] = (
    model_df
    .groupby("Store")[
        "Weekly_Sales"
    ]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(4)
        .mean()
    )
)


model_df[
    "Rolling_Mean_12"
] = (
    model_df
    .groupby("Store")[
        "Weekly_Sales"
    ]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(12)
        .mean()
    )
)


model_df = model_df.dropna()


# ============================================================
# 23. FEATURES
# ============================================================

features = [

    "Store",

    "Holiday_Flag",

    "Temperature",

    "Fuel_Price",

    "CPI",

    "Unemployment",

    "Year",

    "Month",

    "Week",

    "Quarter",

    "DayOfWeek",

    "Lag_1",

    "Lag_2",

    "Lag_3",

    "Lag_4",

    "Lag_12",

    "Rolling_Mean_4",

    "Rolling_Mean_12"

]


target = "Weekly_Sales"


# ============================================================
# 24. TIME BASED TRAIN TEST SPLIT
# ============================================================

split_date = model_df[
    "Date"
].quantile(0.80)


train_data = model_df[
    model_df["Date"] <= split_date
]


test_data = model_df[
    model_df["Date"] > split_date
]


X_train = train_data[
    features
]

y_train = train_data[
    target
]


X_test = test_data[
    features
]

y_test = test_data[
    target
]


# ============================================================
# 25. RANDOM FOREST MODEL
# ============================================================

rf_model = RandomForestRegressor(

    n_estimators=300,

    max_depth=20,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1

)


rf_model.fit(
    X_train,
    y_train
)


# ============================================================
# 26. RANDOM FOREST PREDICTION
# ============================================================

rf_prediction = rf_model.predict(
    X_test
)


# ============================================================
# 27. EVALUATION
# ============================================================

rf_mae = mean_absolute_error(
    y_test,
    rf_prediction
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_prediction
    )
)

rf_r2 = r2_score(
    y_test,
    rf_prediction
)


print("\n")
print("=" * 60)
print("RANDOM FOREST PERFORMANCE")
print("=" * 60)

print("MAE :", rf_mae)

print("RMSE:", rf_rmse)

print("R2  :", rf_r2)


# ============================================================
# 28. SAVE RANDOM FOREST USING JOBLIB
# ============================================================

joblib.dump(
    rf_model,
    "walmart_random_forest.joblib"
)


print(
    "\nRandom Forest saved successfully."
)


# ============================================================
# 29. SAVE MODEL METADATA
# ============================================================

model_metadata = {

    "model": "RandomForestRegressor",

    "features": features,

    "target": target,

    "n_estimators": 300,

    "max_depth": 20,

    "random_state": 42,

    "training_date": str(
        pd.Timestamp.now()
    ),

    "python_version": sys.version,

    "sklearn_version":
        __import__("sklearn").__version__,

    "joblib_version":
        joblib.__version__

}


with open(
    "random_forest_metadata.json",
    "w"
) as f:

    json.dump(
        model_metadata,
        f,
        indent=4
    )


# ============================================================
# 30. LOAD JOBLIB MODEL
# ============================================================

loaded_rf_model = joblib.load(
    "walmart_random_forest.joblib"
)


print(
    "\nJoblib model loaded successfully."
)


# Test loaded model

test_prediction = loaded_rf_model.predict(
    X_test.head(5)
)


print(
    "\nTest predictions from loaded model:"
)

print(test_prediction)


# ============================================================
# 31. FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({

    "Feature": features,

    "Importance":
        rf_model.feature_importances_

})


importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


print(
    "\nFeature Importance:"
)

print(importance_df)


plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.show()


# ============================================================
# 32. DEEP LEARNING MODEL
# ARTIFICIAL NEURAL NETWORK
# ============================================================

# Scale features for neural network

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 33. ANN MODEL
# ============================================================

ann_model = keras.Sequential([

    layers.Input(
        shape=(X_train_scaled.shape[1],)
    ),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.20),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dropout(0.20),

    layers.Dense(
        32,
        activation="relu"
    ),

    layers.Dense(
        1
    )

])


ann_model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=["mae"]

)


ann_model.summary()


# ============================================================
# 34. TRAIN ANN
# ============================================================

early_stopping = keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True

)


history = ann_model.fit(

    X_train_scaled,

    y_train,

    validation_split=0.10,

    epochs=100,

    batch_size=32,

    callbacks=[
        early_stopping
    ],

    verbose=1

)


# ============================================================
# 35. ANN PREDICTION
# ============================================================

ann_prediction = (
    ann_model
    .predict(
        X_test_scaled,
        verbose=0
    )
    .flatten()
)


# ============================================================
# 36. ANN EVALUATION
# ============================================================

ann_mae = mean_absolute_error(
    y_test,
    ann_prediction
)

ann_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        ann_prediction
    )
)

ann_r2 = r2_score(
    y_test,
    ann_prediction
)


print("\n")
print("=" * 60)
print("DEEP LEARNING ANN PERFORMANCE")
print("=" * 60)

print("MAE :", ann_mae)

print("RMSE:", ann_rmse)

print("R2  :", ann_r2)


# ============================================================
# 37. SAVE ANN
# ============================================================

ann_model.save(
    "walmart_ann_model.keras"
)


# Save scaler using joblib

joblib.dump(
    scaler,
    "walmart_ann_scaler.joblib"
)


print(
    "\nANN model and scaler saved."
)


# ============================================================
# 38. SAVE ANN METADATA
# ============================================================

ann_metadata = {

    "model": "Artificial Neural Network",

    "features": features,

    "target": target,

    "input_features":
        X_train_scaled.shape[1],

    "tensorflow_version":
        tf.__version__,

    "training_date":
        str(pd.Timestamp.now())

}


with open(
    "ann_metadata.json",
    "w"
) as f:

    json.dump(
        ann_metadata,
        f,
        indent=4
    )


# ============================================================
# 39. LOAD ANN MODEL
# ============================================================

loaded_ann_model = keras.models.load_model(
    "walmart_ann_model.keras"
)


loaded_scaler = joblib.load(
    "walmart_ann_scaler.joblib"
)


print(
    "\nANN model loaded successfully."
)


# ============================================================
# 40. GENAI-READY BUSINESS INSIGHT GENERATOR
# ============================================================

# This section creates structured information
# that can be passed to a GenAI model.
#
# We DO NOT hard-code a particular AI provider.
# This makes the project easier to maintain.


business_summary = {

    "top_store":
        int(top_store),

    "worst_store":
        int(worst_store),

    "top_store_sales":
        float(top_sales),

    "worst_store_sales":
        float(worst_sales),

    "sales_difference":
        float(difference),

    "sales_difference_percentage":
        float(percentage_difference),

    "unemployment_correlation":
        float(
            unemployment_corr.loc[
                "Weekly_Sales",
                "Unemployment"
            ]
        ),

    "temperature_correlation":
        float(
            temperature_corr.loc[
                "Weekly_Sales",
                "Temperature"
            ]
        ),

    "cpi_correlation":
        float(
            cpi_corr.loc[
                "Weekly_Sales",
                "CPI"
            ]
        ),

    "random_forest_mae":
        float(rf_mae),

    "random_forest_rmse":
        float(rf_rmse),

    "random_forest_r2":
        float(rf_r2),

    "ann_mae":
        float(ann_mae),

    "ann_rmse":
        float(ann_rmse),

    "ann_r2":
        float(ann_r2)

}


with open(
    "walmart_ai_business_summary.json",
    "w"
) as f:

    json.dump(
        business_summary,
        f,
        indent=4
    )


# ============================================================
# 41. GENAI PROMPT
# ============================================================

genai_prompt = f"""

You are a retail sales analytics assistant.

Analyze the following Walmart sales analysis.

Top store:
{top_store}

Worst store:
{worst_store}

Top store total sales:
{top_sales}

Worst store total sales:
{worst_sales}

Difference:
{difference}

Difference percentage:
{percentage_difference:.2f}%

Weekly Sales vs Unemployment correlation:
{business_summary["unemployment_correlation"]:.4f}

Weekly Sales vs Temperature correlation:
{business_summary["temperature_correlation"]:.4f}

Weekly Sales vs CPI correlation:
{business_summary["cpi_correlation"]:.4f}

Random Forest MAE:
{rf_mae}

Random Forest RMSE:
{rf_rmse}

Random Forest R2:
{rf_r2}

ANN MAE:
{ann_mae}

ANN RMSE:
{ann_rmse}

ANN R2:
{ann_r2}

Generate a concise business report containing:

1. Sales performance
2. Store-level observations
3. Unemployment relationship
4. Temperature relationship
5. CPI relationship
6. Seasonal observations
7. Machine learning performance
8. Deep learning performance
9. Business considerations for future sales planning

Do not invent information that is not present in the data.
Clearly distinguish correlation from causation.
"""


with open(
    "genai_prompt.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(genai_prompt)


print(
    "\nGenAI prompt saved as genai_prompt.txt"
)


# ============================================================
# 42. FINAL PROJECT SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("WALMART AI SALES PROJECT COMPLETED")
print("=" * 60)

print(
    "\nTop Store:",
    top_store
)

print(
    "Worst Store:",
    worst_store
)

print(
    "Sales Difference:",
    difference
)

print(
    "\nRandom Forest R2:",
    rf_r2
)

print(
    "ANN R2:",
    ann_r2
)

print(
    "\nSaved Files:"
)

print(
    "1. walmart_random_forest.joblib"
)

print(
    "2. random_forest_metadata.json"
)

print(
    "3. walmart_ann_model.keras"
)

print(
    "4. walmart_ann_scaler.joblib"
)

print(
    "5. ann_metadata.json"
)

print(
    "6. environment_info.json"
)

print(
    "7. walmart_ai_business_summary.json"
)

print(
    "8. genai_prompt.txt"
)

print(
    "\nProject finished successfully! 🚀"
)