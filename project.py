import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# -------------------------------
# 1) Load Data
# -------------------------------
df = pd.read_csv("train.csv")

# Drop rows with missing SalePrice
df = df.dropna(subset=["SalePrice"])

# Separate features and target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Identify categorical & numeric columns
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns


# -------------------------------
# 2) Preprocessing Pipeline
# -------------------------------
def build_pipeline(cat_cols):
    preprocess = ColumnTransformer(
        transformers=[
            ("onehot",
             OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             cat_cols)
        ],
        remainder="passthrough"
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    return pipeline


# -------------------------------
# 3) Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = build_pipeline(cat_cols)

pipeline.fit(X_train, y_train)

# -------------------------------
# 4) Evaluation
# -------------------------------
preds = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"RMSE: {rmse}")

# -------------------------------
# 5) Save Model
# -------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("model.pkl saved successfully!")


# -------------------------------
# Save column names for Flask App
# -------------------------------
train_cols = X.columns
joblib.dump(train_cols, "columns.pkl")
print("columns.pkl saved!")


# -------------------------------
# Optional: Feature Importance Plot
# -------------------------------
try:
    model = pipeline.named_steps["model"]
    importance = model.feature_importances_

    plt.figure(figsize=(8, 6))
    plt.plot(importance)
    plt.title("Feature Importance (XGBoost)")
    plt.savefig("feature_importance.png")
    plt.close()

    print("feature_importance.png saved!")

except:
    print("Could not generate feature importance.")
