# TRAIN.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#LOADING TRAINING DATESET
df = pd.read_csv("fraudTrain.csv")


drop_cols = ["first", "last", "street", "trans_num"]
df = df.drop(columns=drop_cols, errors="ignore")


df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour"] = df["trans_date_trans_time"].dt.hour
df["day"] = df["trans_date_trans_time"].dt.day
df["month"] = df["trans_date_trans_time"].dt.month
df = df.drop(columns=["trans_date_trans_time"])

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le


# Split Features & Target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train Models
print("Training Logistic Regression...")
model_lr = LogisticRegression(max_iter=200)
model_lr.fit(X_train, y_train)

print("Training Decision Tree...")
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)

print("Training Random Forest...")

model_rf = RandomForestClassifier (
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",   
    n_jobs=-1
)

model_rf.fit(X_train, y_train)


# Save Models
joblib.dump(model_lr, "model_lr.pkl")
joblib.dump(model_dt, "model_dt.pkl")
joblib.dump(model_rf, "model_rf.pkl")

# Save encoders
joblib.dump(label_encoders, "encoders.pkl")

# Save feature columns
joblib.dump(X.columns.tolist(), "columns.pkl")

print("Training completed and models saved!")