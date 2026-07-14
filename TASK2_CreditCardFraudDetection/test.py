# TEST.py

import pandas as pd
import joblib
import numpy as np

#LOADING TESTING DATESET
df = pd.read_csv("fraudTest.csv")

original_df = df.copy()

model_lr = joblib.load("model_lr.pkl") #logistic regression model
model_dt = joblib.load("model_dt.pkl") #decision tree model
model_rf = joblib.load("model_rf.pkl") #random forest model

label_encoders = joblib.load("encoders.pkl")
train_columns = joblib.load("columns.pkl")


drop_cols = ["first", "last", "street", "trans_num"]
df = df.drop(columns=drop_cols, errors="ignore")

df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour"] = df["trans_date_trans_time"].dt.hour
df["day"] = df["trans_date_trans_time"].dt.day
df["month"] = df["trans_date_trans_time"].dt.month
df = df.drop(columns=["trans_date_trans_time"])

for col, le in label_encoders.items():
    if col in df.columns:
        df[col] = df[col].astype(str)

        # Handle unseen labels
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else "Unknown")

        # Add Unknown class if missing
        if "Unknown" not in le.classes_:
            le.classes_ = np.append(le.classes_, "Unknown")

        df[col] = le.transform(df[col])


df = df.reindex(columns=train_columns, fill_value=0)



y_pred_lr = model_lr.predict(df)
y_prob_lr = model_lr.predict_proba(df)[:, 1]

y_pred_dt = model_dt.predict(df)
y_prob_dt = model_dt.predict_proba(df)[:, 1]

y_pred_rf = model_rf.predict(df)
y_prob_rf = model_rf.predict_proba(df)[:, 1]

output = original_df.copy()

output["LR_Predicted"] = y_pred_lr
output["LR_Prob"] = y_prob_lr

output["DT_Predicted"] = y_pred_dt
output["DT_Prob"] = y_prob_dt

output["RF_Predicted"] = y_pred_rf
output["RF_Prob"] = y_prob_rf


#MAJORITY VOTE for better Prediction
output["Final_Prediction"] = (
    (output["LR_Predicted"] + output["DT_Predicted"] + output["RF_Predicted"]) >= 2
).astype(int)

# Weighted Probability
output["Final_Probability"] = (
    0.4 * output["LR_Prob"] +
    0.3 * output["DT_Prob"] +
    0.3 * output["RF_Prob"]
)


def final_alert(row):
    if row["Final_Prediction"] == 1 and row["Final_Probability"] >= 0.6:
        return "FRAUD (HIGH RISK)"
    else:
        return "SAFE"

output["Final_Alert"] = output.apply(final_alert, axis=1)


output = output.sort_values(by="Final_Probability", ascending=False)

#saving result in excel(.csv) file
output.to_csv("fraud_predictions.csv", index=False)

#RESULT
print("\n Prediction completed successfully!\n")

print("TOP HIGH RISK TRANSACTIONS:\n")
print(output[[
    "amt", "merchant",
    "Final_Probability", "Final_Alert"
]].head(10))

fraud_cases = output[output["Final_Alert"].str.contains("FRAUD")]
