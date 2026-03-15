# ================= IMPORT LIBRARIES =================

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest


# =================  FULL AUTO CLEANER FUNCTION =================

def full_auto_cleaner(df):

    print("\n AUTO CLEANING STARTED")

    df = df.copy()

    # remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()] # data jevhe pan column na tyan properly removed all duplicated columns (same name)

    # remove duplicate rows
    df = df.drop_duplicates() # kadhun tak same report

    # replace garbage values
    df = df.replace(
        ["None","none","NULL","null","\\",""," ","-","--"],np.nan) # ya ja pan values yetil replace kar

    # try numeric convert
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore") # string data convert karto numeric

    # smart missing fill
    for col in df.select_dtypes(include=np.number): 
        skew_val = df[col].skew()
        if abs(skew_val) < 0.5:
            df[col] = df[col].fillna(df[col].mean()) # nehmi small values mean fill karto 
        else:
            df[col] = df[col].fillna(df[col].median()) # mothi vaules data middel part fill karto median ne

    for col in df.select_dtypes(include="object"):

        mode_val = df[col].mode()

        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val[0])
        else:
            df[col] = df[col].fillna("unknown")

    #  IQR OUTLIER CLEANING
    for col in df.select_dtypes(include=np.number):

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.where(df[col] > upper, upper, df[col])
        df[col] = np.where(df[col] < lower, lower, df[col])

    # SKEW REDUCTION
    for col in df.select_dtypes(include=np.number):

        if df[col].skew() > 0.75:
            df[col] = np.log1p(df[col])

    print("Auto cleaning is finished")

    return df


# ================= LOAD TRAIN FILES =================

finance = pd.read_csv(r"E:\VISHAKHA PYTHON ALL IMPORTANT DATA VERY IMPORTANT DATA\Vishakha ml small project\Ai-dataset-cleaner\data\train\finance-data-train.csv")
health = pd.read_csv(r"E:\VISHAKHA PYTHON ALL IMPORTANT DATA VERY IMPORTANT DATA\Vishakha ml small project\Ai-dataset-cleaner\data\train\health-data-train.csv")
education = pd.read_csv(r"E:\VISHAKHA PYTHON ALL IMPORTANT DATA VERY IMPORTANT DATA\Vishakha ml small project\Ai-dataset-cleaner\data\train\education-data-train.csv")


# ================= ADD--DATASET--LABEL ============

finance["dataset_type"] = "finance"
health["dataset_type"] = "health"
education["dataset_type"] = "education"


# ================= MERGE ALL DATA =================

train_data = pd.concat([finance, health, education], ignore_index=True)

print("\n Train data merged successfully")
print(train_data.shape)


# ================= BASIC CLEANING =================

train_data = train_data.drop(columns=["transaction_id"], errors="ignore")

target = train_data["dataset_type"]

train_data = train_data.drop(columns=["dataset_type"])

# CALL AUTO CLEANER HERE (FLOW SAME)
train_data = full_auto_cleaner(train_data)
# ================= CONVERT CATEGORICAL TO NUMERIC =================

train_data = pd.get_dummies(train_data, drop_first=True)
train_columns = train_data.columns
# ================ VISUALIZATION FUNCTION =================

def auto_visualization(df):

    print("\n starting auto visulization")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in num_cols[:4]:
        plt.figure(figsize=(5,3))
        sns.histplot(df[col], kde=True)
        plt.title(f"Histogram : {col}")
        plt.tight_layout()
        plt.show()

    print("\n done auto visulization")

auto_visualization(train_data)

# ================= SCALING =================

scaler = StandardScaler() # it mens sagle no sarkh karto (1,0,-1 scale karto data tycha convience  mean =0, stdv=1 model stabel )
train_scaled = scaler.fit_transform(train_data)

# ================= MODEL TRAIN =================

rf_model = RandomForestClassifier(n_estimators=50,random_state=42)

rf_model.fit(train_scaled, target)

# ================= ANOMALY MODEL =================

iso_model = IsolationForest(n_estimators=50,contamination=0.05,random_state=42)
iso_model.fit(train_scaled)
# ================= CREATE MODELS FOLDER =================
os.makedirs("models", exist_ok=True)
# ================= SAVE MODELS =================

joblib.dump(rf_model, "models/rf_model.pkl")
joblib.dump(iso_model, "models/iso_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(train_columns, "models/train_columns.pkl")

print("\n models saved successfully")


# ================= PREDICTION FUNCTION ============

def predict_new_dataset(file_path): # file path yeil 
    new_df = pd.read_csv(file_path)

    new_df = new_df.drop(columns=["transaction_id"], errors="ignore") # column training removed kela hota tr test removed hoto 

    # CALL AUTO CLEANER AGAIN
    new_df = full_auto_cleaner(new_df)

    new_df = pd.get_dummies(new_df, drop_first=True) # first removed string nahi samjt na tyamule

    new_df = new_df.reindex(columns=train_columns, fill_value=0) # je training colum thevto tech prectiction karto so ith imp

    new_scaled = scaler.transform(new_df)

    preds = rf_model.predict(new_scaled)

    final = pd.Series(preds).mode()[0]

    # save cleaned file
    new_df.to_csv("cleaned_output.csv", index=False)

    return final
# ================= TEST PREDICTION ==============

test_file = r"E:\VISHAKHA PYTHON ALL IMPORTANT DATA VERY IMPORTANT DATA\Vishakha ml small project\Ai-dataset-cleaner\data\test\SuperMarket Analysis.csv"
result = predict_new_dataset(test_file)
print("\n final predict data-set type :", result)
