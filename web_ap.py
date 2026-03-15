# web_app.py

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --------- Streamlit Page Config ---------
st.set_page_config(page_title="AI Dataset Type Predictor", layout="wide")
st.title("📊  *AI Dataset Cleaner  And Visualization * ")


# ***** FIXED MODEL LOADING FOR STREAMLIT CLOUD *****

rf_model = joblib.load("rf_model.pkl")
iso_model = joblib.load("iso_model.pkl")
scaler = joblib.load("scaler.pkl")
train_columns = joblib.load("train_columns.pkl")

# FULL AUTO CLEANING FUNCTION
def auto_clean_dataframe(df):
    df = df.copy()
    # remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    # remove duplicate rows
    df = df.drop_duplicates()
    # replace garbage
    df = df.replace(["None","none","NULL","null","\\",""," ","-","--"],np.nan)

    # try numeric convert
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # numeric clean
    for col in df.select_dtypes(include=np.number):

        skew_val = df[col].skew()

        if abs(skew_val) < 0.5:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].median())

        # IQR cap
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.where(df[col] > upper, upper, df[col])
        df[col] = np.where(df[col] < lower, lower, df[col])

        # skew reduction
        if df[col].skew() > 0.75:
            df[col] = np.log1p(df[col])

    # categorical clean
    for col in df.select_dtypes(include="object"):

        mode_val = df[col].mode()

        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val[0])
        else:
            df[col] = df[col].fillna("unknown")

    return df


# --------- Function to Prepare New Data ---------
def prepare_new_data(df_new, train_columns, scaler):

    df_new = df_new.copy()
    cat_cols = df_new.select_dtypes(include=['object']).columns

    if len(cat_cols) > 0:
        df_new = pd.get_dummies(df_new, columns=cat_cols, drop_first=False)

    df_new = df_new.reindex(columns=train_columns, fill_value=0)

    try:
        df_new = pd.DataFrame(scaler.transform(df_new),columns=train_columns)
    except Exception as e:
        st.error(f"Scaling Error: {e}")
        return None

    return df_new

#Auto visualization#
def auto_visualization(df):

    st.write("## Auto Data Visualization")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols) == 0:
        st.warning("No numeric columns")
        return

    CHART_WIDTH = 4
    CHART_HEIGHT = 3

    # Histogram
    st.write("### Histogram")
    for col in num_cols[:4]:

        fig = plt.figure(figsize=(CHART_WIDTH, CHART_HEIGHT))
        sns.histplot(df[col], kde=True)

        plt.title(col)
        st.pyplot(fig)
        plt.close()

    # Scatter
    if len(num_cols) >= 2:

        st.write("### Scatter Plot")

        fig = plt.figure(figsize=(CHART_WIDTH, CHART_HEIGHT))
        sns.scatterplot(x=df[num_cols[0]], y=df[num_cols[1]])

        st.pyplot(fig)
        plt.close()

    # Heatmap
    st.write("### Correlation Heatmap")

    fig = plt.figure(figsize=(CHART_WIDTH, CHART_HEIGHT))
    sns.heatmap(df[num_cols].corr(), cmap="coolwarm")

    st.pyplot(fig)
    plt.close()

    # Pairplot
    try:

        st.write("### Pair Plot")

        pair = sns.pairplot(df[num_cols[:4]].sample(min(120, len(df))),height=1.6)

        st.pyplot(pair)
        plt.close()

    except:
        st.warning("Pairplot skipped")



# ------------------- Main App -------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"CSV Read Error: {e}")
        st.stop()

    st.write("#### Original Dataset Preview")
    st.dataframe(df.head())

    if df.empty:
        st.warning("Uploaded dataset is empty.")
        st.stop()

    # CLEAN DATA HERE
    df = auto_clean_dataframe(df)

    st.write("#### Cleaned Dataset Preview")
    st.dataframe(df.head())

    # AUTO VISUALIZATION AFTER CLEANING
    auto_visualization(df)
