# -*- coding: utf-8 -*-
"""app_bc.py — Breast Cancer Diagnosis Dashboard (Built-in Dataset Version)"""

# ============================================
# 🧬 Breast Cancer Diagnosis Prediction Dashboard
# Using XGBoost + Streamlit + Built-in Dataset
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.datasets import load_breast_cancer
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="Breast Cancer Dashboard", page_icon="🧬", layout="wide")

# ============================================
# STYLES
# ============================================
st.markdown("""
    <style>
        body { background: linear-gradient(120deg, #f5f8ff, #e9ecff); }
        .main-title { text-align: center; color: #4B0082; font-size: 40px; font-weight: 800; margin-bottom: 0; }
        .subheader { text-align: center; color: #555; font-size: 18px; margin-top: 5px; }
        .stTabs [data-baseweb="tab-list"] { justify-content: center; gap: 12px; }
        .stTabs [data-baseweb="tab"] {
            font-size: 18px; font-weight: 700; color: #4B0082 !important;
            padding: 12px 26px; border-radius: 8px !important;
            background-color: #f3f1ff; transition: all 0.3s ease-in-out;
        }
        .stTabs [data-baseweb="tab"]:hover { background-color: #e6dbff !important; }
        .stTabs [aria-selected="true"] { background-color: #4B0082 !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown('<h1 class="main-title">🧬 Breast Cancer Diagnosis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Built with Streamlit | Powered by XGBoost | Embedded Dataset</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# LOAD BUILT-IN DATASET (NO FILE NEEDED)
# ============================================
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df.rename(columns={'target': 'diagnosis'}, inplace=True)

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ============================================
# LOAD TRAINED MODEL
# ============================================
MODEL_PATH = "_model.pkl"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
    st.error("⚠️ Model file '_model.pkl' not found or empty. Please train and save it using joblib.dump().")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model:\n\n**{e}**")
    st.stop()

# ============================================
# MODEL EVALUATION
# ============================================
try:
    y_pred = model.predict(X)
except Exception as e:
    st.error(f"⚠️ Error during prediction:\n\n{e}")
    st.stop()

accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

# ============================================
# DASHBOARD SUMMARY CARDS
# ============================================
st.markdown("""
    <style>
        .metric-card {
            background: linear-gradient(145deg, #e6e0ff, #ffffff);
            padding: 22px;
            border-radius: 14px;
            box-shadow: 0 6px 16px rgba(75, 0, 130, 0.15);
            text-align: center;
            transition: all 0.3s ease;
        }
        .metric-card:hover { transform: translateY(-4px); box-shadow: 0 8px 18px rgba(75, 0, 130, 0.25); }
        .metric-card h3 { color: #4B0082; font-size: 18px; margin-bottom: 6px; }
        .metric-card h2 { color: #2a004f; font-size: 28px; margin: 0; }
        .metric-card p { color: #555; font-size: 14px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 📊 Model Overview Summary")

col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"<div class='metric-card'><h3>Dataset Size</h3><h2>{df.shape[0]}</h2><p>Samples</p></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><h3>Features</h3><h2>{df.shape[1]-1}</h2><p>Input Variables</p></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><h3>Accuracy</h3><h2>{accuracy*100:.2f}%</h2><p>Model Performance</p></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-card'><h3>F1 Score</h3><h2>{f1:.2f}</h2><p>Balanced Measure</p></div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================
# TABS
# ============================================
tab1, tab2, tab3 = st.tabs(["🏠 Overview", "📊 Data Insights", "🤖 Prediction Interface"])

# TAB 1 — OVERVIEW
with tab1:
    st.header("🏠 Overview")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        This dashboard predicts whether a breast tumor is **Malignant** or **Benign**
        using a fine-tuned **XGBoost Classifier** trained on diagnostic cell data.
        """)
    with col2:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
        disp.plot(ax=ax, cmap="Purples", colorbar=False)
        plt.title("Confusion Matrix")
        st.pyplot(fig)

    # Confidence + Pie Chart
    st.markdown("### 📈 Model Confidence & Output Distribution")
    col3, col4 = st.columns(2)
    with col3:
        y_proba = model.predict_proba(X)[:, 1]
        fig, ax = plt.subplots()
        sns.histplot(y_proba, bins=15, kde=True, color="#9370db", ax=ax)
        plt.title("Confidence Levels (Malignant)")
        st.pyplot(fig)
    with col4:
        pred_counts = pd.Series(y_pred).value_counts()
        fig, ax = plt.subplots()
        ax.pie(pred_counts, labels=["Benign", "Malignant"], autopct='%1.1f%%',
               colors=["#b19cd9", "#9370db"], startangle=90)
        plt.title("Prediction Breakdown")
        st.pyplot(fig)

    st.markdown("<p style='text-align:center; color:gray; font-size:14px;'>Developed by <b>Aqsa Najeeb</b> | Powered by Streamlit + XGBoost</p>", unsafe_allow_html=True)

# TAB 2 — DATA INSIGHTS
with tab2:
    st.header("📊 Data Insights")
    st.dataframe(df.head(), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🎯 Target Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="diagnosis", data=df, palette="cool")
        plt.title("Benign vs Malignant Count")
        st.pyplot(fig)
    with c2:
        st.markdown("#### 🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), cmap="Purples", center=0)
        plt.title("Feature Correlation Matrix")
        st.pyplot(fig)

    st.markdown("### 🧠 Top Correlated Features")
    corr_target = df.corr()["diagnosis"].sort_values(ascending=False)[1:11]
    st.bar_chart(corr_target)

    st.markdown("### 🔑 Feature Importance (XGBoost)")
    if isinstance(model, XGBClassifier):
        fig, ax = plt.subplots(figsize=(8, 6))
        xgb.plot_importance(model, ax=ax, importance_type="weight", color="purple")
        plt.title("Top Important Features")
        st.pyplot(fig)

# TAB 3 — PREDICTION INTERFACE
with tab3:
    st.header("🤖 Model Prediction Interface")

    cols = st.columns(3)
    input_data = {}
    for i, feature in enumerate(X.columns):
        with cols[i % 3]:
            input_data[feature] = st.number_input(
                f"{feature}", value=float(np.round(df[feature].mean(), 2))
            )

    if st.button("🔍 Predict"):
        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]
        label = "Malignant" if pred == 1 else "Benign"

        st.markdown("---")
        if label == "Malignant":
            st.error("🚨 **Result: Malignant Tumor Detected**")
        else:
            st.success("✅ **Result: Benign Tumor Detected**")

        st.markdown("### 📋 Input Summary")
        st.dataframe(input_df.T, use_container_width=True)
