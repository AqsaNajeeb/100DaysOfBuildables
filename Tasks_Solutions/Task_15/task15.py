import streamlit as st
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42
)
xgb.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

if xgb_acc > rf_acc:
    best_model = xgb
    best_name = "XGBoost Classifier"
else:
    best_model = rf
    best_name = "Random Forest Classifier"




# ---------------------------
# 🎨 Streamlit UI
st.set_page_config(page_title="Wine Class Predictor", page_icon="🍷")
st.title("🍷 Wine Class Prediction App")
st.markdown(
    f"### Using Best Performing Ensemble Model: **{best_name}**"
)
st.write(
    "This app predicts the type of wine based on its chemical composition. "
    "Enter the values below and click **Predict Wine Type** to see the result."
)

st.markdown("---")
st.header("🔢 Enter Wine Characteristics:")

col1, col2 = st.columns(2)

with col1:
    alcohol = st.number_input("Alcohol", 11.0, 15.0, 13.0)
    malic_acid = st.number_input("Malic acid", 0.7, 5.8, 2.3)
    ash = st.number_input("Ash", 1.3, 3.2, 2.4)
    alcalinity = st.number_input("Alcalinity of ash", 10.0, 30.0, 17.0)
    magnesium = st.number_input("Magnesium", 70.0, 160.0, 100.0)
    total_phenols = st.number_input("Total phenols", 0.9, 3.9, 2.3, step=0.1)
    flavanoids = st.number_input("Flavanoids", 0.3, 5.0, 2.0, step=0.1)

with col2:
    nonflavanoid_phenols = st.number_input("Nonflavanoid phenols", 0.1, 0.7, 0.3)
    proanthocyanins = st.number_input("Proanthocyanins", 0.4, 3.6, 1.5)
    color_intensity = st.number_input("Color intensity", 1.0, 13.0, 5.0)
    hue = st.number_input("Hue", 0.4, 1.7, 1.0)
    od280_od315 = st.number_input("OD280/OD315 of diluted wines", 1.2, 4.0, 2.8)
    proline = st.number_input("Proline", 278.0, 1680.0, 750.0)


if st.button("🎯 Predict Wine Type"):
    # Prepare input for prediction
    input_data = np.array(
        [[
            alcohol, malic_acid, ash, alcalinity, magnesium,
            total_phenols, flavanoids, nonflavanoid_phenols,
            proanthocyanins, color_intensity, hue,
            od280_od315, proline
        ]]
    )

    scaled_input = scaler.transform(input_data)

    # Predict class
    prediction = best_model.predict(scaled_input)[0]
    prediction_proba = (
        best_model.predict_proba(scaled_input)[0]
        if hasattr(best_model, "predict_proba")
        else [0, 0, 0]
    )

    label_map = {0: "Class 1", 1: "Class 2", 2: "Class 3"}
    predicted_label = label_map.get(prediction, "Unknown")

    st.markdown("---")
    st.success(f"🏷️ **Predicted Wine Type:** {predicted_label}")
    st.markdown("### 📊 Prediction Probabilities:")
    st.write(f"Class 1: {prediction_proba[0]:.3f}")
    st.write(f"Class 2: {prediction_proba[1]:.3f}")
    st.write(f"Class 3: {prediction_proba[2]:.3f}")

st.markdown("---")
st.markdown("Developed by **Aqsa Najeeb**")

