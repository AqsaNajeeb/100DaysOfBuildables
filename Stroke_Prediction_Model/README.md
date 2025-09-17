# 🎯 Stroke Prediction – Streamlit App

This project is part of my **100 Days of Buildables** journey, where I build practical apps using Python and Streamlit. This app predicts the **risk of stroke** based on patient health information.

---

## 🚀 Current Project – Stroke Prediction App
🌐 **Live Demo**  
👉 View on Streamlit Cloud: *(Add your live demo link here)*

---

## 📂 Repository Structure
```

StrokePredictionApp/
│
├── stroke_prediction_model.py              # Main Streamlit application
├── model.pkl           # Trained ML model
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── healthcare-dataset-stroke-data.csv           # Dataset used
└── Stroke_Pridiction.ipynb         # Jupyter Notebook


````

---

## 🛠️ Getting Started

Follow these steps to run the project locally:

1. **Clone the repository**
```sh
git clone https://github.com/AqsaNajeeb/StrokePredictionApp.git
cd StrokePredictionApp

2. **Install dependencies**

It's recommended to use a virtual environment:

```sh
pip install -r requirements.txt

3. **Run the app**

```sh
streamlit run stroke_prediction_model.py

The app will open in your browser at: [http://localhost:8501](http://localhost:8501)

---

## ✅ Requirements

* Python 3.9+
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Plotly
* Joblib

All dependencies are listed in `requirements.txt`.

---

## 🌟 About This Project

This project is part of my **100 Days of Buildables** journey to:

* Build, deploy, and share practical apps 💻
* Strengthen my coding, design, and deployment skills 🚀
* Share open-source apps that anyone can use or extend 🔧

---

## 📚 My Reflection

What worked?

✅ The Streamlit app successfully takes user inputs and predicts stroke risk using the trained ML model.

✅ The preprocessing pipeline ensures inputs match the model features exactly, preventing prediction errors.

✅ The integration of interactive visualizations (Plotly bar chart & Seaborn heatmap) enhanced user understanding of the data trends.

✅ The app runs smoothly locally and can be deployed on Streamlit Cloud with minimal setup.

What didn’t work?

⚠️ Initially, dummy data was used for visualizations, which didn’t reflect real insights. This has now been replaced with actual dataset samples.

⚠️ Handling rare categories like “Unknown” in smoking status required explicit preprocessing to match the training features.

⚠️ Real-time EDA on full dataset is limited due to performance concerns.

Which visualization was most insightful?

📊 The Stroke Rate by Smoking Status bar chart using Plotly provided the clearest insight into how different smoking habits correlate with stroke risk. The smoking status vs stroke is clearly displayed using bar chart in the Jupyter Notebook.

🔥 The correlation heatmap also helped identify relationships between numerical features like age, BMI, glucose levels, and stroke occurrence, which is critical for understanding model behavior and feature importance.

---

## 🤝 Connect With Me

If you’d like to collaborate, discuss ideas, or share feedback, feel free to reach out:

* GitHub: [Aqsa Najeeb](https://github.com/AqsaNajeeb)
* LinkedIn: [Aqsa Najeeb](https://www.linkedin.com/in/aqsa-najeeb/)

---

✨ If you like this project, don’t forget to **star ⭐ the repo!**

```

