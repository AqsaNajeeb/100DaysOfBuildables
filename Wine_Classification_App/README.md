# 🍷 Wine Classification App

This task is part of my 100 Days of Buildables journey, where I build practical apps using Python and Streamlit. This app classifies wines into **3 classes** based on chemical composition features from the famous **UCI Wine dataset**.

---

## 🚀 Current Project – Wine Classification App
**🌐 Live Demo**

👉 View on Streamlit Cloud: *[Wine Classification App](https://100daysofbuildables-wineclassification.streamlit.app/)*

---

## 📂 Repository Structure
```
WineClassifierApp/
│
├── task_14.py                # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── Task_14.ipynb    # Jupyter Notebook with model training & analysis
```
---

## 🛠️ Getting Started

Follow these steps to run the project locally:

1. **Clone the repository**
```sh
git clone https://github.com/AqsaNajeeb/WineClassifierApp.git
cd WineClassifierApp
```
2. **Install dependencies**

It’s recommended to use a virtual environment:

```sh
pip install -r requirements.txt
```
3. **Run the app**

```sh
streamlit run task_14.py
```

The app will open in your browser at: [http://localhost:8501](http://localhost:8501)

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

* Compare multiple ML models on the Wine dataset
* Integrate model switching into a Streamlit interface
* Display interactive visualizations for better understanding
* Share open-source apps that anyone can use or extend 🔧

---

## 📚 My Reflection

What worked?

✅ The Streamlit app successfully switches between Logistic Regression, Decision Tree, and KNN using a sidebar option.

✅ The app displays accuracy, F1 score, confusion matrix, and user-friendly wine class predictions.

✅ Integration of sliders and probability outputs made the app interactive and professional.

What didn’t work?

⚠️ Logistic Regression sometimes gave perfect scores due to the dataset being highly separable. This was verified with multiclass F1 averaging.

⚠️ Some features (like Malic_acid and Proline) showed skewness, which could affect KNN and Logistic Regression, though Decision Tree remained robust.

Which visualization was most insightful?

📊 The confusion matrix heatmap made it clear which wine classes were hardest to separate.

🔥 The probability bar chart for predictions gave an intuitive understanding of model confidence.

---

## 🤝 Connect With Me

If you’d like to collaborate, discuss ideas, or share feedback, feel free to reach out:

* GitHub: [Aqsa Najeeb](https://github.com/AqsaNajeeb)
* LinkedIn: [Aqsa Najeeb](https://www.linkedin.com/in/aqsa-najeeb/)

---

✨ If you like this project, don’t forget to **star ⭐ the repo!**
