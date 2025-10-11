# 🍷 Wine Classification – Streamlit App

This task is part of my **100 Days of Buildables** journey, where I build practical and interactive ML apps using **Python** and **Streamlit**.  
This app classifies wines into **3 classes** based on their chemical composition using ensemble learning techniques — **Decision Tree (baseline)**, **Random Forest**, and **XGBoost** — trained on the **UCI Wine dataset**.

---

## 🚀 Current Task – Ensemble Wine Classification App

**🌐 Live Demo**  
👉 View on Streamlit Cloud: *([Wine Class Predictor](https://100daysofbuildables-task15.streamlit.app/))*

---

## 📂 Repository Structure

```

WineEnsembleApp/
│
├── task15.py                      # Streamlit app with live model training and prediction
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── best_model_.pkl                # Saved the best trained model
└── Task_15.ipynb                  # Jupyter Notebook for model comparison & analysis

````

---

## 🛠️ Getting Started

Follow these steps to run the project locally:

### 1. **Clone the repository**
```sh
git clone https://github.com/AqsaNajeeb/100DaysOfBuildables/Tasks_Solutions/Task_15.git
cd Task_15
````

### 2. **Install dependencies**

It’s recommended to use a virtual environment:

```sh
pip install -r requirements.txt
```

### 3. **Run the app**

```sh
streamlit run task15.py
```

The app will open in your browser at: [http://localhost:8501](http://localhost:8501)

---

## ✅ Requirements

* Python 3.9+
* Streamlit
* NumPy
* Pandas
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn

All dependencies are listed in `requirements.txt`.

---

## 🌟 About This Task

This task demonstrates the **power of ensemble learning** in improving model performance.
It compares **Decision Tree**, **Random Forest**, and **XGBoost** classifiers on the **Wine dataset**, evaluating them using metrics like **Accuracy**, **Precision**, **Recall**, and **F1-score**.

**Key Highlights:**

* ✅ Ensemble model comparison (Decision Tree, Random Forest, XGBoost)
* ✅ Real-time predictions via Streamlit interface
* ✅ Auto-selection of best-performing model
* ✅ Scaled input preprocessing for consistent predictions
* ✅ Clean, professional UI with labeled feature inputs

---

## 📊 Model Evaluation Summary

| Model                    | Accuracy | Precision | Recall | F1-Score |
| ------------------------ | -------- | --------- | ------ | -------- |
| Decision Tree (Baseline) | 0.94     | 0.95      | 0.94   | 0.94     |
| Random Forest            | 1.00     | 1.00      | 1.00   | 1.00     |
| XGBoost                  | 1.00     | 1.00      | 1.00   | 1.00     |

Both ensemble models achieved perfect accuracy without overfitting, validated via stratified splits and controlled parameters.

---

## 📚 My Reflection

**What worked?**

✅ Ensemble models clearly outperformed the baseline Decision Tree.

✅ Streamlit made real-time predictions intuitive and interactive.

✅ The use of `StandardScaler` helped stabilize model training and improve generalization.

**What didn’t work?**

⚠️ XGBoost can be slower on small datasets due to higher model complexity.

⚠️ Some input features (like Malic Acid and Proline) have large value ranges requiring careful normalization.

**Most Insightful Visualization?**

📊 The feature importance plots showed that **Flavanoids**, **Color Intensity**, and **Proline** were the strongest predictors of wine class.

🔥 Seeing how both Random Forest and XGBoost agreed on top features made model interpretation more reliable.

---

## 🤝 Connect With Me

If you’d like to collaborate, discuss ideas, or share feedback, feel free to reach out:

* GitHub: [Aqsa Najeeb](https://github.com/AqsaNajeeb)
* LinkedIn: [Aqsa Najeeb](https://www.linkedin.com/in/aqsa-najeeb/)

---

✨ If you like this project, don’t forget to **star ⭐ the repo!**
