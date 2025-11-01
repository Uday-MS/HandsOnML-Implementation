# 🧠 Hands-On ML Implementation: End-to-End Machine Learning Project

This project is inspired by **Chapter 2** of *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (O’Reilly).  
It focuses on building an **end-to-end ML pipeline** — from **data collection** to **model deployment** — using real-world housing data.

---

## 🚀 Project Highlights

- 📊 **Data Exploration & Visualization**
- 🧹 **Data Cleaning & Preprocessing** — handling missing values, categorical encoding, and feature scaling  
- 🤖 **Model Building** — trained and compared Linear Regression, Decision Tree, and Random Forest  
- 🔍 **Hyperparameter Tuning** — Grid Search & Cross-Validation for model optimization  
- 🧾 **Model Evaluation** — compared RMSE and R² scores  
- 💾 **Model Saving** — persisted best model using `joblib`  
- 🌐 **Flask Integration** — built a simple web interface to serve predictions  

---

## 🧩 Project Structure
HandsOnML-Implementation/
│
├── data/ # Dataset files
├── notebooks/ # Jupyter Notebooks for exploration
├── models/ # Saved ML models (.pkl/.joblib)
├── static/ # CSS and static files for Flask app
├── templates/ # HTML templates for Flask app
├── app.py # Flask application file
├── model_training.py # ML pipeline script
├── requirements.txt # All dependencies
└── README.md # Project documentation

---

## ⚙️ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Uday-MS/HandsOnML-Implementation.git
   cd HandsOnML-Implementation
2.Create & activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate   # On Mac/Linux

3.Install dependencies
pip install -r requirements.txt

4.Run the Flask app
python app.py

5.Open your browser and go to
👉 http://127.0.0.1:5000/

-->Tech Stack
   *Python
  *Scikit-Learn
  *Pandas, NumPy, Matplotlib
 *Flask
  *HTML, CSS (Frontend UI)

-->Reference
Book: Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow
Author: Aurélien Géron


  
