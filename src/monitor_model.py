# src/monitor_model.py

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error

def monitor_model():
    print("🔹 Monitoring model performance...")
    model = joblib.load("artifacts/my_california_housing_model.pkl")

    # Load new (simulated) live data
    data = pd.read_csv("data/new_data.csv")  # Assume this is recent data
    X_new = data.drop("median_house_value", axis=1)
    y_new = data["median_house_value"]

    preds = model.predict(X_new)
    rmse = mean_squared_error(y_new, preds, squared=False)

    print(f"📊 Current RMSE: {rmse:.2f}")
    if rmse > 50000:
        print("⚠️ Warning: Performance dropped! Retraining needed.")
    else:
        print("✅ Model performance is healthy.")

if __name__ == "__main__":
    monitor_model()
