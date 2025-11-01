# src/launch_model.py

import joblib
import numpy as np
import pandas as pd
from src.data_preparation.split_data import load_split_data

# ------------------------------------------------------------
# 1️⃣ Load your best model (from model_training step)
# ------------------------------------------------------------
print("🔹 Loading best model from artifacts...")
best_model = joblib.load("artifacts/best_model.pkl")

# ------------------------------------------------------------
# 2️⃣ Save it as a final production model
# ------------------------------------------------------------
print("💾 Saving final model for deployment...")
joblib.dump(best_model, "artifacts/my_california_housing_model.pkl")
print("✅ Model saved as my_california_housing_model.pkl")

# ------------------------------------------------------------
# 3️⃣ Example: Load it back in production
# ------------------------------------------------------------
print("\n🚀 Simulating production environment...")
final_model_reloaded = joblib.load("artifacts/my_california_housing_model.pkl")
print("✅ Model successfully reloaded.")

# ------------------------------------------------------------
# 4️⃣ Use the model to make predictions on new data
# ------------------------------------------------------------
_, test_set = load_split_data()
sample_data = test_set.drop("median_house_value", axis=1).iloc[:5]
predictions = final_model_reloaded.predict(sample_data)

print("\n🔍 Predictions for sample data:")
print(predictions)
