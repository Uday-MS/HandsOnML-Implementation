# src/evaluate_model.py

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
from src.data_preparation.split_data import load_split_data

# ------------------------------------------------------------
# 1️⃣ Load the trained final model
# ------------------------------------------------------------
print("🔹 Loading the trained final model...")
final_model = joblib.load("artifacts/best_model.pkl")

# ------------------------------------------------------------
# 2️⃣ Load the test set
# ------------------------------------------------------------
print("🔹 Loading test data...")
_, test_set = load_split_data()

X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

# ------------------------------------------------------------
# 3️⃣ Run predictions
# ------------------------------------------------------------
print("🔹 Making predictions on test data...")
final_predictions = final_model.predict(X_test)

# ------------------------------------------------------------
# 4️⃣ Evaluate performance using RMSE
# ------------------------------------------------------------
final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
print(f"\n✅ Final RMSE on Test Set: {final_rmse:.2f}")

# ------------------------------------------------------------
# 5️⃣ Compute 95% confidence interval
# ------------------------------------------------------------
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
confidence_interval = np.sqrt(
    stats.t.interval(
        confidence,
        len(squared_errors) - 1,
        loc=squared_errors.mean(),
        scale=stats.sem(squared_errors)
    )
)

print(f"📊 95% confidence interval for the RMSE: {confidence_interval}")

# ------------------------------------------------------------
# 6️⃣ Summary
# ------------------------------------------------------------
print("\n📘 SUMMARY:")
print(f"• Test RMSE: {final_rmse:.2f}")
print(f"• 95% Confidence Interval: [{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]")
print("\n🎯 Model evaluation completed successfully!")
