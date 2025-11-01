# src/model_analysis.py

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# 1️⃣ Load the trained model (GridSearchCV best model)
# ------------------------------------------------------------
MODEL_PATH = os.path.join("artifacts", "best_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("⚠️ Trained model not found! Please run model_training.py first.")

print("📦 Loading trained model...")
final_model = joblib.load(MODEL_PATH)

# ------------------------------------------------------------
# 2️⃣ Extract feature importances
# ------------------------------------------------------------
feature_importances = final_model["random_forest"].feature_importances_
feature_names = final_model["preprocessing"].get_feature_names_out()

# Combine into a DataFrame for easy analysis
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# ------------------------------------------------------------
# 3️⃣ Display Top Important Features
# ------------------------------------------------------------
print("\n🏆 Top 10 Most Important Features:")
print(importance_df.head(10))

# ------------------------------------------------------------
# 4️⃣ Visualize Feature Importance
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"][:15], importance_df["Importance"][:15])
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()

# Save plot
os.makedirs("artifacts", exist_ok=True)
plt.savefig(os.path.join("artifacts", "feature_importance.png"))
print("\n📊 Feature importance plot saved to 'artifacts/feature_importance.png'")
plt.show()

# ------------------------------------------------------------
# 5️⃣ Optional: Analyze Weak Features
# ------------------------------------------------------------
low_importance = importance_df.tail(10)
print("\n⚠️ Least Useful Features:")
print(low_importance)
