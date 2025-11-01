# src/model_training.py

from src.data_preparation.data_preparation import preprocessing_pipeline
from src.data_preparation.split_data import load_split_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
import joblib
import os



# ------------------------------------------------------------
# 1️⃣ Load Data
# ------------------------------------------------------------
train_set, test_set = load_split_data()

# Separate labels
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

# ------------------------------------------------------------
# 2️⃣ Build full pipeline with RandomForestRegressor
# ------------------------------------------------------------
full_pipeline = Pipeline([
    ("preprocessing", preprocessing_pipeline),
    ("random_forest", RandomForestRegressor(random_state=42)),
])

# ------------------------------------------------------------
# 3️⃣ Define parameter distributions for Randomized Search
# ------------------------------------------------------------
param_distribs = {
    # You can adjust these according to your project features
    "random_forest__n_estimators": randint(low=50, high=200),   # number of trees
    "random_forest__max_features": randint(low=2, high=8),      # number of features per split
    "random_forest__max_depth": randint(low=5, high=30)         # tree depth
}

# ------------------------------------------------------------
# 4️⃣ Run Randomized Search
# ------------------------------------------------------------
rnd_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions=param_distribs,
    n_iter=10,  # number of random combinations to try
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
    verbose=2
)

print("🎲 Running Randomized Search... Please wait ⏳")
rnd_search.fit(housing, housing_labels)

# ------------------------------------------------------------
# 5️⃣ Display best parameters and model
# ------------------------------------------------------------
print("\n✅ Best Parameters Found:")
print(rnd_search.best_params_)

best_model = rnd_search.best_estimator_
print("\n🏆 Best Model:")
print(best_model)

# ------------------------------------------------------------
# 6️⃣ Evaluate Randomized Search results
# ------------------------------------------------------------
cv_results = pd.DataFrame(rnd_search.cv_results_)
cv_results = cv_results.sort_values(by="mean_test_score", ascending=False)

print("\n📊 Top 5 Randomized Search Results:")
print(cv_results[["param_random_forest__n_estimators",
                  "param_random_forest__max_features",
                  "param_random_forest__max_depth",
                  "mean_test_score"]].head())

best_rmse = -rnd_search.best_score_
print(f"\n Best RMSE (cross-validation): {best_rmse:.2f}")


# 7️ Evaluate on test set

X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

final_predictions = best_model.predict(X_test)

final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
print(f"\n📏 Final RMSE on Test Set: {final_rmse:.2f}")


# 8️ Save the best model for future use

joblib.dump(best_model, "best_random_forest_model.pkl")
print("\n Model saved as 'best_random_forest_model.pkl'")

# ------------------------------------------------------------
# 8️⃣ Save the best model for future analysis
# ------------------------------------------------------------

os.makedirs("artifacts", exist_ok=True)
MODEL_PATH = os.path.join("artifacts", "best_model.pkl")

joblib.dump(best_model, MODEL_PATH)
print(f"\n💾 Model saved successfully at: {MODEL_PATH}")
