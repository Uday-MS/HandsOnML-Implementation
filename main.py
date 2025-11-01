# main.py
import os
import sys
import pandas as pd

from src.data_preparation.download_data import load_housing_data
from src.data_preparation.split_data import split_data, save_splits

# ------------------------------------------------------------
# 1️⃣ DOWNLOAD DATA
# ------------------------------------------------------------
def step_1_download_data():
    print("\n📥 STEP 1: Downloading dataset...")
    housing = load_housing_data()
    os.makedirs("datasets", exist_ok=True)
    housing.to_csv("datasets/housing.csv", index=False)
    print("✅ Dataset downloaded and saved to 'datasets/housing.csv'")
    return housing

# ------------------------------------------------------------
# 2️⃣ SPLIT DATA
# ------------------------------------------------------------
def step_2_split_data(housing):
    print("\n✂️ STEP 2: Splitting data into train and test sets...")
    train_set, test_set = split_data(housing, target_col="median_house_value")
    save_splits(train_set, test_set, "datasets/strat_train_set.csv", "datasets/strat_test_set.csv")
    print("✅ Data successfully split and saved!")
    return train_set, test_set

# ------------------------------------------------------------
# 3️⃣ TRAIN MODEL
# ------------------------------------------------------------
def step_3_train_model():
    print("\n🤖 STEP 3: Training the model...")
    # Import only when needed (after data is ready)
    from src import model_training
    os.system("python src/model_training.py")
    print("✅ Model training complete. Best model saved in 'artifacts/best_model.pkl'")

# ------------------------------------------------------------
# 4️⃣ ANALYZE MODEL
# ------------------------------------------------------------
def step_4_model_analysis():
    print("\n🔎 STEP 4: Analyzing trained model...")
    os.system("python src/model_analysis.py")
    print("✅ Model analysis complete. Feature importance saved in artifacts.")

# ------------------------------------------------------------
# 5️⃣ EVALUATE MODEL
# ------------------------------------------------------------
def step_5_evaluate_model():
    print("\n📊 STEP 5: Evaluating model performance on test set...")
    os.system("python src/evaluate_model.py")
    print("✅ Evaluation complete!")

# ------------------------------------------------------------
# 6️⃣ LAUNCH MODEL (SAVE FINAL VERSION)
# ------------------------------------------------------------
def step_6_launch_model():
    print("\n🚀 STEP 6: Launching model for deployment...")
    os.system("python src/launch_model.py")
    print("✅ Final model ready: 'artifacts/my_california_housing_model.pkl'")

# ------------------------------------------------------------
# MAIN RUNNER
# ------------------------------------------------------------
def main():
    print("\n🏗️ Starting California Housing ML Pipeline...")

    # Step 1: Download data
    housing = step_1_download_data()

    # Step 2: Split data
    step_2_split_data(housing)

    # Step 3: Train model (AFTER split files exist)
    step_3_train_model()

    # Step 4: Analyze model
    step_4_model_analysis()

    # Step 5: Evaluate model
    step_5_evaluate_model()

    # Step 6: Launch final production model
    step_6_launch_model()

    print("\n✅ ALL STEPS COMPLETED SUCCESSFULLY!")
    print("You can now run: python app.py  to serve the model via Flask API.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
