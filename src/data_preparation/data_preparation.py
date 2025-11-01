
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

def separate_features_and_labels(data, target_col="median_house_value"):
    """
    Split dataset into features (X) and labels (y).
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset (train/test set).
    target_col : str
        The name of the column to predict (label).
    
    Returns
    -------
    X : pd.DataFrame
        Features (all columns except target).
    y : pd.Series
        Labels (only target column).
    """
    X = data.drop(target_col, axis=1)   # features
    y = data[target_col].copy()         # labels
    return X, y

def handle_missing_values(data):
    """
    Handle missing values in the dataset using SimpleImputer (median strategy).
    """
    # Select only numerical columns
    num_data = data.select_dtypes(include=[np.number])
    
    # Create and fit imputer
    imputer = SimpleImputer(strategy="median")
    imputer.fit(num_data)
    
    # Transform data and convert back to DataFrame
    transformed_data = pd.DataFrame(imputer.transform(num_data), columns=num_data.columns)
    
    return transformed_data, imputer

def encode_categorical_features(data):
    """One-hot encode categorical columns like 'ocean_proximity'."""
    cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    housing_cat = data[["ocean_proximity"]]
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    
    cat_columns = cat_encoder.get_feature_names_out(["ocean_proximity"])
    housing_cat_df = pd.DataFrame(housing_cat_1hot, columns=cat_columns, index=data.index)
    
    data = data.drop("ocean_proximity", axis=1)
    data = pd.concat([data, housing_cat_df], axis=1)
    return data


def scale_numeric_features(data):
    """Standardize numeric features (mean=0, std=1).""" #this is feature scaling by using stardization method 
    scaler = StandardScaler()                           #standard is used cuz it can handle outliers
    numeric_data = data.select_dtypes(include=[np.number])
    scaled_array = scaler.fit_transform(numeric_data)
    scaled_df = pd.DataFrame(scaled_array, columns=numeric_data.columns, index=data.index)
    
    # Replace old columns with scaled ones
    for col in numeric_data.columns:
        data[col] = scaled_df[col]
    
    return data, scaler 




def create_preprocessing_pipeline():
    """
    Create a complete preprocessing pipeline that handles:
    - Missing values
    - Scaling numeric features
    - Encoding categorical features
    """
    # Numeric pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

    # Define columns
    num_attribs = [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income"
    ]
    cat_attribs = ["ocean_proximity"]

    # Combine both using ColumnTransformer
    preprocessing = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return preprocessing


# ------------------------------------------------------------
# 4️⃣ Main function to prepare the dataset
# ------------------------------------------------------------
def prepare_data(housing_df):
    """
    Apply the preprocessing pipeline to the given housing DataFrame.
    Returns a fully processed (transformed) dataset.
    """
    preprocessing = create_preprocessing_pipeline()
    housing_prepared = preprocessing.fit_transform(housing_df)
    processed_df = pd.DataFrame(
        housing_prepared,
        columns=preprocessing.get_feature_names_out(),
        index=housing_df.index
    )
    return processed_df, preprocessing

preprocessing_pipeline = create_preprocessing_pipeline()
