import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def load_data(path):
    """Load dataset from given path (CSV)."""
    return pd.read_csv(path)

def split_data(data, target_col, test_size=0.2, random_state=42):
    """Auto decide between random split and stratified split."""
    
    # Check if target column is categorical
    if data[target_col].dtype == 'object' or len(data[target_col].unique()) < 20:
        print("Using Stratified Split ✅")
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_idx, test_idx in split.split(data, data[target_col]):
            train_set = data.iloc[train_idx]
            test_set = data.iloc[test_idx]
    else:
        print("Using Random Split ✅")
        train_set, test_set = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
    
    return train_set, test_set

def save_splits(train_set, test_set, train_path, test_path):
    """Save train and test sets as CSV files."""
    train_set.to_csv(train_path, index=False)
    test_set.to_csv(test_path, index=False)
 # Elli navu auto decide of using stratified or random split use madtha edivi which is cool 


def load_split_data(train_path="datasets/strat_train_set.csv", test_path="datasets/strat_test_set.csv"):
    """Load previously saved train/test sets."""
    train_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)
    return train_set, test_set

