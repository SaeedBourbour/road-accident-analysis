import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def preprocess_data(data):
    columns_to_drop = ['status', 'accident_index', 'accident_year', 'accident_reference', 'lsoa_of_casualty']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    # Ensure 'casualty_severity' column exists
    if 'casualty_severity' not in data.columns:
        raise KeyError("Column 'casualty_severity' not found in DataFrame.")

    # Replace -1 values with NaN
    data.replace(-1, np.nan, inplace=True)

    # Convert categorical columns to dummy variables, excluding 'casualty_severity'
    categorical_columns = [col for col in data.columns if col != 'casualty_severity' and data[col].dtype == 'object']
    data = pd.get_dummies(data, columns=categorical_columns)

    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data

def preprocess_data_with_labels(data):
    data = preprocess_data(data)

    # Label encoding for categorical variables
    data['casualty_class'] = data['casualty_class'].map({1: 'Driver', 2: 'Pedestrian', 3: 'Passenger'})
    data['sex_of_casualty'] = data['sex_of_casualty'].map({1: 'Male', 2: 'Female'})
    data['age_of_casualty'] = data['age_of_casualty'].apply(age_group)

    return data

def age_group(age):
    if age <= 16:
        return 'Children'
    elif age <= 30:
        return 'Young Adults'
    elif age <= 45:
        return 'Middle-aged Adults'
    else:
        return 'Old Adults'
