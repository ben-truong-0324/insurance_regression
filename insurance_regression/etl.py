import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import time
from insurance_regression.config import *
import insurance_regression.plots as plots

# from config import *
# import insurance_regression.data_plots
import pickle
 
import pandas as pd
from datetime import datetime
import numpy as np


def handle_null_and_transform(df,label_encoders=None):
    """
    Handles null values in the dataframe and applies transformations based on the specified rules.
    
    Args:
        df (pd.DataFrame): The input dataframe.
    
    Returns:
        pd.DataFrame: The transformed dataframe.
        dict: A dictionary containing label encoders for categorical columns.
    """
    if label_encoders is None:
        label_encoders = {}
    scaler = MinMaxScaler()  # For normalization
    df['Age'] = df['Age'].fillna(0).astype(int)
    df['Age Group'] = pd.cut(
        df['Age'], bins=[-1, 12, 19, 35, 50, 65, np.inf],
        labels=[0, 1, 2, 3, 4, 5])
    df['Age Group'] = df['Age Group'].astype(int)
    df['Annual Income'] = df['Annual Income'].fillna(0)
    df['Annual Income'] = np.log10(df['Annual Income'] + 1)  # Adding 1 to avoid log(0)
    df['Marital Status'] = df['Marital Status'].fillna("unknown")

    if label_encoders and 'Marital Status' in label_encoders:
        df['Marital Status'] = label_encoders['Marital Status'].transform(df['Marital Status'])
    else:
        le_marital_status = LabelEncoder()
        df['Marital Status'] = le_marital_status.fit_transform(df['Marital Status'])
        label_encoders['Marital Status'] = le_marital_status

    le_marital_status = LabelEncoder()
    df['Marital Status'] = le_marital_status.fit_transform(df['Marital Status'])
    label_encoders['Marital Status'] = le_marital_status
    df['Number of Dependents'] = df['Number of Dependents'].fillna(-1)
    df['Number of Dependents'] = scaler.fit_transform(df[['Number of Dependents']])
    df['Occupation'] = df['Occupation'].fillna("unknown")
    if label_encoders and 'Occupation' in label_encoders:
        df['Occupation'] = label_encoders['Occupation'].transform(df['Occupation'])
    else:
        le_occupation = LabelEncoder()
        df['Occupation'] = le_occupation.fit_transform(df['Occupation'])
        label_encoders['Occupation'] = le_occupation

    
    df['Health Score'] = df['Health Score'].fillna(-1)
    df['Health Score'] = scaler.fit_transform(df[['Health Score']])
    df['Previous Claims'] = df['Previous Claims'].fillna(0)
    df['Vehicle Age'] = df['Vehicle Age'].fillna(0)
    df['Credit Score'] = df['Credit Score'].fillna(df['Credit Score'].mean())
    df['Insurance Duration'] = df['Insurance Duration'].fillna(0)
    df['Customer Feedback'] = df['Customer Feedback'].fillna("unknown")
    if label_encoders and 'Customer Feedback' in label_encoders:
        df['Customer Feedback'] = label_encoders['Customer Feedback'].transform(df['Customer Feedback'])
    else:
        le_feedback = LabelEncoder()
        df['Customer Feedback'] = le_feedback.fit_transform(df['Customer Feedback'])
        label_encoders['Customer Feedback'] = le_feedback

    categorical_columns = [
       'Gender', 'Education Level', 
        'Location', 'Policy Type', 'Smoking Status', 
        'Exercise Frequency', 'Property Type'
    ]
    for col in categorical_columns:
        if label_encoders and col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])  # Overwrite the original column with encoded values
            label_encoders[col] = le   # Save the encoder for future use

    return df, label_encoders

def convert_and_normalize_days_since_policy_start(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    today = datetime.today()
    df['days_since_policy_start'] = (today - df[date_column]).dt.days
    df['days_since_policy_start'] = df['days_since_policy_start'].apply(lambda x: max(x, 1) if pd.notnull(x) else np.nan)
    df['log_days_since_policy_start'] = np.log10(df['days_since_policy_start'])
    df.drop(columns=[date_column], inplace=True)
    return df

def get_categorical_columns(df, unique_threshold=30):
    categorical_columns = []
    for col in df.columns:
        unique_values = df[col].nunique()
        if df[col].dtype == 'object' or unique_values <= unique_threshold:
            categorical_columns.append(col)
    print(f"Identified categorical columns: {categorical_columns}")
    return categorical_columns

def encode_and_save_labels(df, categorical_columns):
    label_encoders = {}
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))  # Ensure all values are strings
            label_encoders[col] = le
    encoder_path = f"{LABEL_ENCODERS_PKL_OUTDIR}/lencoders.pkl"
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoders, f)
    print(f"Label encoders saved at {encoder_path}")
    return df

def get_data():
    print(f"Getting data for {DATASET_SELECTION}")
    if "kaggle_insurance_regression" in DATASET_SELECTION:
        if not os.path.exists(PROCESSED_TRAIN_PATH):
            try:
                df = pd.read_csv(TRAIN_PATH)
                print("Accessed .csv in data folder")
                df = convert_and_normalize_days_since_policy_start(df, 'Policy Start Date')
                df, encoders = handle_null_and_transform(df)
                encoder_path = f"{LABEL_ENCODERS_PKL_OUTDIR}/lencoders.pkl"
                with open(encoder_path, "wb") as f:
                    pickle.dump(encoders, f)
                print(f"Label encoders saved at {encoder_path}")
                columns_to_drop = ['id', 'Age']
                df.drop(columns=columns_to_drop, inplace=True)
                nan_summary = df.isnull().sum()
                print("Missing values in each column:")
                print(nan_summary[nan_summary > 0])
                df.to_pickle(PROCESSED_TRAIN_PATH)
                print(f"DataFrame updated and saved as pickle file: {PROCESSED_TRAIN_PATH}")

            except FileNotFoundError:
                print(f"Error: The file '{TRAIN_PATH}' was not found.")
                return None
            except Exception as e:
                print(f"Error loading data: {e}")
                return None
        else:
            #load pickl
            df = pd.read_pickle(PROCESSED_TRAIN_PATH)
        print(df.head())
        print(f"{'Column':<20} {'Type':<20} {'# Unique':<10} {'Top 3 Values (Freq)':<80} {'# Null':<10} ")
        print("=" * 120)
        for col in df.columns:
            col_type = str(df[col].dtype)  # Convert dtype to string to avoid formatting issues
            unique_count = df[col].nunique() if "float" not in col_type and "int" not in col_type else "Numerical"
            top_values = df[col].value_counts().head(3)
            top_values_str = ", ".join([f"{val} ({cnt})" for val, cnt in top_values.items()])
            null_count = df[col].isnull().sum()
            print(f"{col:<20} {col_type:<20} {unique_count:<10} {top_values_str:<80} {null_count:<10}")
        print("=" * 120)
        Y_df = df['Premium Amount']  # Target variable
        X_df = df.drop(columns=[ 'Premium Amount'])
    else: 
        print("#"*18)
        raise ValueError("Invalid dataset specified. Check config.py")

    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)  # Convert to DataFrame
    if Y_df.ndim == 1:
        # If it's 1D, convert to Pandas Series
        Y_df = pd.Series(Y_df)
    else:
        # If it's 2D, convert to Pandas DataFrame
        Y_df = pd.DataFrame(Y_df)
    return X_df, Y_df

def graph_raw_data(X_df, Y_df):
    raw_data_outpath =OUTPUT_DIR_RAW_DATA_A3
    
    # Check if Y_df is multi-label (2D) or single-label (1D)
    if Y_df.ndim == 1:  # Single-label
        if not os.path.exists(f'{raw_data_outpath}/feature_heatmap.png'):
            if X_df.shape[0] > 1000:
                random_subset = X_df.sample(n=1000, random_state=42).index
                X_df = X_df.loc[random_subset]
                Y_df = Y_df.loc[random_subset]
            

            # Plot class imbalance, feature violin, heatmap, etc.
            plots.graph_class_imbalance(Y_df, 
                                             f'{raw_data_outpath}/class_imbalance.png')
            plots.graph_feature_violin(X_df, Y_df, 
                                             f'{raw_data_outpath}/feature_violin.png')
            plots.graph_feature_heatmap(X_df, Y_df,
                                             f'{raw_data_outpath}/feature_heatmap.png')
            plots.graph_feature_histogram(X_df, 
                                             f'{raw_data_outpath}/feature_histogram.png')
            plots.graph_feature_correlation(X_df, Y_df,
                                             f'{raw_data_outpath}/feature_correlation.png')
            plots.graph_feature_cdf(X_df, 
                                             f'{raw_data_outpath}/feature_cdf.png')
    else:  # Multi-label
        if not os.path.exists(f'{raw_data_outpath}/feature_heatmap.png'):
            # Handle multi-label plotting differently if necessary
            pass




def get_test_data():
    print(f"Getting test data for {DATASET_SELECTION}")
    if "kaggle_insurance_regression" in DATASET_SELECTION:
        solution_id_outpath = os.path.join(os.getcwd(), 'data', 'solution_id.csv')

        if not os.path.exists(PROCESSED_TEST_PATH):
            try:
                df = pd.read_csv(TEST_PATH)
                print("Accessed .csv in data folder")
                df = convert_and_normalize_days_since_policy_start(df, 'Policy Start Date')


                with open(f"{LABEL_ENCODERS_PKL_OUTDIR}/lencoders.pkl", "rb") as f:
                    encoders = pickle.load(f)
                df, encoders =  handle_null_and_transform(df,encoders)

                columns_to_drop = ['id', 'Age']
                solution_id = df['id'].copy()
                with open(solution_id_outpath, "wb") as f:
                    pickle.dump(solution_id, f)
                print(f"Solution IDs saved at {solution_id_outpath}")

                df.drop(columns=columns_to_drop, inplace=True)
                nan_summary = df.isnull().sum()
                print("Missing values in each column:")
                print(nan_summary[nan_summary > 0])
                df.to_pickle(PROCESSED_TEST_PATH)
                print(f"DataFrame updated and saved as pickle file: {PROCESSED_TEST_PATH}")
                X_df = df
            except FileNotFoundError:
                print(f"Error: The file '{TEST_PATH}' was not found.")
                return None
            except Exception as e:
                print(f"Error loading data: {e}")
                return None
        else:
            X_df = pd.read_pickle(PROCESSED_TEST_PATH)
            solution_id_df = pd.read_csv(solution_id_outpath)
            solution_id = solution_id_df['solution_id']
        print(X_df.info())
        print(solution_id.info())
    else: 
        print("#"*18)
        raise ValueError("Invalid dataset specified. Check config.py")
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)  # Convert to DataFrame

   
    return X_df, solution_id

import pandas as pd



import pandas as pd

def prelim_view():
    try:
        # Load the CSV file
        df = pd.read_csv(TRAIN_PATH)

        # Print general dataset info
        print("Dataset Overview:")
        print("#" * 50)
        print(f"Shape: {df.shape}")
        print(f"Total Missing Values: {df.isnull().sum().sum()}")
        print("#" * 50)

        # Prepare and print table-like summary for each column
        print(f"{'Column':<20} {'Type':<20} {'# Unique':<10} {'Top 3 Values (Freq)':<40} {'# Null':<10} ")
        print("=" * 120)

        for col in df.columns:
            col_type = str(df[col].dtype)  # Convert dtype to string to avoid formatting issues
            unique_count = df[col].nunique() if "float" not in col_type and "int" not in col_type else "Numerical"
            top_values = df[col].value_counts().head(3)
            top_values_str = ", ".join([f"{val} ({cnt})" for val, cnt in top_values.items()])
            null_count = df[col].isnull().sum()

            print(f"{col:<20} {col_type:<20} {unique_count:<10} {top_values_str:<40} {null_count:<10}")

        print("=" * 120)
        print("Note: Columns with numerical data are marked as 'Numerical' in the '# Unique' field.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

