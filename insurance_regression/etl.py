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


def handle_null_and_transform(df):
    """
    Handles null values in the dataframe and applies transformations based on the specified rules.
    
    Args:
        df (pd.DataFrame): The input dataframe.
    
    Returns:
        pd.DataFrame: The transformed dataframe.
        dict: A dictionary containing label encoders for categorical columns.
    """
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
    le_marital_status = LabelEncoder()
    df['Marital Status'] = le_marital_status.fit_transform(df['Marital Status'])
    label_encoders['Marital Status'] = le_marital_status
    print(df.head())
    df['Number of Dependents'] = df['Number of Dependents'].fillna(-1)
    df['Number of Dependents'] = scaler.fit_transform(df[['Number of Dependents']])
    df['Occupation'] = df['Occupation'].fillna("unknown")
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
    le_feedback = LabelEncoder()
    df['Customer Feedback'] = le_feedback.fit_transform(df['Customer Feedback'])
    label_encoders['Customer Feedback'] = le_feedback
    categorical_columns = [
        'Customer Feedback', 'Gender', 'Education Level', 
        'Location', 'Policy Type', 'Smoking Status', 
        'Exercise Frequency', 'Property Type'
    ]
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])  # Overwrite the original column with encoded values
        label_encoders[col] = le  # Save the encoder for future use

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
                calendar = pd.read_csv(CALENDAR_PATH)
                inventory = pd.read_csv(INVENTORY_PATH)
                sales_test = pd.read_csv(TEST_PATH)
                print(f"Initial number of rows in sales_test: {len(sales_test)}")
                if not os.path.exists(solution_id_outpath):
                    sales_test['unique_id'] = sales_test['unique_id'].astype(str)
                    sales_test['date'] = sales_test['date'].astype(str)
                    solution_id = sales_test['unique_id'] + "_" + sales_test['date']
                    solution_id_df = pd.DataFrame({'solution_id': solution_id})
                    solution_id_df.to_csv(solution_id_outpath, index=False)
                else:
                    # Read solution_id back from CSV
                    solution_id_df = pd.read_csv(solution_id_outpath)
                    solution_id = solution_id_df['solution_id']
                    print(f"solution_id has been loaded from {solution_id_outpath}")

                #########

                try:
                    inventory['product_name'] = inventory['name'].str.split('_').str[0]
                    inventory['product_num'] = inventory['name'].str.split('_').str[1]

                    print("Splitting 'L4_category_name_en' into 'cat_name' and 'cat_num'")
                    inventory['cat_name'] = inventory['L4_category_name_en'].str.split('_L4_').str[0]
                    inventory['cat_num'] = inventory['L4_category_name_en'].str.split('_L4_').str[1]
                except Exception as e:
                    print("Checking for None or missing values in 'name' and 'L4_category_name_en'")
                    print(inventory[['name', 'L4_category_name_en']].isnull().sum())
                    # Debugging: Display a sample of rows with missing or irregular data
                    missing_data = inventory[inventory['name'].isnull() | inventory['L4_category_name_en'].isnull()]
                    if not missing_data.empty:
                        print("Rows with missing data in 'name' or 'L4_category_name_en':")
                        print(missing_data)
                    print("An error occurred during the splitting process.")
                    print(f"Error: {e}")
                    print("Displaying first few rows of inventory for debugging:")
                    print(inventory.head())

                print(f" number of rows in sales_test: {len(sales_test)}")
                inventory = inventory[['unique_id', 'warehouse', 'product_name', 'product_num', 'cat_name', 'cat_num']]
                sales_test = sales_test.merge(inventory, on=['unique_id', 'warehouse'], how='left')
               
                sales_test = sales_test.merge(calendar[['date', 'holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays','warehouse']],
                                                on=['date', 'warehouse'], how='left')
                print("merged calendar data")
                print(f"Initial number of rows in sales_test: {len(sales_test)}")
                ############ nan qa
                
                nan_summary = sales_test.isnull().sum()
                print("Missing values in each column:")
                print(nan_summary[nan_summary > 0])
                sales_test['date'] = pd.to_datetime(sales_test['date'])
                sales_test['day_of_week'] = sales_test['date'].dt.dayofweek
                sales_test['month'] = sales_test['date'].dt.month
                sales_test['year'] = sales_test['date'].dt.year
                
                #when used for test set# Load the saved label encoders
                with open( f"{LABEL_ENCODERS_PKL_OUTDIR}/lencoders.pkl", "rb") as f:
                    label_encoders = pickle.load(f)
                for col, le in label_encoders.items():
                    sales_test[col] = le.transform(sales_test[col])

                sales_test['product_num'] = pd.to_numeric(sales_test['product_num'], errors='coerce')
                sales_test['cat_num'] = pd.to_numeric(sales_test['product_num'], errors='coerce')
                sales_test.drop(columns=['date','unique_id'], inplace=True)
                print("updated processed sales_test")
                sales_test.to_pickle(PROCESSED_TEST_PATH)
                print(f"Initial number of rows in sales_test: {len(sales_test)}")
                print(f"DataFrame updated and saved as pickle file: {PROCESSED_TEST_PATH}")
                X_df = sales_test
            except FileNotFoundError:
                print(f"Error: The file '{TEST_PATH}' was not found.")
                return None
            except Exception as e:
                print(f"Error loading data: {e}")
                return None
        else:
            #load pickl
            X_df = pd.read_pickle(PROCESSED_TEST_PATH)
            solution_id_df = pd.read_csv(solution_id_outpath)
            solution_id = solution_id_df['solution_id']

        print("retreived sales_test")
        ###############
        print(X_df.info())
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

