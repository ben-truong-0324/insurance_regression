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
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PowerTransformer
import joblib

def handle_null_and_transform_old(df,label_encoders=None):
    """
    Handles null values in the dataframe and applies transformations based on the specified rules.
    
    Args:
        df (pd.DataFrame): The input dataframe.
    
    Returns:
        pd.DataFrame: The transformed dataframe.
        dict: A dictionary containing label encoders for categorical columns.
    """
    print("herro")
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
    
    if 'Marital Status' in label_encoders:
        df['Marital Status'] = label_encoders['Marital Status'].transform(df['Marital Status'])
    else:
        le_marital_status = LabelEncoder()
        df['Marital Status'] = le_marital_status.fit_transform(df['Marital Status'])
        label_encoders['Marital Status'] = le_marital_status
    print("check2")

    if 'Number of Dependents' in label_encoders:
        df['Number of Dependents'] = label_encoders['Number of Dependents'].transform(df['Number of Dependents'])
    else:
        le_depend = LabelEncoder()
        df['Number of Dependents'] = le_depend.fit_transform(df['Number of Dependents'])
        label_encoders['Number of Dependents'] = le_depend
    print("check3")

    df['Occupation'] = df['Occupation'].fillna("unknown")
    print("check4")
    if 'Occupation' in label_encoders:
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
    if 'Customer Feedback' in label_encoders:
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


def create_derived_df(df):
    """Create a new dataframe with derived features."""
    derived_df = pd.DataFrame()
    derived_df['Age Group'] = pd.cut(df['Age'], bins=[0, 25, 40, 60, 100], labels=['18-25', '26-40', '41-60', '60+'])
    print("check")
    derived_df['Dependents to Location Index'] = (df['Number of Dependents'].astype(str) + '_' + df['Location']).str.lower()
    income_bins = [0, 30000, 60000, 100000,200000, np.inf]
    income_labels = ['Low', 'Middle', 'UpperMiddle', 'High','VeryHigh']
    print("check")
    derived_df['Income Band'] = pd.cut(df['Annual Income'], bins=income_bins, labels=income_labels)
    derived_df['Family Size'] = df['Number of Dependents'] + (df['Marital Status'] == 'Married').astype(int)
    print("check")
    risk_bins = pd.qcut(((df['Age'] / df['Health Score']) * (1 + df['Previous Claims'])), q=10, labels=False)
    derived_df['Risk Index'] = risk_bins
    print("check")
    today = datetime.today()
    insurance_duration = (today - pd.to_datetime(df['Policy Start Date'])).dt.days
    derived_df['Insurance Duration'] = pd.qcut(insurance_duration, q=10, labels=False)
    print("check")

    claim_freq = df['Previous Claims'] / (insurance_duration / 365.25)
    derived_df['Claim Frequency'] = pd.qcut(claim_freq, q=10, labels=False, duplicates='drop')
    print("check")
    derived_df['Property Risk'] = (df['Location'] + '_' + df['Property Type']).str.lower()
    df['Exercise Frequency'] = pd.to_numeric(df['Exercise Frequency'], errors='coerce').fillna(0)
    print("check")
    lifestyle_calc = (df['Health Score'] * (1 + df['Exercise Frequency'] - df['Smoking Status'].map({'Yes': 1, 'No': 0})) 
                      + df['Annual Income'] / 10000 
                      + derived_df['Family Size'])
    derived_df['Lifestyle Index'] = pd.qcut(lifestyle_calc, q=10, labels=False)
    print("check")
    derived_df['Vehicle Age Band'] = pd.cut(df['Vehicle Age'], bins=[0, 3, 10, 20], labels=['New', 'Moderate', 'Old'])
    derived_df['Credit Score Band'] = pd.cut(df['Credit Score'], bins=[300, 580, 670, 740, 800, 850], labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    return derived_df

def encode_and_normalize(derived_df):
    # Label Encoding for categorical features
    for column in derived_df.select_dtypes(include=['category', 'object']).columns:
        # encoder_path = os.path.join(LABEL_ENCODERS_PKL_OUTDIR, f"{column}_encoder.pkl")
        # if not os.path.exists(encoder_path):
        le = LabelEncoder()
        derived_df[column] = le.fit_transform(derived_df[column])
        #     joblib.dump(le, encoder_path)
        # else:
        #     le = joblib.load(encoder_path)
        #     derived_df[column] = le.transform(derived_df[column])

    # Normalization/Log transformation for numerical features
    for column in derived_df.select_dtypes(include=['float64', 'int64','int32']).columns:
        # nomalize_path = os.path.join(LABEL_ENCODERS_PKL_OUTDIR, f"{column}_transformer.pkl")
        # if not os.path.exists(nomalize_path):
        #     print(column)
        if (derived_df[column] > 0).all():
            transformer = PowerTransformer(method='yeo-johnson')
        else:
            transformer = MinMaxScaler()
        derived_df[column] = transformer.fit_transform(derived_df[[column]])
        #     joblib.dump(transformer, nomalize_path)
        # else:
        #     trans = joblib.load(nomalize_path)
        #     derived_df[column] = trans.transform(derived_df[column])
        
    return derived_df

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
    print("here")
    scaler = MinMaxScaler()  # For normalization
    df['Age'] = df['Age'].fillna(-1).astype(int)
    df['Age Group'] = pd.cut(
        df['Age'], bins=[-2,0, 12, 19, 35, 50, 65, np.inf],
        labels=[0, 1, 2, 3, 4, 5,6])
    print("here")
    df['Age Group'] = df['Age Group'].astype(int)
    print("here")
    df['Annual Income'] = df['Annual Income'].fillna(0)
    print("here")
    df['Annual Income'] = np.log10(df['Annual Income'] + 1)  # Adding 1 to avoid log(0)
    print("here1")
    df['Marital Status'] = df['Marital Status'].fillna("unknown")
    if 'Marital Status' in label_encoders:
        df['Marital Status'] = label_encoders['Marital Status'].transform(df['Marital Status'])
    else:
        le_marital_status = LabelEncoder()
        df['Marital Status'] = le_marital_status.fit_transform(df['Marital Status'])
        label_encoders['Marital Status'] = le_marital_status
    print("here2")
    if 'Number of Dependents' in label_encoders:
        df['Number of Dependents'] = label_encoders['Number of Dependents'].transform(df['Number of Dependents'])
    else:
        le_depend = LabelEncoder()
        df['Number of Dependents'] = le_depend.fit_transform(df['Number of Dependents'])
        label_encoders['Number of Dependents'] = le_depend
    print("here3")
    df['Occupation'] = df['Occupation'].fillna("unknown")
    if 'Occupation' in label_encoders:
        df['Occupation'] = label_encoders['Occupation'].transform(df['Occupation'])
    else:
        le_occupation = LabelEncoder()
        df['Occupation'] = le_occupation.fit_transform(df['Occupation'])
        label_encoders['Occupation'] = le_occupation
    print("here4")
    
    df['Health Score'] = df['Health Score'].fillna(-1)
    df['Health Score'] = scaler.fit_transform(df[['Health Score']])
    df['Previous Claims'] = df['Previous Claims'].fillna(-1)
    df['Vehicle Age'] = df['Vehicle Age'].fillna(-1)
    df['Credit Score'] = df['Credit Score'].fillna(-1)
    df['Insurance Duration'] = df['Insurance Duration'].fillna(-1)
    df['Customer Feedback'] = df['Customer Feedback'].fillna("unknown")
    if 'Customer Feedback' in label_encoders:
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

def do_ul_cluster(X_df):
    X_og = X_df.copy()
    ul_cluster_count = 7

    print("kmeans")
    kmeans = KMeans(n_clusters=ul_cluster_count, random_state=GT_ID)
    km_clus = kmeans.fit_predict(X_og)
    X_df['km_clus'] = km_clus

    print("gmm")
    gmm = GaussianMixture(n_components=ul_cluster_count, random_state=GT_ID)
    gmm_clus = gmm.fit_predict(X_og)
    X_df['gmm_clus'] = gmm_clus

    # print("dbscan")
    # dbscan = DBSCAN(eps=1.0, min_samples=3)
    # dbscan_clus = dbscan.fit_predict(X_og)
    # X_df['dbscan_clus'] = dbscan_clus
    
    # print("spec clus")
    # spectral = SpectralClustering(n_clusters=ul_cluster_count, affinity='nearest_neighbors', random_state=GT_ID)
    # spectral_clus = spectral.fit_predict(X_og)
    # X_df['spectral_clus'] = spectral_clus

    return X_df

def get_data():
    #step 1: extract data
    #step 2: transform, add derived features
    #step 3: load - feature selection metrics check, data format assert
    print(f"Getting data for {DATASET_SELECTION}")
    if "kaggle_insurance_regression" in DATASET_SELECTION:
        if not os.path.exists(PROCESSED_TRAIN_PATH):
            try:
                df = pd.read_csv(TRAIN_PATH)
                print("Accessed .csv in data folder")
                df = df.dropna()
                Y_df = df['Premium Amount']  # Target variable
                df = df.drop(columns=[ 'Premium Amount'])
                print("creating derived df")

                derived_df = create_derived_df(df)
                print("normalizing")
                derived_df = encode_and_normalize(derived_df)
                derived_df = do_ul_cluster(derived_df)
                print("encoding ul")
                derived_df = encode_and_normalize(derived_df)
                
                # df = convert_and_normalize_days_since_policy_start(df, 'Policy Start Date')
                # df, encoders = handle_null_and_transform(df)
                # encoder_path = f"{LABEL_ENCODERS_PKL_OUTDIR}/lencoders.pkl"
                # with open(encoder_path, "wb") as f:
                #     pickle.dump(encoders, f)
                # print(f"Label encoders saved at {encoder_path}")
                # df = do_ul_cluster(df)
                # columns_to_drop = ['id', 'Age']
                # df.drop(columns=columns_to_drop, inplace=True)
                # nan_summary = df.isnull().sum()
                # print("Missing values in each column:")
                # print(nan_summary[nan_summary > 0])
                # scaler = MinMaxScaler()
                # numeric_cols = df.select_dtypes(include=['float64', 'int64','int32']).columns
                # numeric_cols = numeric_cols[df.columns != 'Premium Amount']
                # df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

                derived_df['Premium Amount'] = Y_df
                print(derived_df.head())
                df = derived_df
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

                # df = convert_and_normalize_days_since_policy_start(df, 'Policy Start Date')
                # with open(f"{LABEL_ENCODERS_PKL_OUTDIR}/lencoders.pkl", "rb") as f:
                #     encoders = pickle.load(f)
                # print("got here")
                # df, encoders =  handle_null_and_transform(df,encoders)
                for column in df.columns:
                    df[column] = df[column].fillna(df[column].mode()[0])
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

                print("creating derived df")
                derived_df = create_derived_df(df)
                print("normalizing")
                derived_df = encode_and_normalize(derived_df)
                derived_df = do_ul_cluster(derived_df)
                print("encoding ul")
                derived_df = encode_and_normalize(derived_df)

                columns_to_drop = ['id', 'Age']
                solution_id = df['id'].copy()
                with open(solution_id_outpath, "wb") as f:
                    pickle.dump(solution_id, f)
                print(f"Solution IDs saved at {solution_id_outpath}")
                df.drop(columns=columns_to_drop, inplace=True)

                df = derived_df
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
            solution_id_df = pd.read_pickle(solution_id_outpath)
            
            solution_id = pd.DataFrame(solution_id_df.values, columns=['id'])
            print(solution_id.head())
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

