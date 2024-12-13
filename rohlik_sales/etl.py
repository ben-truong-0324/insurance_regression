import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import time
from rohlik_sales.config import *
import rohlik_sales.plots as plots

# from config import *
# import rohlik_sales.data_plots
import pickle



def get_sp500_data(dataset, do_scaling, do_pca, do_panda):
    if "sp500" in DATASET_SELECTION:
        # Load the data into a Pandas DataFrame
        sp500_df = dataset
        sp500_df.columns = sp500_df.iloc[0]
        sp500_df = sp500_df[1:]
        sp500_df = sp500_df.dropna(subset=['Close'])
        sp500_df['Close'] = pd.to_numeric(sp500_df['Close'], errors='coerce')
        sp500_df['Volume'] = pd.to_numeric(sp500_df['Volume'], errors='coerce')
        
        # Calculate Moving Averages
        sp500_df['MA_10d'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: (x - x.rolling(window=10).mean()) / x)
        sp500_df['MA_25d'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: (x - x.rolling(window=25).mean()) / x)
        sp500_df['MA_50d'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: (x - x.rolling(window=50).mean()) / x)
        sp500_df['MA_200d'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: (x - x.rolling(window=200).mean()) / x)

        tolerance = 1e-5
        sp500_df['MA_vol_10d'] = sp500_df.groupby('Ticker')['Volume'].transform(lambda x: (x - x.rolling(window=10).mean()) / (x.rolling(window=10).mean() + tolerance))
        sp500_df['MA_vol_25d'] = sp500_df.groupby('Ticker')['Volume'].transform(lambda x: (x - x.rolling(window=25).mean()) / (x.rolling(window=10).mean() + tolerance))
        sp500_df['MA_vol_50d'] = sp500_df.groupby('Ticker')['Volume'].transform(lambda x: (x - x.rolling(window=50).mean()) / (x.rolling(window=10).mean() + tolerance))
        sp500_df['MA_vol_200d'] = sp500_df.groupby('Ticker')['Volume'].transform(lambda x: (x - x.rolling(window=200).mean()) / (x.rolling(window=10).mean() + tolerance))
        
        # Calculate MACD
        def calculate_macd(df, short_window, long_window, signal_window, tolerance=1e-4):
            short_ema = sp500_df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=short_window, adjust=False).mean())
            long_ema = sp500_df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=long_window, adjust=False).mean())
            macd = short_ema - long_ema
            signal_line = macd.ewm(span=signal_window, adjust=False).mean()
            macd = macd.clip(lower=tolerance) 
            return (macd - signal_line)/macd
        # sp500_df['macd'], sp500_df['signal_line'] = calculate_macd(sp500_df, short_window, long_window, signal_window)
        sp500_df['macd_signal_ratio'] = calculate_macd(sp500_df, short_window, long_window, signal_window)

        
        # Calculate Bollinger Bands

        def calculate_bollinger_ratio(df, window_size, tolerance=1e-8):
            bollinger_mid = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=window_size).mean())
            bollinger_std = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=window_size).std())
            bollinger_std = bollinger_std.clip(lower=tolerance)  # Clip the standard deviation to a minimum value
            return (df['Close'] - bollinger_mid) / (bollinger_std * 2)

        sp500_df['bollinger_band_ratio'] = calculate_bollinger_ratio(sp500_df, window_size)
       
        # Calculate RSI
        def calculate_rsi(df, period):
            
            delta = df['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            exponentially_weighted_moving_average_gain = gain.ewm(com=period - 1, adjust=False).mean()
            exponentially_weighted_moving_average_loss = loss.ewm(com=period - 1, adjust=False).mean()
            rs = exponentially_weighted_moving_average_gain / exponentially_weighted_moving_average_loss
            return 100 - (100 / (1 + rs))
        sp500_df['rsi'] = calculate_rsi(sp500_df, rsi_period)
       

        # Calculate ROC
        sp500_df['roc'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: ((x - x.shift(roc_period)) /  (x.shift(roc_period) + tolerance) ).clip(lower=tolerance)  * 100)

        # Calculate DoD Delta
        sp500_df['dod_delta'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: np.where(x < x.shift(-1), 1, 0))
        sp500_df['dod5_delta'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: np.where(x < x.shift(-5), 1, 0))

        # Drop NaN values
        sp500_df.dropna(inplace=True)
        columns_to_drop = ['Open', 'Close', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Ticker']
        sp500_df.drop(columns=columns_to_drop, inplace=True)
        print("Columns:", sp500_df.columns)
        print("Shape:", sp500_df.shape)
        print(sp500_df.head(20))

        with open(SP500_PROCESSED_DATA_PATH, 'wb') as f:
            pickle.dump(sp500_df, f)
        print(f"Results saved to {SP500_PROCESSED_DATA_PATH}")

        def print_df_stats(df):
            print("Shape:", df.shape)
            print("Columns:", df.columns)
            print("Data Types:", df.dtypes)

            for col in df.columns:
                print(f"\nColumn: {col}")
                print("Unique Values:", df[col].nunique())
                if pd.api.types.is_numeric_dtype(df[col]):
                    print("Min:", df[col].min())
                    print("Max:", df[col].max())
                    print("Median:", df[col].median())
                    print("Example Values:", df[col].sample(3).tolist())
                else:
                    print("Example Values:", df[col].sample(3).tolist())

        

        print_df_stats(sp500_df)



    else: 
        raise ValueError("Invalid dataset specified. Check config.py")
          
    
    #probably need to deprecate
    if do_scaling:
        scaling=StandardScaler()
        scaling.fit(X_df) 
        # X_df=scaling.transform(X_df)
        X_df_scaled = scaling.transform(X_df)  # Scaled data is a NumPy array
        X_df = pd.DataFrame(X_df_scaled, )
        print("Scaling implemneted")
    if do_pca:    
        pca = PCA(n_components=.95)  # get enough for 95% of var explained
        pca.fit(X_df)
        X_df_pca = pca.fit_transform(X_df)  # PCA transforms into NumPy array
        X_df = pd.DataFrame(X_df_pca, index=X_df.index, 
                        columns=[f'PC{i+1}' for i in range(X_df_pca.shape[1])])
        print("PCA implemneted")
    if do_panda:
        pass

    return sp500_df



def get_data():
    print(f"Getting data for {DATASET_SELECTION}")
    if "kaggle_rohlik_sales" in DATASET_SELECTION:
        # Load the DataFrame from the pickle file
        if not os.path.exists(PROCESSED_TRAIN_PATH):
            try:
                calendar = pd.read_csv(CALENDAR_PATH)
                inventory = pd.read_csv(INVENTORY_PATH)
                sales_train = pd.read_csv(TRAIN_PATH)
                print("Accessed .csv in data folder")
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


                inventory = inventory[['unique_id', 'warehouse', 'product_name', 'product_num', 'cat_name', 'cat_num']]
                sales_train = sales_train.merge(inventory, on=['unique_id', 'warehouse'], how='left')
                print("processed and merged inventory data")
                sales_train = sales_train.merge(calendar[['date', 'holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays','warehouse']],
                                                on=['date', 'warehouse'], how='left')
                print("merged calendar data")

                sales_train['date'] = pd.to_datetime(sales_train['date'])
                sales_train['day_of_week'] = sales_train['date'].dt.dayofweek
                sales_train['month'] = sales_train['date'].dt.month
                sales_train['year'] = sales_train['date'].dt.year
                
                label_encoders = {}
                for col in ['warehouse', 'product_name', 'cat_name']:
                    le = LabelEncoder()
                    sales_train[col] = le.fit_transform(sales_train[col])
                    label_encoders[col] = le  # Store the encoder for this column
                with open( f"{LABEL_ENCODERS_PKL_OUTDIR}/lencoders.pkl", "wb") as f:
                    pickle.dump(label_encoders, f)
                print("Label encoders saved!")
                sales_train['product_num'] = pd.to_numeric(sales_train['product_num'], errors='coerce')
                sales_train['cat_num'] = pd.to_numeric(sales_train['product_num'], errors='coerce')
                sales_train['sales_whole'] = sales_train.apply(
                    lambda row: row['sales'] / row['availability'] if row['availability'] < 1.0 else row['sales'], axis=1
                )
                sales_train = sales_train.dropna(subset=['sales'])
                nan_counts = sales_train[['sales', 'sales_whole', 'availability']].isnull().sum()
                availability_zero_count = (sales_train['availability'] == 0).sum()
                print("NaN counts:")
                print(nan_counts)
                print(f"\nNumber of rows where availability == 0: {availability_zero_count}")
                nan_summary = sales_train.isnull().sum()
                print("Missing values in each column:")
                print(nan_summary[nan_summary > 0])
                print("updated processed sales_train")
                sales_train = sales_train.drop(columns=['sales', 'availability','date','unique_id'])
                sales_train.to_pickle(PROCESSED_TRAIN_PATH)
                print(f"DataFrame updated and saved as pickle file: {PROCESSED_TRAIN_PATH}")
                print(sales_train.head())
                print(sales_train.info())

            except FileNotFoundError:
                print(f"Error: The file '{TRAIN_PATH}' was not found.")
                return None
            except Exception as e:
                print(f"Error loading data: {e}")
                return None
        else:
            #load pickl
            sales_train = pd.read_pickle(PROCESSED_TRAIN_PATH)
        print("retreived sales_train")
       
        print(f"DataFrame updated and saved as pickle file: {PROCESSED_TRAIN_PATH}")

        ###############
        Y_df = sales_train['sales_whole']  # Target variable
        X_df = sales_train.drop(columns=[ 'sales_whole'])
        print(X_df.info())
        print(Y_df.info())
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



def inspect_pickle_content(pkl_path):
    """
    Inspect contents of a pickle file, showing structure and samples
    """
    print(f"\nInspecting pickle file: {pkl_path}")
    print("=" * 80)
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    print("\n1. Basic Info:")
    print(f"Type of loaded data: {type(data)}")
    print(f"Is empty? {not bool(data)}")
    
    if isinstance(data, dict):
        print(f"\n2. Dictionary Structure:")
        print(f"Number of top-level keys: {len(data)}")
        print("\nTop-level keys:")
        for key in list(data.keys())[:5]:  # First 5 keys
            print(f"- {key} ({type(key)})")
            
        # Sample a random key for deeper inspection
        if data:
            sample_key = next(iter(data))
            print(f"\n3. Sample value for key '{sample_key}':")
            sample_value = data[sample_key]
            print(f"Type: {type(sample_value)}")
            
            # If the value is also a dictionary, show its structure
            if isinstance(sample_value, dict):
                print("Nested dictionary structure:")
                for k, v in list(sample_value.items())[:3]:  # First 3 items
                    print(f"- {k}: {type(v)}")
                    if isinstance(v, (dict, list)):
                        print(f"  Length: {len(v)}")
                    try:
                        print(f"  Sample: {str(v)[:100]}...")  # First 100 chars
                    except:
                        print("  Sample: [Cannot display sample]")
            
            # If it's a list or array, show some info
            elif isinstance(sample_value, (list, np.ndarray)):
                print(f"Length: {len(sample_value)}")
                print("First few elements:")
                print(sample_value[:3])
    
    # If it's not a dictionary, show appropriate info
    else:
        if isinstance(data, (list, np.ndarray)):
            print(f"\nArray/List Info:")
            print(f"Length: {len(data)}")
            print("First few elements:")
            print(data[:3])
        else:
            print("\nData sample:")
            try:
                print(str(data)[:200])  # First 200 chars
            except:
                print("[Cannot display data sample]")



def get_test_data():
    print(f"Getting test data for {DATASET_SELECTION}")
    if "kaggle_rohlik_sales" in DATASET_SELECTION:
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


if __name__ == "__main__":
    ###################
    pass

