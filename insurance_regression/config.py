import socket

# Determine the hostname
hostname = socket.gethostname()
if hostname == "Khais-MacBook-Pro.local" or hostname == "Khais-MBP.attlocal.net":  
    from insurance_regression.config_mac import *  
else:
    from insurance_regression.config_cuda import * 

import os


# CALENDAR_PATH = os.path.join(os.getcwd(), 'data', 'calendar.csv')
# INVENTORY_PATH = os.path.join(os.getcwd(), 'data', 'inventory.csv')
TRAIN_PATH = os.path.join(os.getcwd(), 'data', 'train.csv')

TEST_PATH = os.path.join(os.getcwd(), 'data', 'test.csv')

# Global ETL Version
ETL_VERSION = 'v2'  

# Define paths for processed datasets based on ETL_version
PROCESSED_TRAIN_PATH = os.path.join(os.getcwd(), 'data', f'train_processed_{ETL_VERSION}.csv')
PROCESSED_TEST_PATH = os.path.join(os.getcwd(), 'data', f'test_processed_{ETL_VERSION}.csv')



DATASET_SELECTION = "kaggle_insurance_regression" #kaggle_housing #kaggle_housing_test

EVAL_FUNC_METRIC = 'rmlse' #'mae'  #rmse #'f1' # 'accuracy' 
N_ESTIMATOR = 5

EVAL_MODELS = [
                # 'default',
                'MPL',
                'CNN', 
                'LSTM', 
                'bi-LSTM',
                'conv-LSTM', 
                #'seg-gru',
                ]

PARAM_GRID = {
    'lr': [0.01, 0.005, 0.0005],
    'batch_size': [16, 32],
    
    # 'hidden_layers': [[75,19]],
    'dropout_rate': [0, 0.005, 0.01, ],
    'hidden_layers': [[64, 32], [128, 64, 32], [64],[75]],
    # 'activation_function': just use relu
}


from pathlib import Path
def set_output_dir(path):
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    return path
# Get the root project directory (the parent directory of kaggle_housing)
project_root = Path(__file__).resolve().parent.parent
# Define the output directory path relative to the project root
OUTPUT_DIR_A3 = project_root / 'outputs' / DATASET_SELECTION
DRAFT_VER_A3 = 1
# Set the directories using set_output_dir
AGGREGATED_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}_etl{ETL_VERSION}/aggregated_graphs')
Y_PRED_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}_etl{ETL_VERSION}/y_pred_graphs')
CV_LOSSES_PKL_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}_etl{ETL_VERSION}/pkl_cv')
PERFM_PKL_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}_etl{ETL_VERSION}/perf_pkl')
MODELS_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}_etl{ETL_VERSION}/saved_models')
LABEL_ENCODERS_PKL_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}_etl{ETL_VERSION}/label_encoders')
SOLUTIONS_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}_etl{ETL_VERSION}/solutions')
TXT_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}_etl{ETL_VERSION}/txt_stats')
OUTPUT_DIR_RAW_DATA_A3 =set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}_etl{ETL_VERSION}/raw_data_assessments')

MODEL_ALL_LOG_FILE = os.path.join(os.getcwd(), TXT_OUTDIR, 'all_models_logs.txt')


#ML PARAMS
K_FOLD_CV = 5