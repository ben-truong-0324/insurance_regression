

import numpy as np
from sklearn.model_selection import ParameterSampler


GT_ID = 68420 #randomizer seed

DATASET_SELECTION = 'doc2vec' #credit using credit dataset for NN weight optimizing via RO

# 'sp500'
# 'gps'
# 'phishing'
DOC2VEC_DATA_PATH = '../data/doc2vec_dataset_full.pkl'
NMF_BOW_DATA_PATH = '../data/nmf_bow_dataset.pkl'
NMF_TW_DATA_PATH = '../data/nmf_tw_dataset.pkl'

EXP_DEBUG = 0
DATA_DEBUG = 0

OUTPUT_DIR_A3 = f'../outputs/{DATASET_SELECTION}'
DRAFT_VER_A3 = 9

RANDOM_OPTIMIZATION_ITERATION_COUNT = 1
NN_MAX_EPOCH = 100

# _lr___0_0005___batch_size___16___dropout_rate___0_1___hidden_layers____64_
FARSIGHT_PARAM_GRID = {
    'lr': [.0001],
    'batch_size': [ 16],
    'dropout_rate': [ 0 ],
    'hidden_layers': [ [64]],
    # 'activation_function': just use relu
}
FARSIGHT_SRX_PARAMS = list(ParameterSampler(FARSIGHT_PARAM_GRID, n_iter=RANDOM_OPTIMIZATION_ITERATION_COUNT, random_state=GT_ID))







############################
#not used in this project, but some var may still be referenced
OUTPUT_DIR_OPTIMIZE = f'../graphs'
# OPT_DRAFT_VER = 0

CREDIT_DATA_PATH = '../../data/credit+approval/crx.data'
SP500_DATA_PATH = '../../data/sp500_dataset_Oct27'
SP500_PROCESSED_DATA_PATH = '../../data/sp500_processed.pkl'

GPS_DATA_PATH1 = '../../data/GPS_Trajectory/go_track_tracks.csv'
GPS_DATA_PATH2 = '../../data/GPS_Trajectory/go_track_trackspoints.csv'
PHISHING_DATA_PATH = '../../data/Phishing_Legitimate_full.csv'

CLUSTERING_MIN_K = 18
CLUSTERING_MAX_K = 19

CLUSTER_ALGORITHMS = [ 'kmeans', 
                        # 'gmm',
                        # 'dbscan',
                        # 'specclus','birch','meanshift','aggclus',
                        
                        ]



RO_ALGORITHMS = ['RHC', 'SA', 'GA', 'MIMIC']
ALGO_COLORS = {'RHC': 'blue', 'SA': 'red', 'GA': 'green', 'MIMIC': 'black'}

MAX_WEIGHT_PCT = .6

PROBLEM_SIZES = list(range(10, 51, 10))
PROBLEM_SIZE_DEFAULT = 30
HYPERPARAM_VALUES_TO_VALIDATE = 5 #graph for 15 increments of hyperparm values to validate

MAX_ATTEMPTS = 200
MAX_ITERS = np.inf
MAX_MIMIC_ITER = 1000

RESTARTS = 50  # For RHC

# schedule = mlrose.ExpDecay()  # For SA
POP_SIZE = 200  # For GA and MIMIC
MUTATION_PROB = 0.1  # For GA
KEEP_PCT = 0.3  # For MIMIC

#Cross validation
MONTE_CARLO_ITER = 3
MONTE_CARLO_CV_ITER = 100
MONTE_CARLO_CV_TRAIN_SIZE = .8

#NN
MONTE_CARLO_NN_ITER = 5
TRAIN_SIZE = .8
TEST_SIZE = .2
HIDDEN_NODES = [16,8]
NN_PATIENCE = 1

#DIM REDUCTION
DIMENSION_REDUCE_METHODS = ["PCA", 
                        # "ICA",
                        # "RP",
                        #  "RCA",
                #  "LDA", "RandomForest",
                # "NMF", "CUR",
                ]

# NMF: Useful for non-negative data like topic modeling, image processing, or audio signal decomposition.
# SVD and PCA: Used for dimensionality reduction, feature extraction, and latent semantic analysis.
# CUR: Applied when interpretable submatrices of the original matrix are needed, often in large, sparse datasets.
# Tensor Decomposition: Useful for multi-way data (e.g., in recommendation systems, multi-dimensional time series).



# Define ranges for hyperparameters
PARAM_GRID = {
    'lr': [0.01, 0.005, 0.0005],
    'batch_size': [16, 32, 64],
    'hidden_layers': [[64, 32], [128, 64, 32], [64]],
    # 'hidden_layers': [[75,19]],
    'dropout_rate': [0, 0.1, 0.05, 0.3],
    # 'activation_function': just use relu
}

# Generate a random sample of 15 combinations from the grid
RANDOM_SRX_PARAMS = list(ParameterSampler(PARAM_GRID, n_iter=RANDOM_OPTIMIZATION_ITERATION_COUNT, random_state=GT_ID))

