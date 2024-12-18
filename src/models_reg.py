

import re
import time
import os
from datetime import datetime
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate 
import pickle
import random
from copy import deepcopy
import math

from scipy.stats import ttest_1samp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,precision_score, \
                        recall_score,classification_report, \
                        accuracy_score, f1_score, log_loss, \
                       confusion_matrix, ConfusionMatrixDisplay,\
                          roc_auc_score, matthews_corrcoef, average_precision_score
from sklearn.cluster import KMeans, AgglomerativeClustering,DBSCAN,Birch,MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterSampler

#import dimension reduction modules
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset


from sklearn.cluster import KMeans
import numpy as np

class MLPRegression(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048, dropout_rate=0.1):
        super(MLPRegression, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, output_dim)

    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)  

class MLPRegression1(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048, dropout_rate=0.1):
        super(MLPRegression1, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  
    

class MLPRegression2(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048, dropout_rate=0.1):
        super(MLPRegression2, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)  
    
class MLPRegression3(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048, dropout_rate=0.1):
        super(MLPRegression3, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  
    
class MLPRegression4(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048, dropout_rate=0.1):
        super(MLPRegression4, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        return torch.tanh(self.fc2(x))
    
class MLPRegression5(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048, dropout_rate=0.1):
        super(MLPRegression5, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.tanh(self.fc2(x))
    
class MLPRegression6(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048, dropout_rate=0.1):
        super(MLPRegression6, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.relu(self.fc2(x))
    
class MLPRegression7(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048, dropout_rate=0.1):
        super(MLPRegression7, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        return torch.relu(self.fc2(x))

class MLPRegression8(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048, dropout_rate=0.1):
        super(MLPRegression8, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.relu(self.fc3(x))
    
class MLPRegression9(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048, dropout_rate=0.1):
        super(MLPRegression9, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.nn.functional.softplus(self.fc3(x))

class CNNRegression(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, hidden_dim, dropout_rate):
        super(CNNRegression, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(hidden_dim * (input_dim // 2), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LSTMRegression(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate):
        super(LSTMRegression, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Use the last output of the LSTM for regression
        return output


class SalienceNNRegression(nn.Module):
    def __init__(self, input_dim, output_dim,hidden_dim, n_clusters=128):
        super(SalienceNNRegression, self).__init__()
        # K-Means clustering model to find clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        # MLP layer to process the clustered data
        self.fc1 = nn.Linear(input_dim + n_clusters * input_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, output_dim) 
        self.relu = nn.ReLU()

    def fit_kmeans(self, X_train):
        self.kmeans.fit(X_train)
    
    def forward(self, X, X_train):
        # Step 1: Get the cluster assignments for the input instances
        cluster_labels = self.kmeans.predict(X.cpu().numpy())  # Get cluster assignments
        # Step 2: Get the cluster centroids (hidden states)
        centroids = self.kmeans.cluster_centers_  # Shape: (n_clusters, input_dim)
        # Step 3: For each instance, get the cluster centroid (or use other strategies like KNN to get neighbors)
        cluster_context = centroids[cluster_labels]  # Shape: (batch_size, input_dim)
        # Step 4: Concatenate the original features with the cluster context (hidden state)
        combined_input = torch.cat((X, torch.tensor(cluster_context).float().to(X.device)), dim=1)
        # Step 5: Feed through the MLP layers
        x = self.relu(self.fc1(combined_input))
        x = self.fc2(x)
        return x
