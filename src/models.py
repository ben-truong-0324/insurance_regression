

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



class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[64, 32], dropout_rate=0.5):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout_rate = dropout_rate
        # Input layer to first hidden layer
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        # Hidden layers with dropout in between
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            x = self.dropout(x)  # Apply dropout after each hidden layer
        
        return self.layers[-1](x)  # No activation or dropout on the output layer


class FarsightMPL(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super(FarsightMPL, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, 75))
        self.layers.append(nn.Linear(75, output_dim))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.relu(self.layers[0](x))  # First Linear layer
        x = self.dropout(x)
        x =self.layers[1](x)
        
        return x


class FarsightCNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=289, feature_maps=19, dropout_rate=0.5):
        super(FarsightCNN, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=feature_maps, kernel_size=3, stride=1, padding=0))
        self.layers.append(nn.Linear(feature_maps * 15 * 15, output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            if isinstance(layer, nn.Conv2d):
                # Reshape the input to match Conv2d's expected input
                #x = x.view(-1, 1, int(289**0.5), int(289**0.5))
                x = x.view(x.size(0), 1, 17, 17)  # Batch size, Channels, Height, Width
            x = torch.relu(layer(x))
            x = self.dropout(x)  # Apply dropout after each hidden layer
        #x = torch.flatten(x, 1)  # lfatten for fully connected layer
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, num_features]
        return self.layers[-1](x)
    
 
    
class FarsightLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=289, lstm_hidden_dim=300, dropout_rate=0.1):
        super(FarsightLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True))
        self.layers.append(nn.Linear(lstm_hidden_dim, output_dim))

    def forward(self, x):
        x = torch.relu(self.layers[0](x))  # First Linear layer
        x = self.dropout(x)
        x = x.unsqueeze(1)
        lstm_out, _ = self.layers[1](x)  # LSTM returns (output, (hidden_state, cell_state))
        x = torch.relu(lstm_out[:, -1, :]) 
        x = self.dropout(x)
        return self.layers[-1](x)


    
class FarsightBiLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=289, lstm_hidden_dim=150, dropout_rate=0.5):
        super(FarsightBiLSTM, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True, bidirectional=True))
        self.layers.append(nn.LSTM(lstm_hidden_dim * 2, lstm_hidden_dim, batch_first=True, bidirectional=True))
        self.layers.append(nn.Linear(lstm_hidden_dim * 2, output_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.relu(self.layers[0](x))  # Apply ReLU after the first linear layer
        x = self.dropout(x)  # Apply dropout
        if x.dim() == 2:  # If x is 2D, add sequence dimension
            x = x.unsqueeze(1)  # Shape becomes (batch_size, seq_length=1, hidden_dim)
        lstm_out1, _ = self.layers[1](x)  # LSTM returns (output, (hidden_state, cell_state))
        lstm_out1 = self.dropout(lstm_out1)  # Apply dropout
        lstm_out2, _ = self.layers[2](lstm_out1)  # LSTM output from the first BiLSTM
        lstm_out2 = self.dropout(lstm_out2)  # Apply dropout
        x = lstm_out2[:, -1, :]  # Get the last output in the sequence (batch_size, lstm_hidden_dim * 2)
        return self.layers[-1](x)
    


class FarsightConvLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=289, feature_maps=19, lstm_hidden_dim=300, dropout_rate=0.5):
        super(FarsightConvLSTM, self).__init__()
        self.layers = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=feature_maps, kernel_size=3, stride=1, padding=0))
        self.layers.append(nn.Linear(feature_maps * 15 * 15, hidden_dim))
        self.layers.append(nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True))
        self.layers.append(nn.Linear(lstm_hidden_dim, output_dim))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            if isinstance(layer, nn.Conv2d):
                # Reshape the input to match Conv2d's expected input
                x = x.view(x.size(0), 1, 17, 17)  # Batch size, Channels, Height, Width
                x = torch.relu(layer(x))
                x = self.dropout(x)
                x = x.view(x.size(0), -1) 
            elif isinstance(layer, nn.LSTM):
                if x.dim() == 2:  # If x is 2D, add sequence dimension
                    x = x.unsqueeze(1) 
                lstm_out, _ = self.layers[3](x)  # LSTM output (batch_size, seq_length, lstm_hidden_dim)
                x = lstm_out[:, -1, :]
            elif isinstance(layer, nn.Linear):
                x = torch.relu(layer(x))
                x = self.dropout(x)
        return self.layers[-1](x)



def evaluate_model(model, X_val, y_val, device,criterion):
    model.eval()
    with torch.no_grad():
        # Ensure inputs are on the correct device
        X_val_tensor = X_val.to(device) if isinstance(X_val, torch.Tensor) else torch.tensor(X_val).to(device)
        y_val_tensor = y_val.to(device) if isinstance(y_val, torch.Tensor) else torch.tensor(y_val).to(device)
        
        # Forward pass
        outputs = model(X_val_tensor)
        
        # Calculate loss (for single-label classification)
        loss = criterion(outputs, y_val_tensor)
    return loss.item()

# Function to optimize the threshold for accuracy
def optimize_threshold(test_outputs_np, y_test_np):
    best_accuracy = 0
    best_threshold = 0.5  # Start with a default threshold (0.5)
    
    # Test a range of threshold values from 0 to 1
    for threshold in np.linspace(0, 1, 101):  # 101 points between 0 and 1
        predicted_labels = (test_outputs_np >= threshold).astype(int)
        
        # Calculate accuracy for this threshold
        label_accuracy = (predicted_labels == y_test_np).mean(axis=0)  # accuracy for each label
        accuracy = label_accuracy.mean()  # average accuracy across all labels
        
        # If this threshold gives a better accuracy, update the best threshold and accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"Optimal Threshold: {best_threshold:.2f}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    return best_threshold, best_accuracy