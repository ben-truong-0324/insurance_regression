from config import *
from tests import *
from src.models import *
from utils import *
import rohlik_sales.etl as etl
import rohlik_sales.plots as plots
import pickle
import glob
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
import hypotheses
import math
import itertools

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



def train_nn_with_early_stopping_with_param(X_train, y_train, X_test, y_test, params, max_epochs, patience, model_name="default"):
    lr = params['lr']
    batch_size = params['batch_size']
    hidden_layers = params['hidden_layers']
    dropout_rate = params['dropout_rate']
    input_dim = X_train.shape[1]
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        # Multi-label classification (y_train has multiple labels per instance)
        output_dim = y_train.shape[1]  # Number of labels
    else:
        # Single-label classification (y_train has a single label per instance)
        output_dim = len(np.unique(y_train.cpu())) 
    if model_name == "default":
        model = SimpleNN(input_dim, output_dim, hidden_layers, dropout_rate=dropout_rate).to(device)
    elif model_name == "MPL":
        model = FarsightMPL(input_dim=input_dim, output_dim=output_dim).to(device)
    elif model_name == "CNN":
        model = FarsightCNN(input_dim=input_dim, output_dim=output_dim,hidden_dim=289, feature_maps=19, dropout_rate=params['dropout_rate']).to(device)

    elif model_name == "LSTM":
        model = FarsightLSTM(input_dim=input_dim, output_dim=output_dim,hidden_dim=289, lstm_hidden_dim=300, dropout_rate=params['dropout_rate']).to(device)

    elif model_name == "bi-LSTM":
        model = FarsightBiLSTM(input_dim=input_dim, output_dim=output_dim,  hidden_dim=289, lstm_hidden_dim=150, dropout_rate=params['dropout_rate']).to(device)

    elif model_name == "conv-LSTM":
        model = FarsightConvLSTM(input_dim=input_dim, output_dim=output_dim,  hidden_dim=289, feature_maps=19, lstm_hidden_dim=300, dropout_rate=params['dropout_rate']).to(device)

    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    # model = FarsightMPL(input_dim, output_dim, dropout_rate).to(device)



    if len(y_train.shape) == 1:  
        criterion = nn.CrossEntropyLoss()  # For single label classification (multi-class)
    else:  
        criterion = nn.BCEWithLogitsLoss()  # For multi-label classification

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    epoch_trained = 0

    epoch_losses = []
    start_time = time.time()
    print("Starting training loop...")
    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        epoch_trained+=1
        model.train()

        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_epoch_loss = running_loss / len(train_loader)
        # Validation
        val_loss = evaluate_model(model, X_test, y_test, device,criterion)
        epoch_losses.append((avg_epoch_loss,val_loss))
        print(f"Epoch {epoch}, last train_loss {epoch_losses[-1][0]:.5F} val_loss {val_loss:.5F} per {criterion}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
        epoch_runtime = time.time() - epoch_start_time
        print(f"Epoch completed in {epoch_runtime // 60:.0f}m {epoch_runtime % 60:.0f}s\n")
    runtime = time.time() - start_time
    print(model)
    print(f"Model {model_name} Training completed in {runtime // 60:.0f}m {runtime % 60:.0f}s\n")
    print(f"Average time per epoch: {(runtime / epoch_trained )//60:.0f}m {(runtime / epoch_trained)%60:.0f}s")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)

        _, predicted = torch.max(outputs, 1)

        if y_train.shape[1] > 1:
            test_outputs_np = outputs.sigmoid().cpu().numpy()  # Sigmoid for multi-label probability
           
            y_test = y_test.cpu().numpy()
            best_threshold = .5
            predicted = (test_outputs_np >= best_threshold).astype(int)
            # Calculate accuracy, AUC-ROC, and F1-score for multi-label classification
            # Individual label accuracy (mean accuracy for each label)
            label_accuracy = (predicted == y_test).mean(axis=0)
            accuracy = label_accuracy.mean()  # Average accuracy across all labels
            try:
                auc_roc = roc_auc_score(y_test, test_outputs_np, average="macro")
            except ValueError:
                auc_roc = float("nan")  # Handle cases where AUC-ROC can't be calculated
            f1 = f1_score(y_test, predicted, average="macro")
            mcc = matthews_corrcoef(y_test.flatten(), predicted.flatten())
            auprc = average_precision_score(y_test, test_outputs_np, average="macro")

        #if y is single label
        else:
            accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
            f1 = f1_score(y_test.cpu(), predicted.cpu(), average='weighted')
            # probs = torch.softmax(outputs, dim=1)
            # auc_roc = roc_auc_score(y_test.cpu(), probs.cpu(), multi_class='ovr')  # For multi-class problems
            probs = torch.sigmoid(outputs)[:, 0]  # Assuming the positive class is the first one
            auc_roc = roc_auc_score(y_test.cpu(), probs.cpu())
            mcc = matthews_corrcoef(y_test.cpu(), predicted.cpu())
            auprc = average_precision_score(y_test.cpu(), probs.cpu())


        ##################
    print(f"Training terminated after epoch {epoch_trained}, "
            f"Avg Label Accuracy: {accuracy:.4f}, "
            f"AUC-ROC: {auc_roc:.4f}, "
            f"F1-Score: {f1:.4f}, "
            f"MCC: {mcc:.4f}, "
            f"AU-PRC: {auprc:.4f}")


    
    return accuracy, f1,auc_roc, mcc, auprc, runtime,model,epoch_losses,y_test,predicted


def run_model_tuning_RO_for_Xy_srx_space(X, y, do_cv, random_opt_algo, best_overall_metric, best_overall_method, best_overall_model, best_overall_cv_losses, type_tag = "",model_name = "default"):
    """
    Generalized helper method to run random optimization algorithms.
    
    Args:
        X: Feature data.
        y: Target labels.
        do_cv: Cross-validation flag.
        random_opt_algo: Optimization algorithm, e.g., "RHC", "GA", "SA".
        best_overall_metric: Initial or baseline metric to compare against.
        best_overall_method: Description of the method or optimization algorithm used.
        best_overall_model: Initial model or None.
    
    Returns:
        best_overall_metric: The best metric obtained.
        best_overall_model: The best model found during optimization.
        best_overall_method: The name of the best method.
        running_best_metrics_of_Xy: List of best metrics per iteration.
    """
    
    # Initialize the current best metric and model for this run
    running_best_model = None
    running_best_overall_cv_losses = None
    running_best_metrics_of_Xy_srx_space = []
    running_best_y_preds = None
    outer_ro_running_best_metric = 0
    running_best_result_dict = {}
    #nested loops, outer loop iterates different params, inner loop iterates for kf CV

    # Set best metric and model placeholders for return
    final_best_metric = best_overall_metric
    final_best_model = best_overall_model
    final_best_method = best_overall_method
    final_best_overall_cv_losses = best_overall_cv_losses
    
    param_combinations = list(itertools.product(*FARSIGHT_PARAM_GRID.values()))

    for params in param_combinations:
        current_params = {
            'lr': params[0],
            'batch_size': params[1],
            'dropout_rate': params[2],
            'hidden_layers': params[3],
        }
        params_str = re.sub(r'[^\w\-_]', '_', str(current_params))  # Replace invalid characters with '_'
        pkl_filename = f"{NN_PKL_OUTDIR}/farsight_results_{EVAL_FUNC_METRIC}_{model_name}_{params_str}.pkl"
        stats_filename = f"{TXT_OUTDIR}/farsight_results_{EVAL_FUNC_METRIC}_{model_name}_{params_str}.txt"
        if not os.path.exists(pkl_filename) or not os.path.exists(stats_filename):

            current_metrics_of_Xy = []
            inner_cv_running_best_metric = 0

            avg_metric_per_cv = [0 for _ in range(K_FOLD_CV)] if do_cv else [0]
            cv_losses = []
            y_preds = []
            for iteration in range(NUM_STATISTICAL_ITER):
                print(f"Starting iteration {iteration + 1} of {NUM_STATISTICAL_ITER}")
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X) if do_cv else [(range(len(X)), range(len(X)))]):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train and evaluate model with current parameters
                    accuracy, f1,auc_roc,mcc, auprc, runtime,temp_model,epoch_losses,y_test,predicted = train_nn_with_early_stopping_with_param(
                        X_train, y_train, X_val, y_val, current_params,NN_MAX_EPOCH, NN_PATIENCE, model_name,
                    )
                    
                    # Store the current metrics
                    current_metrics_of_Xy.append((accuracy, f1, runtime, auc_roc,mcc, auprc))
                    
                    # Choose evaluation metric
                    if "f1" in EVAL_FUNC_METRIC:
                        avg_metric_per_cv[fold_idx] = f1
                    elif "accuracy" in EVAL_FUNC_METRIC:
                        avg_metric_per_cv[fold_idx] = accuracy
                    elif "auc" in EVAL_FUNC_METRIC:
                        avg_metric_per_cv[fold_idx] = auc_roc
                    cv_losses.append(epoch_losses)
                    y_preds.append((y_test,predicted))

            # Calculate average metric across folds
            avg_metric = np.mean(avg_metric_per_cv)
            
            # Update running best if the new metric is better
            if avg_metric > inner_cv_running_best_metric:
                inner_cv_running_best_metric = avg_metric

            if inner_cv_running_best_metric > outer_ro_running_best_metric:
                outer_ro_running_best_metric = inner_cv_running_best_metric
                running_best_model = temp_model
                running_best_metrics_of_Xy_srx_space = current_metrics_of_Xy
                running_best_y_preds = y_preds
                running_best_overall_cv_losses = cv_losses
                avg_accuracy, std_accuracy, avg_mcc, avg_f1, avg_roc_auc, avg_pr_auc = get_metrics_of_hyperparm_set(y_preds)
            
                running_best_result_dict = {
                    'model_name': model_name,
                    'avg_accuracy': avg_accuracy,
                    'std_accuracy': std_accuracy,  # Save the standard deviation for accuracy
                    'avg_mcc': avg_mcc,
                    'avg_f1': avg_f1,
                    'avg_roc_auc': avg_roc_auc,
                    'avg_pr_auc': avg_pr_auc,
                    'max_epoch': NN_MAX_EPOCH,
                    'current_params': current_params,
                    'current_metrics_of_Xy': current_metrics_of_Xy,
                    'y_preds': y_preds,
                    'cv_losses': cv_losses,
                    
                }
            
                with open(stats_filename, 'w') as f:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n")
                    f.write(f"Average MCC: {avg_mcc:.4f}\n")
                    f.write(f"Average F1 Score: {avg_f1:.4f}\n")
                    f.write(f"Average AUC-ROC: {avg_roc_auc:.4f}\n")
                    f.write(f"Average AUC-PR: {avg_pr_auc:.4f}\n")
                    f.write(f"max_epoch: {NN_MAX_EPOCH}\n")
                    f.write(f"Hyperparameters: {current_params}\n")
                print(f"Saved stats to {stats_filename}")
        

    # Check against overall best and update if necessary
    if outer_ro_running_best_metric > final_best_metric:
        final_best_metric = outer_ro_running_best_metric
        final_best_model = running_best_model
        final_best_method = type_tag
        final_best_overall_cv_losses = running_best_overall_cv_losses
            
    return final_best_metric, final_best_model, final_best_method, running_best_metrics_of_Xy_srx_space, final_best_overall_cv_losses, running_best_y_preds, running_best_result_dict

def get_eval_with_nn(X,y,nn_pkl_path,cv_losses_outpath):
    if not os.path.exists(nn_pkl_path) or not os.path.exists(cv_losses_outpath):
        X = pd.DataFrame(X)  # Assuming X_train is a DataFrame
        X_features = torch.FloatTensor(X.values).to(device)
        best_overall_model = None
        best_overall_method = None
        best_overall_metric = 0
        best_overall_cv_losses = None
        nn_results={}
        try:
            if len(y.shape) > 1 and y.shape[1] > 1:
                # Multi-label case: use FloatTensor for multi-hot encoded labels
                try:
                    y_labels = torch.FloatTensor(y).to(device)
                except:
                    y_labels = torch.FloatTensor(y.values).to(device)
            else:
                try:
                    y_labels = torch.LongTensor(y).to(device)
                except:
                    y_labels = torch.LongTensor(y.values).to(device)
        except AttributeError:
            # If y is a Pandas DataFrame or Series, convert to NumPy first
            y_values = y.values if hasattr(y, 'values') else y
            if len(y_values.shape) > 1 and y_values.shape[1] > 1:
                y_labels = torch.FloatTensor(y_values).to(device)
            else:
                y_labels = torch.LongTensor(y_values).to(device)
        ###################################
        for model_name in FARSIGHT_MODELS:
            best_overall_metric, best_overall_model, best_overall_method, running_metrics_Xy_srx_space, \
                best_overall_cv_losses,running_best_y_preds,running_best_result_dict = run_model_tuning_RO_for_Xy_srx_space(
                    X_features, 
                    y_labels, 
                    do_cv=True, 
                    random_opt_algo="default", 
                    best_overall_metric=best_overall_metric,  # Keyword argument
                    best_overall_method=best_overall_method,    # Keyword argument
                    best_overall_model=best_overall_model,    # Keyword argument
                    best_overall_cv_losses = best_overall_cv_losses,
                    type_tag=f"farsight_{model_name}",             # Keyword argument,
                    model_name = model_name,
                )
            nn_results[model_name] = {'mc_results': running_metrics_Xy_srx_space}
            with open(f'{NN_PKL_OUTDIR}/farsight_best_{model_name}_hyperparam_set.pkl', 'wb') as f:
                pickle.dump(running_best_result_dict,f)
            with open(f'{Y_PRED_PKL_OUTDIR}/y_pred_best_of_{model_name}.pkl', 'wb') as f:
                pickle.dump(running_best_y_preds,f)
            print(f"Saved results to {Y_PRED_PKL_OUTDIR}/y_pred_best_of_{model_name}.pkl")
        with open(f'{NN_PKL_OUTDIR}/farsight_best_of_{model_name}_nn_results.pkl', 'wb') as f:
            pickle.dump(nn_results,f)
        print(f"Saved results to {NN_PKL_OUTDIR}/farsight_best_of_{model_name}_nn_results.pkl")
        
    pass


def implement_farsight(X,y):

    #############RUNNING NN
    #to test for results using torch with cpu, run a differnt DRAFT_VER_A3
    nn_pkl_path = f'{NN_PKL_OUTDIR}/nn_results.pkl'
    cv_losses_outpath = f'{CV_LOSSES_PKL_OUTDIR}/cv_losses.pkl'
    get_eval_with_nn(X,y,nn_pkl_path,cv_losses_outpath)
    
   
def load_and_predict(model_path, X_test):
    loaded_model = torch.load(model_path)
    loaded_model.to(device)
    loaded_model.eval()
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.FloatTensor(X_test.values).to(device)

    # Generate predictions
    with torch.no_grad():  # No need to track gradients for inference
        outputs = loaded_model(X_test)
        if y_train.shape[1] > 1:
            test_outputs_np = outputs.sigmoid().cpu().numpy()  
            best_threshold = .5
            predictions = (test_outputs_np >= best_threshold).astype(int)
        else:
             _, predictions = torch.max(outputs, 1)
    
    return predictions

def result_check(X_test, y_test):
    model_files = [f for f in os.listdir(AGGREGATED_OUTDIR) if f.startswith('best_') and f.endswith('.pth')]
    for model_file in model_files:
        print(model_file)
        model_path = os.path.join(AGGREGATED_OUTDIR, model_file)
        print(model_path)
        predictions = load_and_predict(model_path, X_test)
        
        for indx, pred in enumerate(predictions):
            print(pred)
            print(y_test[indx])







def extract_parameters(filename):
    """Extract parameters from filename like 'pred_stats_accuracy_y_pred_PCA498kmeans23.pkl'"""
    # Extract the part after y_pred_
    params_part = filename.split('y_pred_')[1].replace('.pkl', '')
    
    # Use regex to extract parameters
    pattern = r'([A-Za-z]+)(\d+)([A-Za-z]+)(\d+)'
    match = re.match(pattern, params_part)
    
    if match:
        dreduc_algo = match.group(1)    # PCA
        dreduc_k = int(match.group(2))  # 498
        cluster_algo = match.group(3)    # kmeans
        cluster_k = int(match.group(4))  # 23
        
        return {
            'dreduc_algo': dreduc_algo,
            'dreduc_k': dreduc_k,
            'cluster_algo': cluster_algo,
            'cluster_k': cluster_k
        }
    return None

def extract_y_pred_parameters(filename):
    """
    Extract parameters from the filename. Adjusts for baseline cases and missing components.
    """
    params = {
        'dreduc_algo': None,  # Dimensionality reduction algorithm
        'dreduc_k': None,     # Dimensionality reduction parameter (k)
        'cluster_algo': None, # Clustering algorithm
        'cluster_k': None,    # Clustering parameter (k)
        'model_name': None    # Model name for farsight cases
    }

    # If "farsight" is in the filename, extract the model name
    if "farsight" in filename:
        model_name_match = re.search(r"farsight_(\w+)", filename)
        if model_name_match:
            params['model_name'] = model_name_match.group(1)
        return {'model_name': params['model_name']}  # Return only model_name for farsight case

    dreduc_pattern = rf"({'|'.join(DIMENSION_REDUCE_METHODS)})(\d+)"  # Match algo and k
    cluster_pattern = rf"({'|'.join(CLUSTER_ALGORITHMS)})(\d+)"       # Match algo and k
    # Baseline case
    if "baseline" in filename:
        params['cluster_algo'] = "baseline"
        return params
    # Extract dimensionality reduction parameters
    dreduc_match = re.search(dreduc_pattern, filename)
    if dreduc_match:
        params['dreduc_algo'] = dreduc_match.group(1)
        params['dreduc_k'] = int(dreduc_match.group(2))
    # Extract clustering parameters
    cluster_match = re.search(cluster_pattern, filename)
    if cluster_match:
        params['cluster_algo'] = cluster_match.group(1)
        params['cluster_k'] = int(cluster_match.group(2))

    return params if dreduc_match or cluster_match else None



def merge_y_pred_pkl_files(input_dir, output_file):
    """
    Merge all prediction stat pkl files into one consolidated pickle file
    
    Parameters:
    input_dir: directory containing individual pkl files
    output_file: path for the merged pickle file
    """
    if not os.path.exists(output_file):
        # Dictionary to store all results
        merged_results = {}
        
        # Get all relevant pkl files
        pkl_files = glob.glob(f"{input_dir}/pred_stats_*.pkl")
        
        for pkl_file in pkl_files:
            # Get base filename
            base_filename = pkl_file.split('/')[-1]

            # Extract parameters from filename
            params = extract_y_pred_parameters(base_filename)
            if params is None:
                print(f"Skipping file {base_filename} - doesn't match expected pattern")
                continue

                
            # Load the pickle file
            with open(pkl_file, 'rb') as f:
                stats_data = pickle.load(f)

            if 'model_name' in params and params['model_name']:
                key = params['model_name']
            else:
                key = (
                    params.get('dreduc_algo', 'None'),
                    params.get('dreduc_k', 'None'),
                    params.get('cluster_algo', 'None'),
                    params.get('cluster_k', 'None'),
                )
            
            # Store relevant metrics
            merged_results[key] = {
                'average': stats_data['average'],
                'std': stats_data['std']
            }
            
        # Save merged results
        with open(output_file, 'wb') as f:
            pickle.dump(merged_results, f)
        
        print(f"Merged {len(merged_results)} results into {output_file}")
    else: 
        with open(output_file, 'rb') as f:
            merged_results = pickle.load(f)

    return merged_results



def y_pred_check():
    y_pred_files = [f for f in os.listdir(Y_PRED_PKL_OUTDIR) if f.startswith('y_pred') and f.endswith('.pkl')]
    is_multi = False
    for y_pred_file in y_pred_files:
        y_pred_path = os.path.join(Y_PRED_PKL_OUTDIR, y_pred_file)
        with open(y_pred_path, 'rb') as f:
            y_preds = pickle.load(f)
        # Analyze the structure of y_preds
        y_test, predicted = y_preds[-1]

        
        if isinstance(predicted[0], (list, np.ndarray)):  # Multi-label
            # print("Detected multi-label classification.")
            is_multi = True
            plots.plot_prediction_comparison(y_test, predicted, EVAL_FUNC_METRIC, 
                    title=f"{'.'.join(y_pred_file.split('.')[:-1])}", )
            
        else:  # Single-label
            pass
    if is_multi:
        # Example usage:
        input_directory = Y_PRED_PKL_OUTDIR
        output_file = f"{Y_PRED_PKL_OUTDIR}/merged_y_prediction_stats.pkl"
        merged_data = merge_y_pred_pkl_files(input_directory, output_file)

        plots.plot_merged_y_pred_data(merged_data)

def farsight_res_vis():
    datasets = ['doc2vec', 'NMF_BOW', 'NMF_TW', 'NMF_BOW_SC', 'NMF_TW_SC']
    
    # Initialize the results dictionary to store metrics for each dataset
    results = {}
    output_file_pkl = '../outputs/metric_table1.pkl'
    output_file_txt = '../outputs/metric_table1.txt'
    if not os.path.exists(output_file_pkl):
        # Iterate over datasets
        for dataset in datasets:
            y_pred_files = [f for f in os.listdir(f'../outputs/{dataset}/fin/pkl_ypred') if f.startswith('y_pred_best_of') and f.endswith('.pkl')]
            param_files = [f for f in os.listdir(f'../outputs/{dataset}/fin/pkl_nn') if f.startswith('farsight_best_') and f.endswith('.pkl')]
            
            dataset_metrics = []  # Store metrics for each model in the dataset
            
            for y_pred_file in y_pred_files:
                y_pred_path = os.path.join(f'../outputs/{dataset}/fin/pkl_ypred', y_pred_file)
                print(f"Processing {y_pred_path}")
                
                # Load y_pred
                with open(y_pred_path, 'rb') as f:
                    y_preds = pickle.load(f)
                
                print(f"Number of predictions sets: {len(y_preds)}")
                
                # Initialize variables to accumulate metrics across all iterations
                all_label_accuracy = np.zeros(19)
                all_auc_roc = []
                all_f1 = []
                all_mcc = []
                all_auprc = []
                overall_accuracies = []

                # Loop through all prediction sets in y_preds
                for y_pred, y_actual in y_preds:
                    # Calculate metrics for the current set
                    label_accuracy, auc_roc, f1, mcc, auprc, overall_accuracy = calculate_fin_metrics(y_pred, y_actual)
                    
                    # Aggregate metrics for each label and overall accuracy
                    all_label_accuracy += np.array(label_accuracy)
                    all_auc_roc.append(auc_roc)
                    all_f1.append(f1)
                    all_mcc.append(mcc)
                    all_auprc.append(auprc)
                    overall_accuracies.append(overall_accuracy)
                
                # Average the metrics across all iterations in y_preds
                avg_label_accuracy = all_label_accuracy / len(y_preds)
                avg_auc_roc = np.nanmean(all_auc_roc, axis=0)  # Handle NaN values in AUC
                avg_f1 = np.mean(all_f1, axis=0)
                avg_mcc = np.mean(all_mcc, axis=0)
                avg_auprc = np.mean(all_auprc, axis=0)
                avg_overall_accuracy = np.mean(overall_accuracies)
                
                # Save metrics for this model
                model_name = y_pred_file.split('_')[-1].split('.')[0]  # 'bi-LSTM' from 'y_pred_best_of_bi-LSTM.pkl'
                model_metrics = {
                    "model": model_name,
                    "label_accuracy": avg_label_accuracy,
                    "auc_roc": avg_auc_roc,
                    "f1": avg_f1,
                    "mcc": avg_mcc,
                    "auprc": avg_auprc,
                    "overall_accuracy": avg_overall_accuracy
                }
                
                dataset_metrics.append(model_metrics)
            
            # Store dataset metrics
            results[dataset] = dataset_metrics
        # Example usage
        with open(output_file_pkl, "wb") as f:
            pickle.dump(results, f)
        
    else:
        with open(output_file_pkl, 'rb') as f:
            results = pickle.load(f)
    # data_plots.plot_mismatches(results)
    print_and_save_metrics(results, output_file_txt)
    for metric_name in ['label_accuracy', 'auc_roc', 'f1', 'mcc', 'auprc']:
        plots.plot_farsight_metric_with_std(results, metric_name)
    plots.plot_farsight_metric_with_std_agg(results)
    plots.plot_accuracy_grid_from_files(datasets)

          

def y_pred_farsight_check():
    dataset = 'doc2vec'
    y_pred_files = [f for f in os.listdir(f'../outputs/{dataset}') if f.startswith('y_pred') and f.endswith('.pkl')]
    is_multi = False
    for y_pred_file in y_pred_files:
        y_pred_path = os.path.join(f'../outputs/{dataset}/', y_pred_file)
        print(y_pred_path)
        with open(y_pred_path, 'rb') as f:
            y_preds = pickle.load(f)
        # Analyze the structure of y_preds
        y_test, predicted = y_preds[-1]
        
        if isinstance(predicted[0], (list, np.ndarray)):  # Multi-label
            # print("Detected multi-label classification.")
            is_multi = True
            plots.plot_prediction_comparison(y_test, predicted, EVAL_FUNC_METRIC, 
                    title=f"{'.'.join(y_pred_file.split('.')[:-1])}", )
            
        else:  # Single-label
            pass
    if is_multi:
        # Example usage:
        input_directory = Y_PRED_PKL_OUTDIR
        output_file = f"{Y_PRED_PKL_OUTDIR}/merged_y_prediction_stats.pkl"
        merged_data = merge_y_pred_pkl_files(input_directory, output_file)

        plots.plot_merged_y_pred_data(merged_data)


def check_etl():
    X, y = etl.get_data(DATASET_SELECTION,1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=GT_ID)
    test_data_etl_input_check(X, y, X_train, X_test, y_train, y_test, verbose = 1)
    etl.graph_raw_data(X, y)
    TestClusteringFunctions()
    print("======> Data verification complete")
    return X,y,X_train, X_test, y_train, y_test 

###############
def main(): 
    np.random.seed(GT_ID)
    X,y,X_train, X_test, y_train, y_test  = check_etl()
    # implement_farsight(X, y)
    # y_pred_farsight_check()
    farsight_res_vis()



if __name__ == "__main__":
    ###################
    print("PyTorch mps check: ",torch.backends.mps.is_available())
    print("PyTorch cuda check: ",torch.cuda.is_available())
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=K_FOLD_CV, shuffle=True, random_state=GT_ID)
    print(f"Torch will be running on {device}")
    ####################
    main()
    

