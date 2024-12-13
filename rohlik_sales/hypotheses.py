from scipy.stats import ttest_rel
from scipy.stats import ttest_1samp
import pickle

import os
from rohlik_sales.config import *

def run_dred_improves_purity_score_hypo_test(purity_scores, thresholds):
    """
    Run hypothesis tests for multiple thresholds and write all results to a single file.
    
    Args:
        purity_scores: Dictionary of purity scores
        thresholds: List of threshold values to test
    """
    # Define output paths - now just one file for all results
    purity_txt_path = f'{TXT_OUTDIR}/hypo_test_cluster_purity_scores_improve_with_dreduction_all.txt'
    pkl_path = f'{DREDUCED_CLUSTER_PKL_OUTDIR}/p_value_purity_results.pkl'

    if not os.path.exists(pkl_path):
        # Extract unique methods and clustering algorithms
        methods = sorted(set(key.split('_')[0] for key in purity_scores.keys()))
        cluster_algos = sorted(set(key.split('_')[2] for key in purity_scores.keys()))
        dimensions = sorted(set(int(key.split('_')[1][:-1]) for key in purity_scores.keys()))
        
        # Organize purity scores by method, dimension, and algorithm for easy access
        data = {algo: {method: {} for method in methods} for algo in cluster_algos}
        for key, score in purity_scores.items():
            method, dim, algo = key.split('_')[0], int(key.split('_')[1][:-1]), key.split('_')[2]
            data[algo][method][dim] = score
        all_results = {}

        # Open the file once for all thresholds
        with open(purity_txt_path, 'w') as file:
            file.write("Hypothesis Test Results: Improvement in Purity Score\n")
            file.write("=" * 80 + "\n\n")
            
            # Process each threshold
            for threshold in np.nditer(thresholds):
                p_value_results = {}
                
                file.write(f"\nTesting improvement threshold: {threshold * 100}%\n")
                file.write("=" * 50 + "\n")
                
                # Perform tests and collect results
                for algo in cluster_algos:
                    p_value_results[algo] = {}
                    
                    for method in methods:
                        baseline_score = data[algo][method].get(0, None)
                        if baseline_score is None:
                            continue
                            
                        improvements = []
                        for dim in dimensions:
                            if dim == 0 or dim not in data[algo][method]:
                                continue
                            observed_score = data[algo][method][dim]
                            if observed_score is not None:
                                improvement = (observed_score - baseline_score) / baseline_score
                                improvements.append(improvement)
                        
                        if not improvements:
                            continue
                            
                        # Perform one-sample t-test against the threshold
                        t_stat, p_value = ttest_1samp(improvements, threshold)
                        if t_stat > 0:
                            p_value /= 2  # One-sided test
                        else:
                            p_value = 1
                            
                        # Store in dictionary and write to file
                        avg_improvement = np.mean(improvements)
                        p_value_results[algo][method] = {
                            'avg_improvement': avg_improvement,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                        # Write results to text file
                        file.write(f"\nClustering: {algo}, DReduc: {method}\n")
                        file.write("-" * 50 + "\n")
                        file.write(f"Avg Improvement = {avg_improvement:.2%}\n")
                        file.write(f"P-value = {p_value:.4f}, Significant: {'Yes' if p_value < 0.05 else 'No'}\n")
                all_results[float(threshold)] = p_value_results
                file.write("\n" + "=" * 80 + "\n")
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(all_results, pkl_file)

from scipy.stats import ttest_rel

def evaluate_dreduced_vs_baseline(nn_dreduced, tag):
    """
    Evaluates each dimension reduction method against the baseline for significant improvement in accuracy and F1.
    Saves the results to a text file and a pickle file.

    Parameters:
    - nn_dreduced: Dictionary of dimension reduction results (e.g., nn_dreduced["RCA_47"]["mc_results"] contains accuracy, F1, runtime)
    - out_dir: Directory path where the results are saved
    """
    # Define output file paths
    txt_path = f"{TXT_OUTDIR}/{tag}_hypothesis_test_results.txt"
    pkl_path = f"{DREDUCED_PKL_OUTDIR}/{tag}_hypothesis_test_results.pkl"
    if not os.path.exists(pkl_path) and not os.path.exists(txt_path):

        # Extract baseline scores for accuracy and F1
        baseline_accuracies = [result[0] for result in nn_dreduced['baseline']['mc_results']]
        baseline_f1_scores = [result[1] for result in nn_dreduced['baseline']['mc_results']]

        # Compute average values for baseline
        baseline_avg_accuracy = np.mean(baseline_accuracies)
        baseline_avg_f1 = np.mean(baseline_f1_scores)

        # Dictionary to store all results for pickling
        results_summary = {}

        # Open a text file for writing
        with open(txt_path, "w") as file:
            file.write("Hypothesis Testing Results: Dimension Reduction vs Baseline\n")
            file.write("=" * 80 + "\n\n")
            file.write(f"Baseline Average Accuracy: {baseline_avg_accuracy:.4f}\n")
            file.write(f"Baseline Average F1 Score: {baseline_avg_f1:.4f}\n")
            file.write("-" * 80 + "\n\n")

            # Loop over each dimension reduction method
            for method_dim, result_data in nn_dreduced.items():
                if method_dim == "baseline":
                    continue  # Skip the baseline

                # Extract the accuracy and F1 scores for the current method and dimension
                accuracies = [result[0] for result in result_data['mc_results']]
                f1_scores = [result[1] for result in result_data['mc_results']]

                # Calculate average values for the current method
                avg_accuracy = np.mean(accuracies)
                avg_f1 = np.mean(f1_scores)

                # Perform paired t-tests between the current method and the baseline
                accuracy_p_value = ttest_rel(accuracies, baseline_accuracies).pvalue if len(accuracies) > 1 else np.nan
                f1_p_value = ttest_rel(f1_scores, baseline_f1_scores).pvalue if len(f1_scores) > 1 else np.nan

                # Determine significance and direction of improvement
                accuracy_significant = accuracy_p_value < 0.05 if not np.isnan(accuracy_p_value) else False
                f1_significant = f1_p_value < 0.05 if not np.isnan(f1_p_value) else False
                accuracy_improved = avg_accuracy > baseline_avg_accuracy
                f1_improved = avg_f1 > baseline_avg_f1

                # Store results in the dictionary for pickle saving
                results_summary[method_dim] = {
                    'accuracy_p_value': accuracy_p_value,
                    'accuracy_significant': accuracy_significant,
                    'accuracy_improved': accuracy_improved,
                    'avg_accuracy': avg_accuracy,
                    'f1_p_value': f1_p_value,
                    'f1_significant': f1_significant,
                    'f1_improved': f1_improved,
                    'avg_f1': avg_f1
                }

                # Write results to the text file
                file.write(f"Method-Dim: {method_dim}\n")
                file.write(f"  Average Accuracy: {avg_accuracy:.4f} vs Baseline: {baseline_avg_accuracy:.4f}, improvement ratio {(avg_accuracy-baseline_avg_accuracy)/baseline_avg_accuracy:.4f}\n")
                file.write(f"  Accuracy - p-value: {accuracy_p_value:.4f} - Significant: {'Yes' if accuracy_significant else 'No'} - {'Improved' if accuracy_improved else 'Not Improved'}\n")
                file.write(f"  Average F1 Score: {avg_f1:.4f} vs Baseline: {baseline_avg_f1:.4f}, improvement ratio {(avg_f1-baseline_avg_f1)/baseline_avg_f1:.4f}\n")
                file.write(f"  F1 Score - p-value: {f1_p_value:.4f} - Significant: {'Yes' if f1_significant else 'No'} - {'Improved' if f1_improved else 'Not Improved'}\n")
                file.write("-" * 50 + "\n")

        # Save the results summary as a pickle file
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(results_summary, pkl_file)

        print(f"Results saved to {txt_path} and {pkl_path}")