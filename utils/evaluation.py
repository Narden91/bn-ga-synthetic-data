"""
Evaluation utilities for comparing real and synthetic data.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy, ks_2samp
from scipy.spatial.distance import jensenshannon
import warnings


def compute_marginal_distributions(data):
    """
    Compute marginal distributions for all variables.
    
    Args:
        data (pd.DataFrame): Input data
        
    Returns:
        dict: Marginal distributions for each variable
    """
    marginals = {}
    
    for col in data.columns:
        value_counts = data[col].value_counts(normalize=True).sort_index()
        marginals[col] = value_counts
    
    return marginals


def compute_pairwise_mutual_info(data):
    """
    Compute pairwise mutual information between all variable pairs.
    
    Args:
        data (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Mutual information matrix
    """
    variables = list(data.columns)
    n_vars = len(variables)
    mi_matrix = pd.DataFrame(np.zeros((n_vars, n_vars)), 
                           index=variables, columns=variables)
    
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i <= j:  # Compute only upper triangle (MI is symmetric)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mi = mutual_info_score(data[var1], data[var2])
                mi_matrix.loc[var1, var2] = mi
                mi_matrix.loc[var2, var1] = mi  # Fill symmetric entry
    
    return mi_matrix


def compute_kl_divergence(real_dist, synthetic_dist, epsilon=1e-10):
    """
    Compute KL divergence between two probability distributions.
    
    Args:
        real_dist (pd.Series): Real data distribution
        synthetic_dist (pd.Series): Synthetic data distribution
        epsilon (float): Smoothing parameter to avoid log(0)
        
    Returns:
        float: KL divergence
    """
    # Align indices and fill missing values
    all_values = set(real_dist.index) | set(synthetic_dist.index)
    real_aligned = real_dist.reindex(all_values, fill_value=0)
    synth_aligned = synthetic_dist.reindex(all_values, fill_value=0)
    
    # Add smoothing
    real_smooth = real_aligned + epsilon
    synth_smooth = synth_aligned + epsilon
    
    # Normalize to ensure they sum to 1
    real_smooth = real_smooth / real_smooth.sum()
    synth_smooth = synth_smooth / synth_smooth.sum()
    
    # Compute KL divergence
    kl_div = entropy(real_smooth, synth_smooth)
    
    return kl_div


def compute_jensen_shannon_divergence(real_dist, synthetic_dist):
    """
    Compute Jensen-Shannon divergence between two distributions.
    
    Args:
        real_dist (pd.Series): Real data distribution
        synthetic_dist (pd.Series): Synthetic data distribution
        
    Returns:
        float: Jensen-Shannon divergence
    """
    # Align distributions
    all_values = set(real_dist.index) | set(synthetic_dist.index)
    real_aligned = real_dist.reindex(all_values, fill_value=0).values
    synth_aligned = synthetic_dist.reindex(all_values, fill_value=0).values
    
    # Normalize
    real_aligned = real_aligned / real_aligned.sum()
    synth_aligned = synth_aligned / synth_aligned.sum()
    
    # Compute JS divergence
    js_div = jensenshannon(real_aligned, synth_aligned) ** 2
    
    return js_div


def compute_kolmogorov_smirnov_test(real_data, synthetic_data):
    """
    Compute Kolmogorov-Smirnov test for each variable.
    
    Args:
        real_data (pd.DataFrame): Real data
        synthetic_data (pd.DataFrame): Synthetic data
        
    Returns:
        dict: KS test results for each variable
    """
    ks_results = {}
    
    for col in real_data.columns:
        if col in synthetic_data.columns:
            try:
                statistic, p_value = ks_2samp(real_data[col], synthetic_data[col])
                ks_results[col] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'similar': p_value > 0.05  # Not significantly different
                }
            except Exception as e:
                ks_results[col] = {
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'similar': False,
                    'error': str(e)
                }
    
    return ks_results


def compute_comprehensive_evaluation(real_data, synthetic_data):
    """
    Compute comprehensive evaluation metrics comparing real and synthetic data.
    
    Args:
        real_data (pd.DataFrame): Real data
        synthetic_data (pd.DataFrame): Synthetic data
        
    Returns:
        dict: Comprehensive evaluation results
    """
    print("Computing comprehensive evaluation metrics...")
    
    results = {
        'marginal_comparisons': {},
        'mutual_info_comparison': {},
        'distributional_tests': {},
        'summary_metrics': {}
    }
    
    # 1. Marginal distribution comparisons
    real_marginals = compute_marginal_distributions(real_data)
    synth_marginals = compute_marginal_distributions(synthetic_data)
    
    for var in real_data.columns:
        if var in synth_marginals:
            kl_div = compute_kl_divergence(real_marginals[var], synth_marginals[var])
            js_div = compute_jensen_shannon_divergence(real_marginals[var], synth_marginals[var])
            
            results['marginal_comparisons'][var] = {
                'kl_divergence': kl_div,
                'js_divergence': js_div
            }
    
    # 2. Mutual information comparison
    real_mi = compute_pairwise_mutual_info(real_data)
    synth_mi = compute_pairwise_mutual_info(synthetic_data)
    
    # Compare MI matrices
    mi_diff = np.abs(real_mi.values - synth_mi.values)
    results['mutual_info_comparison'] = {
        'mean_absolute_difference': np.mean(mi_diff),
        'max_absolute_difference': np.max(mi_diff),
        'correlation': np.corrcoef(real_mi.values.flatten(), synth_mi.values.flatten())[0, 1]
    }
    
    # 3. Statistical tests
    results['distributional_tests'] = compute_kolmogorov_smirnov_test(real_data, synthetic_data)
    
    # 4. Summary metrics
    avg_kl = np.mean([comp['kl_divergence'] for comp in results['marginal_comparisons'].values()])
    avg_js = np.mean([comp['js_divergence'] for comp in results['marginal_comparisons'].values()])
    
    similar_distributions = sum([test['similar'] for test in results['distributional_tests'].values()])
    total_distributions = len(results['distributional_tests'])
    
    results['summary_metrics'] = {
        'average_kl_divergence': avg_kl,
        'average_js_divergence': avg_js,
        'mutual_info_correlation': results['mutual_info_comparison']['correlation'],
        'similar_distributions_ratio': similar_distributions / total_distributions if total_distributions > 0 else 0
    }
    
    print("Comprehensive evaluation completed")
    return results


def compute_fitness_score(real_data, synthetic_data, weights=None):
    """
    Compute a single fitness score for genetic algorithm optimization.
    
    Args:
        real_data (pd.DataFrame): Real data
        synthetic_data (pd.DataFrame): Synthetic data
        weights (dict): Weights for different metrics
        
    Returns:
        float: Fitness score (higher is better)
    """
    if weights is None:
        weights = {
            'marginal_weight': 0.4,
            'mutual_info_weight': 0.3,
            'ks_test_weight': 0.3
        }
    
    try:
        # 1. Marginal distribution score (inverse of average JS divergence)
        real_marginals = compute_marginal_distributions(real_data)
        synth_marginals = compute_marginal_distributions(synthetic_data)
        
        js_divergences = []
        for var in real_data.columns:
            if var in synth_marginals:
                js_div = compute_jensen_shannon_divergence(real_marginals[var], synth_marginals[var])
                js_divergences.append(js_div)
        
        marginal_score = 1.0 / (1.0 + np.mean(js_divergences)) if js_divergences else 0
        
        # 2. Mutual information score
        real_mi = compute_pairwise_mutual_info(real_data)
        synth_mi = compute_pairwise_mutual_info(synthetic_data)
        
        mi_correlation = np.corrcoef(real_mi.values.flatten(), synth_mi.values.flatten())[0, 1]
        mi_score = max(0, mi_correlation)  # Ensure non-negative
        
        # 3. KS test score (proportion of similar distributions)
        ks_results = compute_kolmogorov_smirnov_test(real_data, synthetic_data)
        similar_count = sum([test.get('similar', False) for test in ks_results.values()])
        ks_score = similar_count / len(ks_results) if ks_results else 0
        
        # Combine scores
        fitness = (weights['marginal_weight'] * marginal_score +
                  weights['mutual_info_weight'] * mi_score +
                  weights['ks_test_weight'] * ks_score)
        
        return fitness
        
    except Exception as e:
        print(f"Error computing fitness: {e}")
        return 0.0


def print_evaluation_summary(evaluation_results):
    """
    Print a human-readable summary of evaluation results.
    
    Args:
        evaluation_results (dict): Results from compute_comprehensive_evaluation
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    summary = evaluation_results['summary_metrics']
    
    print(f"Average KL Divergence: {summary['average_kl_divergence']:.4f}")
    print(f"Average JS Divergence: {summary['average_js_divergence']:.4f}")
    print(f"Mutual Info Correlation: {summary['mutual_info_correlation']:.4f}")
    print(f"Similar Distributions: {summary['similar_distributions_ratio']:.1%}")
    
    print("\nPer-Variable Marginal Comparisons:")
    print("-" * 40)
    for var, comp in evaluation_results['marginal_comparisons'].items():
        print(f"{var:15} | KL: {comp['kl_divergence']:.4f} | JS: {comp['js_divergence']:.4f}")
    
    print("\nDistribution Similarity Tests (KS):")
    print("-" * 40)
    for var, test in evaluation_results['distributional_tests'].items():
        similar_str = "✓" if test.get('similar', False) else "✗"
        p_val = test.get('p_value', np.nan)
        print(f"{var:15} | {similar_str} | p-value: {p_val:.4f}")
    
    print("="*60)


def save_evaluation_results(evaluation_results, filepath):
    """
    Save evaluation results to a file.
    
    Args:
        evaluation_results (dict): Evaluation results
        filepath (str): Output file path
    """
    try:
        # Convert to a more serializable format
        import json
        
        # Handle numpy types that aren't JSON serializable
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return obj
        
        # Deep convert the results
        serializable_results = json.loads(
            json.dumps(evaluation_results, default=convert_numpy)
        )
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Evaluation results saved to: {filepath}")
        
    except Exception as e:
        print(f"Error saving evaluation results: {e}")


def compare_multiple_models(real_data, synthetic_datasets, model_names=None):
    """
    Compare multiple synthetic datasets against real data.
    
    Args:
        real_data (pd.DataFrame): Real data
        synthetic_datasets (list): List of synthetic datasets
        model_names (list): Names for each model
        
    Returns:
        dict: Comparison results for all models
    """
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(synthetic_datasets))]
    
    comparison_results = {}
    
    for i, (synthetic_data, model_name) in enumerate(zip(synthetic_datasets, model_names)):
        print(f"Evaluating {model_name}...")
        evaluation = compute_comprehensive_evaluation(real_data, synthetic_data)
        comparison_results[model_name] = evaluation
    
    # Create summary comparison
    summary_comparison = pd.DataFrame({
        name: results['summary_metrics'] 
        for name, results in comparison_results.items()
    }).T
    
    print("\nModel Comparison Summary:")
    print(summary_comparison)
    
    return comparison_results, summary_comparison
