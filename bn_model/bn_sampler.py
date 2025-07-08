"""
Bayesian Network sampling utilities for generating synthetic data.
"""

import pandas as pd
import numpy as np
from pgmpy.sampling import BayesianModelSampling
import warnings


def sample_bn_data(model, n_samples=1000, method='forward_sample'):
    """
    Sample synthetic data from a Bayesian Network.
    
    Args:
        model (BayesianNetwork): Trained BN model
        n_samples (int): Number of samples to generate
        method (str): Sampling method ('forward_sample' or 'rejection_sample')
        
    Returns:
        pd.DataFrame: Synthetic data samples
    """
    print(f"Generating {n_samples} synthetic samples using {method}")
    
    try:
        # Initialize sampler
        sampler = BayesianModelSampling(model)
        
        # Generate samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if method == 'forward_sample':
                synthetic_data = sampler.forward_sample(size=n_samples)
            elif method == 'rejection_sample':
                # Rejection sampling (slower but more accurate for complex dependencies)
                synthetic_data = sampler.rejection_sample(size=n_samples)
            else:
                raise ValueError(f"Unsupported sampling method: {method}")
        
        # Convert to DataFrame if it's not already
        if not isinstance(synthetic_data, pd.DataFrame):
            synthetic_data = pd.DataFrame(synthetic_data)
        
        # Ensure the order of columns matches the original data
        synthetic_data = synthetic_data[list(model.nodes())]
        
        print(f"Successfully generated synthetic data with shape {synthetic_data.shape}")
        
        return synthetic_data
        
    except Exception as e:
        print(f"Error in sampling: {e}")
        print("Generating random data as fallback")
        return generate_random_fallback_data(model, n_samples)


def generate_random_fallback_data(model, n_samples):
    """
    Generate random data as fallback when sampling fails.
    
    Args:
        model (BayesianNetwork): BN model (for structure info)
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Random synthetic data
    """
    print("Generating random fallback data...")
    
    data = {}
    
    for node in model.nodes():
        # Get cardinality from CPD
        cpd = model.get_cpds(node)
        cardinality = cpd.cardinality[0]
        
        # Generate random integers from 0 to cardinality-1
        data[node] = np.random.randint(0, cardinality, size=n_samples)
    
    return pd.DataFrame(data)


def sample_with_evidence(model, evidence, n_samples=1000):
    """
    Sample from BN with evidence (conditional sampling).
    
    Args:
        model (BayesianNetwork): Trained BN model
        evidence (dict): Evidence variables and their values
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Conditional synthetic samples
    """
    print(f"Generating {n_samples} conditional samples with evidence: {evidence}")
    
    try:
        sampler = BayesianModelSampling(model)
        
        # Use rejection sampling for conditional generation
        synthetic_data = sampler.rejection_sample(
            evidence=evidence,
            size=n_samples
        )
        
        print(f"Successfully generated conditional samples with shape {synthetic_data.shape}")
        return synthetic_data
        
    except Exception as e:
        print(f"Error in conditional sampling: {e}")
        return None


def validate_synthetic_data(synthetic_data, original_data):
    """
    Validate that synthetic data has reasonable properties.
    
    Args:
        synthetic_data (pd.DataFrame): Generated synthetic data
        original_data (pd.DataFrame): Original training data
        
    Returns:
        dict: Validation results
    """
    validation_results = {
        'shape_match': False,
        'column_match': False,
        'value_ranges_match': {},
        'distribution_sanity': {}
    }
    
    # Check shape compatibility
    if synthetic_data.shape[1] == original_data.shape[1]:
        validation_results['shape_match'] = True
    
    # Check column names
    if set(synthetic_data.columns) == set(original_data.columns):
        validation_results['column_match'] = True
    
    # Check value ranges for each column
    for col in original_data.columns:
        if col in synthetic_data.columns:
            orig_min, orig_max = original_data[col].min(), original_data[col].max()
            synth_min, synth_max = synthetic_data[col].min(), synthetic_data[col].max()
            
            # Check if synthetic data stays within original bounds
            within_bounds = (synth_min >= orig_min) and (synth_max <= orig_max)
            validation_results['value_ranges_match'][col] = within_bounds
            
            # Check distribution sanity (not all same value)
            unique_vals = synthetic_data[col].nunique()
            validation_results['distribution_sanity'][col] = unique_vals > 1
    
    # Print validation summary
    print("Synthetic data validation results:")
    print(f"  Shape match: {validation_results['shape_match']}")
    print(f"  Column match: {validation_results['column_match']}")
    
    valid_ranges = sum(validation_results['value_ranges_match'].values())
    total_cols = len(validation_results['value_ranges_match'])
    print(f"  Value ranges valid: {valid_ranges}/{total_cols} columns")
    
    valid_distributions = sum(validation_results['distribution_sanity'].values())
    print(f"  Non-degenerate distributions: {valid_distributions}/{total_cols} columns")
    
    return validation_results


def save_synthetic_data(synthetic_data, filepath, include_timestamp=True):
    """
    Save synthetic data to CSV file.
    
    Args:
        synthetic_data (pd.DataFrame): Synthetic data to save
        filepath (str): Output file path
        include_timestamp (bool): Whether to include timestamp in filename
        
    Returns:
        str: Actual file path used
    """
    if include_timestamp:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path, ext = filepath.rsplit('.', 1)
        actual_filepath = f"{base_path}_{timestamp}.{ext}"
    else:
        actual_filepath = filepath
    
    try:
        synthetic_data.to_csv(actual_filepath, index=False)
        print(f"Synthetic data saved to: {actual_filepath}")
        return actual_filepath
        
    except Exception as e:
        print(f"Error saving synthetic data: {e}")
        return None


def compare_distributions(original_data, synthetic_data, columns=None):
    """
    Compare distributions between original and synthetic data.
    
    Args:
        original_data (pd.DataFrame): Original data
        synthetic_data (pd.DataFrame): Synthetic data
        columns (list): Columns to compare (all if None)
        
    Returns:
        dict: Distribution comparison results
    """
    if columns is None:
        columns = original_data.columns
    
    comparison_results = {}
    
    for col in columns:
        if col in synthetic_data.columns:
            # Get value counts (normalized)
            orig_dist = original_data[col].value_counts(normalize=True).sort_index()
            synth_dist = synthetic_data[col].value_counts(normalize=True).sort_index()
            
            # Align indices
            all_values = set(orig_dist.index) | set(synth_dist.index)
            orig_aligned = orig_dist.reindex(all_values, fill_value=0)
            synth_aligned = synth_dist.reindex(all_values, fill_value=0)
            
            # Calculate KL divergence (with smoothing to avoid log(0))
            epsilon = 1e-10
            orig_smooth = orig_aligned + epsilon
            synth_smooth = synth_aligned + epsilon
            
            kl_div = sum(orig_smooth * np.log(orig_smooth / synth_smooth))
            
            comparison_results[col] = {
                'kl_divergence': kl_div,
                'original_entropy': -sum(orig_smooth * np.log(orig_smooth)),
                'synthetic_entropy': -sum(synth_smooth * np.log(synth_smooth))
            }
    
    return comparison_results
