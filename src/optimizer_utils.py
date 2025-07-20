"""
Shared utilities for GA and CMA-ES optimizers.

This module contains common functions used by both optimizers
to avoid code duplication.
"""

import numpy as np
import pandas as pd
from typing import Dict


def determine_threshold(anomaly_scores: np.ndarray, threshold_percentile: float) -> float:
    """
    Determine anomaly threshold using percentile method.
    
    Args:
        anomaly_scores: Array of anomaly scores
        threshold_percentile: Percentile for threshold (1-10)
        
    Returns:
        float: Threshold value
    """
    return np.percentile(anomaly_scores, 100 - threshold_percentile)


def normalize_bn_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize BN weights to sum to 1.0.
    
    Args:
        weights: Raw weight values
        
    Returns:
        np.ndarray: Normalized weights
    """
    weights = np.maximum(weights, 1e-6)  # Avoid zero weights
    return weights / np.sum(weights)


def decode_individual_parameters(individual: list, n_bn_groups: int) -> Dict:
    """
    Decode individual chromosome into parameters dictionary.
    
    Args:
        individual: Individual chromosome (weights + threshold)
        n_bn_groups: Number of BN groups
        
    Returns:
        Dict: Parameters including normalized bn_weights and threshold_percentile
    """
    # Extract BN weights and threshold
    raw_weights = np.array(individual[:n_bn_groups])
    threshold_percentile = individual[n_bn_groups]
    
    # Normalize weights
    bn_weights = normalize_bn_weights(raw_weights)
    
    return {
        'bn_weights': bn_weights,
        'threshold_percentile': threshold_percentile,
        'aggregation_method': 'weighted'
    }
