"""
Data preprocessing utilities for discretization and encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def discretize_data(df, n_bins=3, strategy='quantile'):
    """
    Discretize continuous variables into bins.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        n_bins (int): Number of bins for discretization
        strategy (str): Discretization strategy ('quantile' or 'uniform')
        
    Returns:
        pd.DataFrame: DataFrame with discretized variables
    """
    discretized = df.copy()
    discretization_info = {}
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if the column needs discretization (has enough unique values)
            unique_values = df[col].nunique()
            
            if unique_values > n_bins:
                print(f"Discretizing column '{col}' with {unique_values} unique values into {n_bins} bins")
                
                if strategy == 'quantile':
                    # Use quantile-based discretization
                    discretized[col], bin_edges = pd.qcut(
                        df[col], 
                        n_bins, 
                        labels=False, 
                        duplicates='drop',
                        retbins=True
                    )
                else:
                    # Use uniform width discretization
                    discretized[col], bin_edges = pd.cut(
                        df[col], 
                        n_bins, 
                        labels=False, 
                        retbins=True
                    )
                
                discretization_info[col] = {
                    'original_type': 'continuous',
                    'bin_edges': bin_edges,
                    'strategy': strategy
                }
            else:
                print(f"Column '{col}' has only {unique_values} unique values, treating as categorical")
                discretized[col] = df[col].astype('category').cat.codes
                discretization_info[col] = {
                    'original_type': 'categorical_numeric',
                    'unique_values': df[col].unique()
                }
        else:
            # Handle categorical variables
            print(f"Encoding categorical column '{col}'")
            le = LabelEncoder()
            discretized[col] = le.fit_transform(df[col].astype(str))
            discretization_info[col] = {
                'original_type': 'categorical',
                'label_encoder': le,
                'classes': le.classes_
            }
    
    # Ensure all values are integers
    discretized = discretized.astype(int)
    
    print(f"Discretization completed. Data shape: {discretized.shape}")
    print(f"Value ranges per column:")
    for col in discretized.columns:
        print(f"  {col}: {discretized[col].min()} - {discretized[col].max()} ({discretized[col].nunique()} unique values)")
    
    return discretized, discretization_info


def encode_categorical_variables(df):
    """
    Encode categorical variables numerically.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        tuple: (encoded_df, encoding_info)
    """
    encoded = df.copy()
    encoding_info = {}
    
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or df[col].dtype.name == 'category':
            le = LabelEncoder()
            encoded[col] = le.fit_transform(df[col].astype(str))
            encoding_info[col] = {
                'encoder': le,
                'classes': le.classes_
            }
            print(f"Encoded column '{col}': {len(le.classes_)} unique categories")
    
    return encoded, encoding_info


def validate_discretized_data(df):
    """
    Validate that discretized data is suitable for Bayesian Network learning.
    
    Args:
        df (pd.DataFrame): Discretized DataFrame
        
    Returns:
        bool: True if data is valid
    """
    # Check that all columns are numeric
    for col in df.columns:
        if not pd.api.types.is_integer_dtype(df[col]):
            print(f"Error: Column {col} is not integer type after discretization")
            return False
    
    # Check for reasonable number of states per variable
    for col in df.columns:
        unique_vals = df[col].nunique()
        if unique_vals < 2:
            print(f"Warning: Column {col} has only {unique_vals} unique value(s)")
        elif unique_vals > 10:
            print(f"Warning: Column {col} has {unique_vals} unique values (may be too many for BN)")
    
    # Check that values start from 0 (required for pgmpy)
    for col in df.columns:
        min_val = df[col].min()
        if min_val != 0:
            print(f"Error: Column {col} minimum value is {min_val}, should be 0")
            return False
    
    print("Discretized data validation passed")
    return True


def prepare_data_for_bn(df, n_bins=3, strategy='quantile'):
    """
    Complete preprocessing pipeline for Bayesian Network learning.
    
    Args:
        df (pd.DataFrame): Raw input DataFrame
        n_bins (int): Number of bins for discretization
        strategy (str): Discretization strategy
        
    Returns:
        tuple: (processed_df, preprocessing_info)
    """
    print("Starting data preprocessing for Bayesian Network...")
    
    # Step 1: Discretize continuous variables
    discretized_df, discretization_info = discretize_data(df, n_bins, strategy)
    
    # Step 2: Validate the processed data
    if not validate_discretized_data(discretized_df):
        raise ValueError("Data validation failed after preprocessing")
    
    preprocessing_info = {
        'discretization': discretization_info,
        'original_shape': df.shape,
        'processed_shape': discretized_df.shape,
        'n_bins': n_bins,
        'strategy': strategy
    }
    
    print("Data preprocessing completed successfully")
    return discretized_df, preprocessing_info
