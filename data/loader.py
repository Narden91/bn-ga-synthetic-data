"""
Data loading utilities for the Bayesian Network synthetic data generator.
"""

import pandas as pd
import os


def load_data(filepath):
    """
    Load CSV data into a pandas DataFrame and handle missing values.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with missing values removed
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Original data shape: {df.shape}")
    
    # Display basic info about the data
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    # Drop rows with missing values
    df_clean = df.dropna()
    print(f"Data shape after removing missing values: {df_clean.shape}")
    
    return df_clean


def validate_data(df):
    """
    Validate the loaded data for common issues.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    if df.empty:
        print("Error: DataFrame is empty")
        return False
        
    if df.shape[1] < 2:
        print("Warning: DataFrame has less than 2 columns")
        
    # Check for non-numeric, non-categorical data
    for col in df.columns:
        if not (pd.api.types.is_numeric_dtype(df[col]) or 
                df[col].dtype.name == 'category' or
                pd.api.types.is_object_dtype(df[col])):
            print(f"Warning: Column {col} has unsupported data type: {df[col].dtype}")
    
    return True
