import pandas as pd
import numpy as np
from typing import Optional
import os

class DataLoader:
    """
    Handles data loading and basic validation.
    """
    
    def __init__(self):
        """Initialize the data loader."""
        pass
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file with validation.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data is invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            # Load the CSV file
            data = pd.read_csv(file_path)
            
            # Basic validation
            self._validate_data(data)
            
            print(f"✅ Successfully loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
            
            # Display basic info
            self._display_data_info(data)
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate the loaded data.
        
        Args:
            data (pd.DataFrame): Data to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if data.empty:
            raise ValueError("Data file is empty")
        
        if data.shape[1] < 2:
            raise ValueError("Data must have at least 2 columns")
        
        if data.shape[0] < 10:
            raise ValueError("Data must have at least 10 rows")
    
    def _display_data_info(self, data: pd.DataFrame) -> None:
        """
        Display basic information about the data.
        
        Args:
            data (pd.DataFrame): Data to analyze
        """
        print(f"   Data shape: {data.shape}")
        print(f"   Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Missing values info
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            missing_cols = missing_counts[missing_counts > 0]
            print(f"   Missing values in {len(missing_cols)} columns")
            print(f"   Total missing: {missing_counts.sum()} ({missing_counts.sum()/data.size*100:.1f}%)")
        else:
            print("   No missing values detected")
        
        # Data types info
        numeric_cols = data.select_dtypes(include=[np.number]).shape[1]
        categorical_cols = data.select_dtypes(include=['object', 'category']).shape[1]
        print(f"   Numeric columns: {numeric_cols}")
        print(f"   Categorical columns: {categorical_cols}")
    
    def save_data(self, data: pd.DataFrame, file_path: str) -> None:
        """
        Save processed data to CSV file.
        
        Args:
            data (pd.DataFrame): Data to save
            file_path (str): Output file path
        """
        try:
            data.to_csv(file_path, index=False)
            print(f"✅ Data saved to: {file_path}")
        except Exception as e:
            print(f"❌ Error saving data: {str(e)}")
    
    def get_data_summary(self, data: pd.DataFrame) -> dict:
        """
        Get comprehensive data summary.
        
        Args:
            data (pd.DataFrame): Data to summarize
            
        Returns:
            dict: Data summary statistics
        """
        summary = {
            'shape': data.shape,
            'memory_mb': data.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': data.isnull().sum().sum(),
            'missing_percentage': data.isnull().sum().sum() / data.size * 100,
            'numeric_columns': data.select_dtypes(include=[np.number]).shape[1],
            'categorical_columns': data.select_dtypes(include=['object', 'category']).shape[1],
            'duplicate_rows': data.duplicated().sum(),
            'columns_with_missing': (data.isnull().sum() > 0).sum(),
            'high_missing_columns': (data.isnull().sum() > len(data) * 0.5).sum()
        }
        
        return summary
