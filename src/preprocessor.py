import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Optional, Tuple
import warnings

class DataPreprocessor:
    """
    Handles data preprocessing including missing values, encoding, and scaling.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the preprocessor.
        
        Args:
            config (Dict): Preprocessing configuration
        """
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.dropped_columns = []
        self.feature_names = []
        
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            data (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        print("   Starting preprocessing pipeline...")
        processed_data = data.copy()
        
        # Step 1: Handle missing values
        processed_data = self._handle_missing_values(processed_data)
        
        # Step 2: Handle categorical variables
        processed_data = self._handle_categorical_variables(processed_data)
        
        # Step 3: Handle infinite and extreme values
        processed_data = self._handle_infinite_values(processed_data)
        
        # Step 4: Scale features if requested
        if self.config.get('scale_features', True):
            processed_data = self._scale_features(processed_data)
        
        # Step 5: Final validation
        processed_data = self._final_validation(processed_data)
        
        self.feature_names = list(processed_data.columns)
        print(f"   Preprocessing completed: {processed_data.shape[1]} features retained")
        
        return processed_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        print("     Handling missing values...")
        
        # Get missing value statistics
        missing_stats = data.isnull().sum()
        missing_percentage = missing_stats / len(data)
        
        # Drop columns with excessive missing values
        threshold = self.config.get('missing_threshold', 0.5)
        cols_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()
        
        if cols_to_drop:
            print(f"     Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing values")
            self.dropped_columns.extend(cols_to_drop)
            data = data.drop(columns=cols_to_drop)
        
        # Impute remaining missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # Numeric imputation (median)
        if len(numeric_cols) > 0 and data[numeric_cols].isnull().any().any():
            self.imputers['numeric'] = SimpleImputer(strategy='median')
            data[numeric_cols] = self.imputers['numeric'].fit_transform(data[numeric_cols])
            print(f"     Imputed numeric missing values in {len(numeric_cols)} columns")
        
        # Categorical imputation (mode)
        if len(categorical_cols) > 0 and data[categorical_cols].isnull().any().any():
            self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
            data[categorical_cols] = self.imputers['categorical'].fit_transform(data[categorical_cols])
            print(f"     Imputed categorical missing values in {len(categorical_cols)} columns")
        
        return data
    
    def _handle_categorical_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with encoded categorical variables
        """
        if not self.config.get('handle_categorical', True):
            return data
        
        print("     Encoding categorical variables...")
        
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            print("     No categorical variables found")
            return data
        
        for col in categorical_cols:
            try:
                # Use label encoding for categorical variables
                self.encoders[col] = LabelEncoder()
                data[col] = self.encoders[col].fit_transform(data[col].astype(str))
                print(f"     Encoded column: {col}")
            except Exception as e:
                print(f"     Warning: Could not encode column {col}: {str(e)}")
                # Drop problematic categorical column
                data = data.drop(columns=[col])
                self.dropped_columns.append(col)
        
        return data
    
    def _handle_infinite_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle infinite and extreme values.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with infinite values handled
        """
        print("     Handling infinite and extreme values...")
        
        # Replace infinite values with NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Check for any new NaN values and impute them
        if data.isnull().any().any():
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Use median imputation for new NaN values
                imputer = SimpleImputer(strategy='median')
                data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        
        # Cap extreme values using IQR method
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Cap extreme values
            extreme_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            if extreme_count > 0:
                data[col] = np.clip(data[col], lower_bound, upper_bound)
                if extreme_count > 10:  # Only report if significant
                    print(f"     Capped {extreme_count} extreme values in {col}")
        
        return data
    
    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Scaled data
        """
        print("     Scaling features...")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("     No numeric columns to scale")
            return data
        
        # Use StandardScaler (z-score normalization)
        self.scalers['standard'] = StandardScaler()
        data[numeric_cols] = self.scalers['standard'].fit_transform(data[numeric_cols])
        
        print(f"     Scaled {len(numeric_cols)} numeric features")
        
        return data
    
    def _final_validation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Final validation and cleaning.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Validated data
        """
        print("     Final validation...")
        
        # Check for any remaining NaN or infinite values
        if data.isnull().any().any():
            print("     Warning: NaN values still present after preprocessing")
            # Drop rows with any NaN values as last resort
            initial_rows = len(data)
            data = data.dropna()
            dropped_rows = initial_rows - len(data)
            if dropped_rows > 0:
                print(f"     Dropped {dropped_rows} rows with remaining NaN values")
        
        # Check for infinite values
        if np.isinf(data.select_dtypes(include=[np.number])).any().any():
            print("     Warning: Infinite values still present")
            data = data.replace([np.inf, -np.inf], 0)
        
        # Check for constant columns (no variance)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        constant_cols = []
        for col in numeric_cols:
            if data[col].std() == 0:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"     Dropping {len(constant_cols)} constant columns")
            data = data.drop(columns=constant_cols)
            self.dropped_columns.extend(constant_cols)
        
        # Ensure we have enough features
        if data.shape[1] < 2:
            raise ValueError("Not enough features remaining after preprocessing")
        
        return data
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get summary of preprocessing steps applied.
        
        Returns:
            Dict: Summary of preprocessing
        """
        return {
            'dropped_columns': self.dropped_columns,
            'n_dropped_columns': len(self.dropped_columns),
            'remaining_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'scalers_used': list(self.scalers.keys()),
            'encoders_used': list(self.encoders.keys()),
            'imputers_used': list(self.imputers.keys())
        }
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessors.
        
        Args:
            data (pd.DataFrame): New data to transform
            
        Returns:
            pd.DataFrame: Transformed data
        """
        processed_data = data.copy()
        
        # Apply same column drops
        cols_to_drop = [col for col in self.dropped_columns if col in processed_data.columns]
        if cols_to_drop:
            processed_data = processed_data.drop(columns=cols_to_drop)
        
        # Apply encoders
        for col, encoder in self.encoders.items():
            if col in processed_data.columns:
                try:
                    processed_data[col] = encoder.transform(processed_data[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    processed_data[col] = 0  # Default value for unseen categories
        
        # Apply imputers
        if 'numeric' in self.imputers:
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                processed_data[numeric_cols] = self.imputers['numeric'].transform(processed_data[numeric_cols])
        
        # Apply scalers
        if 'standard' in self.scalers:
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                processed_data[numeric_cols] = self.scalers['standard'].transform(processed_data[numeric_cols])
        
        return processed_data
