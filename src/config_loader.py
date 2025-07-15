import os
import yaml
from typing import Dict, Any, Optional
import warnings


class ConfigLoader:
    """
    Loads and manages YAML configuration files for the anomaly detection system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config loader.
        
        Args:
            config_path (str, optional): Path to the YAML config file.
                                       Defaults to 'config.yaml' in the project root.
        """
        self.config_path = config_path or "config.yaml"
        self.config = {}
        
    def load_config(self, override_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file with optional overrides.
        
        Args:
            override_config (dict, optional): Configuration overrides to apply
            
        Returns:
            Dict[str, Any]: Complete configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        # Load base configuration from YAML file
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
                
            if self.config is None:
                self.config = {}
                
            print(f"✅ Configuration loaded from: {self.config_path}")
            
        except FileNotFoundError:
            print(f"⚠️  Config file not found: {self.config_path}")
            print("   Using default configuration...")
            self.config = self._get_default_config()
            
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML config: {e}")
            print("   Using default configuration...")
            self.config = self._get_default_config()
            
        except Exception as e:
            print(f"❌ Unexpected error loading config: {e}")
            print("   Using default configuration...")
            self.config = self._get_default_config()
        
        # Apply environment variable substitutions
        self.config = self._substitute_env_vars(self.config)
        
        # Apply override configuration if provided
        if override_config:
            self.config = self._merge_configs(self.config, override_config)
            print("✅ Override configuration applied")
        
        # Validate configuration
        self._validate_config()
        
        return self.config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration as fallback.
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            'data': {
                'path': "data/Dati_wallbox_aggregati.csv"
            },
            'preprocessing': {
                'missing_threshold': 0.5,
                'scale_features': True,
                'handle_categorical': True
            },
            'feature_grouping': {
                'group_size': 15,
                'strategy': 'correlation'
            },
            'bayesian_network': {
                'structure_learning': 'naive_bayes',
                'discretization_bins': 5,
                'max_parents': 3
            },
            'anomaly_detection': {
                'threshold_percentile': 5.0,
                'threshold_method': 'percentile',
                'aggregation_method': 'mean',
                'use_zscore_transformation': True,
                'use_rank_transformation': False
            },
            'genetic_algorithm': {
                'population_size': 50,
                'generations': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'tournament_size': 3
            },
            'cmaes_algorithm': {
                'population_size': None,
                'generations': 100,
                'initial_sigma': 0.3
            },
            'optimization': {
                'algorithm': 'genetic',
                'use_optimization': True
            },
            'visualization': {
                'save_plots': True,
                'show_plots': False,
                'plot_format': 'png',
                'dpi': 300
            },
            'output': {
                'results_dir': 'results',
                'save_detailed_results': True,
                'save_intermediate_results': False,
                'compression': False
            },
            'logging': {
                'level': 'INFO',
                'save_to_file': False,
                'verbose': True
            },
            'advanced': {
                'random_seed': 42,
                'parallel_processing': False,
                'memory_limit_gb': 8.0,
                'early_stopping': True,
                'convergence_threshold': 1e-6
            }
        }
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute environment variables in configuration values.
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            dict: Configuration with environment variables substituted
        """
        def substitute_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {key: substitute_recursive(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                # Extract environment variable name
                env_var = obj[2:-1]
                default_value = None
                
                # Check if default value is specified
                if ':' in env_var:
                    env_var, default_value = env_var.split(':', 1)
                
                # Get environment variable value
                return os.getenv(env_var, default_value)
            else:
                return obj
        
        result = substitute_recursive(config)
        return result if isinstance(result, dict) else config
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base_config (dict): Base configuration
            override_config (dict): Override configuration
            
        Returns:
            dict: Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged
    
    def _validate_config(self):
        """
        Validate the loaded configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate required sections
        required_sections = [
            'preprocessing', 'feature_grouping', 'bayesian_network', 
            'anomaly_detection', 'optimization'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate specific parameters
        self._validate_numeric_ranges()
        self._validate_string_choices()
        
        print("✅ Configuration validation passed")
    
    def _validate_numeric_ranges(self):
        """Validate numeric parameters are within acceptable ranges."""
        validations = [
            ('preprocessing.missing_threshold', 0.0, 1.0),
            ('feature_grouping.group_size', 2, 50),
            ('bayesian_network.discretization_bins', 2, 20),
            ('bayesian_network.max_parents', 1, 10),
            ('anomaly_detection.threshold_percentile', 0.1, 50.0),
            ('genetic_algorithm.population_size', 10, 1000),
            ('genetic_algorithm.generations', 1, 10000),
            ('genetic_algorithm.mutation_rate', 0.0, 1.0),
            ('genetic_algorithm.crossover_rate', 0.0, 1.0),
            ('cmaes_algorithm.generations', 1, 10000),
            ('cmaes_algorithm.initial_sigma', 0.01, 10.0),
        ]
        
        for param_path, min_val, max_val in validations:
            value = self._get_nested_value(param_path)
            if value is not None and not (min_val <= value <= max_val):
                warnings.warn(
                    f"Parameter {param_path} = {value} is outside recommended range "
                    f"[{min_val}, {max_val}]"
                )
    
    def _validate_string_choices(self):
        """Validate string parameters have acceptable values."""
        validations = [
            ('feature_grouping.strategy', ['random', 'correlation', 'domain', 'variance']),
            ('bayesian_network.structure_learning', ['naive_bayes', 'pc', 'hc', 'mmhc']),
            ('anomaly_detection.threshold_method', ['percentile', 'std', 'iqr', 'adaptive']),
            ('anomaly_detection.aggregation_method', ['mean', 'min', 'max', 'median', 'weighted', 'sum']),
            ('optimization.algorithm', ['genetic', 'cmaes']),
            ('visualization.plot_format', ['png', 'pdf', 'svg', 'jpg']),
            ('logging.level', ['DEBUG', 'INFO', 'WARNING', 'ERROR']),
        ]
        
        for param_path, valid_choices in validations:
            value = self._get_nested_value(param_path)
            if value is not None and value not in valid_choices:
                warnings.warn(
                    f"Parameter {param_path} = '{value}' is not in valid choices: {valid_choices}"
                )
    
    def _get_nested_value(self, param_path: str) -> Any:
        """
        Get a nested configuration value using dot notation.
        
        Args:
            param_path (str): Dot-separated parameter path
            
        Returns:
            Any: Parameter value or None if not found
        """
        keys = param_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
    
    def save_config(self, output_path: str):
        """
        Save current configuration to a YAML file.
        
        Args:
            output_path (str): Path to save the configuration
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            print(f"✅ Configuration saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        Get a specific configuration section.
        
        Args:
            section_name (str): Name of the configuration section
            
        Returns:
            Dict[str, Any]: Configuration section
        """
        return self.config.get(section_name, {})
    
    def update_parameter(self, param_path: str, value: Any):
        """
        Update a specific parameter using dot notation.
        
        Args:
            param_path (str): Dot-separated parameter path
            value (Any): New parameter value
        """
        keys = param_path.split('.')
        config_ref = self.config
        
        # Navigate to the parent of the target parameter
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set the parameter value
        config_ref[keys[-1]] = value
        print(f"✅ Updated {param_path} = {value}")


def load_config(config_path: Optional[str] = None, 
                override_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        config_path (str, optional): Path to YAML config file
        override_config (dict, optional): Configuration overrides
        
    Returns:
        Dict[str, Any]: Complete configuration
    """
    loader = ConfigLoader(config_path)
    return loader.load_config(override_config)


def create_example_config(output_path: str = "config_example.yaml"):
    """
    Create an example configuration file with comments.
    
    Args:
        output_path (str): Path to save the example configuration
    """
    loader = ConfigLoader()
    default_config = loader._get_default_config()
    
    # Add example configuration with comments
    example_content = """# Bayesian Anomaly Detection System Configuration
# This file contains all configuration parameters for the anomaly detection pipeline

# Data source configuration
data:
  path: "data/Dati_wallbox_aggregati.csv"  # Path to input CSV file
  
# Data preprocessing configuration
preprocessing:
  missing_threshold: 0.5  # Drop columns with >50% missing values
  scale_features: true    # Apply StandardScaler to numeric features
  handle_categorical: true # Encode categorical variables

# Feature grouping configuration
feature_grouping:
  group_size: 15          # Number of features per group
  strategy: "correlation" # Options: 'random', 'correlation', 'domain', 'variance'

# Bayesian Network learning configuration
bayesian_network:
  structure_learning: "naive_bayes"  # Options: 'naive_bayes', 'pc', 'hc', 'mmhc'
  discretization_bins: 5             # Number of bins for discretization
  max_parents: 3                     # Maximum number of parent nodes

# Anomaly detection configuration
anomaly_detection:
  threshold_percentile: 5.0           # Bottom n% as anomalies
  threshold_method: "percentile"      # Options: 'percentile', 'std', 'iqr', 'adaptive'
  aggregation_method: "mean"          # Options: 'mean', 'min', 'max', 'median', 'weighted', 'sum'
  use_zscore_transformation: true     # Apply z-score transformation before thresholding
  use_rank_transformation: false     # Apply rank-based transformation

# Genetic Algorithm optimization configuration
genetic_algorithm:
  population_size: 50     # Population size for GA
  generations: 100        # Number of generations
  mutation_rate: 0.1      # Mutation probability
  crossover_rate: 0.8     # Crossover probability
  tournament_size: 3      # Tournament selection size

# CMA-ES optimization configuration
cmaes_algorithm:
  population_size: null   # Let CMA-ES decide automatically
  generations: 100        # Number of iterations
  initial_sigma: 0.3      # Initial standard deviation for exploration

# Optimization settings
optimization:
  algorithm: "genetic"    # Options: 'genetic', 'cmaes'
  use_optimization: true  # Enable/disable parameter optimization

# Visualization settings
visualization:
  save_plots: true        # Save plots to files
  show_plots: false       # Display plots on screen
  plot_format: "png"      # Plot file format
  dpi: 300               # Plot resolution

# Output settings
output:
  results_dir: "results"              # Base directory for results
  save_detailed_results: true        # Save detailed analysis files
  save_intermediate_results: false   # Save intermediate processing results
  compression: false                  # Compress output files

# Logging settings
logging:
  level: "INFO"           # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
  save_to_file: false     # Save logs to file
  verbose: true           # Print detailed progress information

# Advanced settings
advanced:
  random_seed: 42                     # Random seed for reproducibility
  parallel_processing: false         # Enable parallel processing (experimental)
  memory_limit_gb: 8.0               # Memory limit in GB
  early_stopping: true               # Enable early stopping in optimization
  convergence_threshold: 1e-6        # Convergence threshold for optimization
"""
    
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(example_content)
        print(f"✅ Example configuration saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving example configuration: {e}")


if __name__ == "__main__":
    # Example usage
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Algorithm: {config['optimization']['algorithm']}")
    print(f"Group size: {config['feature_grouping']['group_size']}")
