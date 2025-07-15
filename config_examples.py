import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import BayesianAnomalyDetectionSystem
from src.config_loader import ConfigLoader, create_example_config


def example_1_default_config():
    """Example 1: Using default config.yaml file."""
    print("=" * 60)
    print("EXAMPLE 1: Using Default Configuration")
    print("=" * 60)
    
    # Simple initialization - uses config.yaml automatically
    system = BayesianAnomalyDetectionSystem()
    
    print(f"Data path: {system.data_path}")
    print(f"Algorithm: {system.config['optimization']['algorithm']}")
    print(f"Group size: {system.config['feature_grouping']['group_size']}")
    print()


def example_2_custom_config_file():
    """Example 2: Using a custom configuration file."""
    print("=" * 60)
    print("EXAMPLE 2: Using Custom Configuration File")
    print("=" * 60)
    
    # Use experiment configuration
    config_path = "experiments/experiment_1.yaml"
    
    if os.path.exists(config_path):
        system = BayesianAnomalyDetectionSystem(config_path=config_path)
        print(f"Loaded config from: {config_path}")
        print(f"Algorithm: {system.config['optimization']['algorithm']}")
        print(f"Population size: {system.config['genetic_algorithm']['population_size']}")
        print(f"Group size: {system.config['feature_grouping']['group_size']}")
    else:
        print(f"Config file not found: {config_path}")
    print()


def example_3_config_overrides():
    """Example 3: Using configuration overrides."""
    print("=" * 60)
    print("EXAMPLE 3: Using Configuration Overrides")
    print("=" * 60)
    
    # Override specific parameters while keeping the rest from YAML
    overrides = {
        'optimization': {
            'algorithm': 'cmaes',
            'use_optimization': False  # Disable optimization for quick testing
        },
        'feature_grouping': {
            'group_size': 5  # Very small groups for testing
        },
        'output': {
            'save_detailed_results': False  # Minimal output for testing
        }
    }
    
    system = BayesianAnomalyDetectionSystem(config=overrides)
    print(f"Algorithm: {system.config['optimization']['algorithm']}")
    print(f"Use optimization: {system.config['optimization']['use_optimization']}")
    print(f"Group size: {system.config['feature_grouping']['group_size']}")
    print()


def example_4_environment_variables():
    """Example 4: Using environment variables in configuration."""
    print("=" * 60)
    print("EXAMPLE 4: Environment Variables in Configuration")
    print("=" * 60)
    
    # Create a config with environment variables
    env_config_content = """
data:
  path: "${DATA_PATH:data/Dati_wallbox_aggregati.csv}"

optimization:
  algorithm: "${OPTIMIZER:genetic}"
  
feature_grouping:
  group_size: 15

preprocessing:
  missing_threshold: 0.5
  scale_features: true
  handle_categorical: true

bayesian_network:
  structure_learning: "naive_bayes"
  discretization_bins: 5
  max_parents: 3

anomaly_detection:
  threshold_percentile: 5.0
  threshold_method: "percentile"
  aggregation_method: "mean"
  use_zscore_transformation: true

genetic_algorithm:
  population_size: 50
  generations: 100
  mutation_rate: 0.1
  crossover_rate: 0.8
  tournament_size: 3

cmaes_algorithm:
  population_size: null
  generations: 100
  initial_sigma: 0.3

visualization:
  save_plots: true
  show_plots: false
  plot_format: "png"
  dpi: 300

output:
  results_dir: "results"
  save_detailed_results: true
  save_intermediate_results: false
  compression: false

logging:
  level: "INFO"
  save_to_file: false
  verbose: true

advanced:
  random_seed: 42
  parallel_processing: false
  memory_limit_gb: 8.0
  early_stopping: true
  convergence_threshold: 1e-6
"""
    
    # Save temporary config file
    temp_config_path = "temp_env_config.yaml"
    with open(temp_config_path, 'w') as f:
        f.write(env_config_content)
    
    # Set environment variables
    os.environ['DATA_PATH'] = 'data/Dati_wallbox_aggregati.csv'
    os.environ['OPTIMIZER'] = 'cmaes'
    
    try:
        system = BayesianAnomalyDetectionSystem(config_path=temp_config_path)
        print(f"Data path from env: {system.data_path}")
        print(f"Algorithm from env: {system.config['optimization']['algorithm']}")
        print("Environment variables successfully substituted!")
    finally:
        # Cleanup
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    print()


def example_5_config_validation():
    """Example 5: Configuration validation and error handling."""
    print("=" * 60)
    print("EXAMPLE 5: Configuration Validation")
    print("=" * 60)
    
    # Create invalid configuration
    invalid_config = {
        'feature_grouping': {
            'group_size': -5,  # Invalid: negative value
            'strategy': 'invalid_strategy'  # Invalid: unknown strategy
        },
        'genetic_algorithm': {
            'mutation_rate': 1.5  # Invalid: > 1.0
        }
    }
    
    try:
        system = BayesianAnomalyDetectionSystem(config=invalid_config)
        print("⚠️  System created despite invalid configuration (validation warnings shown above)")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
    print()


def create_all_example_configs():
    """Create all example configuration files."""
    print("=" * 60)
    print("CREATING EXAMPLE CONFIGURATION FILES")
    print("=" * 60)
    
    # Create standard example config
    create_example_config("config_example.yaml")
    
    # Create experiment configs (already done in main script)
    print("✅ Example configurations created in experiments/ folder")
    print("   - experiment_1.yaml (High-performance GA)")
    print("   - experiment_2_cmaes.yaml (CMA-ES optimization)")
    print()


def run_quick_test():
    """Run a quick test with minimal configuration."""
    print("=" * 60)
    print("QUICK TEST RUN")
    print("=" * 60)
    
    # Quick test configuration
    test_config = {
        'data': {
            'path': 'data/Dati_wallbox_aggregati.csv'
        },
        'optimization': {
            'use_optimization': False  # Skip optimization for speed
        },
        'feature_grouping': {
            'group_size': 5  # Small groups for speed
        },
        'output': {
            'save_detailed_results': False  # Minimal output
        }
    }
    
    try:
        print("🚀 Running quick test...")
        system = BayesianAnomalyDetectionSystem(config=test_config)
        
        # Check if data file exists
        if os.path.exists(system.data_path):
            print(f"✅ Data file found: {system.data_path}")
            print("   System ready for full pipeline execution")
        else:
            print(f"⚠️  Data file not found: {system.data_path}")
            print("   Please ensure the data file exists before running the full pipeline")
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
    print()


def main():
    """Run all configuration examples."""
    print("🔧 BAYESIAN ANOMALY DETECTION SYSTEM")
    print("📚 Configuration Examples and Usage Guide")
    print("=" * 80)
    
    # Run examples
    create_all_example_configs()
    example_1_default_config()
    example_2_custom_config_file()
    example_3_config_overrides()
    example_4_environment_variables()
    example_5_config_validation()
    run_quick_test()
    
    print("=" * 80)
    print("🎯 CONFIGURATION USAGE SUMMARY")
    print("=" * 80)
    print("1. Default: BayesianAnomalyDetectionSystem()")
    print("2. Custom file: BayesianAnomalyDetectionSystem(config_path='my_config.yaml')")
    print("3. Overrides: BayesianAnomalyDetectionSystem(config={'key': 'value'})")
    print("4. Both: BayesianAnomalyDetectionSystem(config={'key': 'value'}, config_path='my_config.yaml')")
    print()
    print("📁 Configuration files:")
    print("   - config.yaml (default)")
    print("   - experiments/experiment_1.yaml (high-performance GA)")
    print("   - experiments/experiment_2_cmaes.yaml (CMA-ES)")
    print("   - config_example.yaml (documented example)")
    print()
    print("🔧 To run the main system:")
    print("   python main.py")
    print()


if __name__ == "__main__":
    main()
