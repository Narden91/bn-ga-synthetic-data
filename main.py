import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
import sys

sys.dont_write_bytecode = True

warnings.filterwarnings('ignore')

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.feature_grouper import FeatureGrouper
from src.bayesian_network import BayesianNetworkLearner
from src.anomaly_detector import AnomalyDetector
from src.genetic_optimizer import GeneticOptimizer
from src.cmaes_optimizer import CMAESOptimizer
from src.visualizer import ResultVisualizer
from src.config_loader import ConfigLoader


class BayesianAnomalyDetectionSystem:
    """
    Main system for Bayesian Network-based anomaly detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """
        Initialize the anomaly detection system.
        
        Args:
            config (dict, optional): Configuration parameters (overrides YAML config)
            config_path (str, optional): Path to YAML config file
        """
        # Load configuration using ConfigLoader
        config_loader = ConfigLoader(config_path)
        self.config = config_loader.load_config(config)
        
        # Extract data path from config
        self.data_path = self.config.get('data', {}).get('path', 'data/Dati_wallbox_aggregati.csv')
        
        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor(self.config['preprocessing'])
        self.feature_grouper = FeatureGrouper(self.config['feature_grouping'])
        self.bn_learner = BayesianNetworkLearner(self.config['bayesian_network'])
        self.anomaly_detector = AnomalyDetector(self.config['anomaly_detection'])
        self.genetic_optimizer = GeneticOptimizer(self.config['genetic_algorithm'])
        self.cmaes_optimizer = CMAESOptimizer(self.config['cmaes_algorithm'])
        self.visualizer = ResultVisualizer()
        
        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.feature_groups: Optional[List[List[str]]] = None
        self.bayesian_networks: Dict[int, Any] = {}
        self.likelihood_scores: Optional[pd.DataFrame] = None
        self.anomaly_scores: Optional[np.ndarray] = None
        self.anomalies: Optional[np.ndarray] = None
    
    def run_full_pipeline(self):
        """
        Execute the complete anomaly detection pipeline.
        
        Returns:
            dict: Results containing anomalies, scores, and metrics
        """
        print("🚀 Starting Bayesian Anomaly Detection Pipeline")
        print("=" * 60)
        
        # Step 1: Data Loading
        print("📊 Step 1: Loading data...")
        self.raw_data = self.data_loader.load_data(self.data_path)
        print(f"   Loaded {self.raw_data.shape[0]} samples with {self.raw_data.shape[1]} features")
        
        # Step 2: Data Preprocessing
        print("🔧 Step 2: Preprocessing data...")
        self.processed_data = self.preprocessor.preprocess(self.raw_data)
        print(f"   After preprocessing: {self.processed_data.shape[1]} features")
        
        # Step 3: Feature Grouping
        print("📦 Step 3: Grouping features...")
        self.feature_groups = self.feature_grouper.create_groups(self.processed_data)
        print(f"   Created {len(self.feature_groups)} feature groups")
        
        # Step 4: Bayesian Network Learning
        print("🧠 Step 4: Learning Bayesian Networks...")
        self.bayesian_networks = self.bn_learner.learn_networks(
            self.processed_data, self.feature_groups
        )
        print(f"   Learned {len(self.bayesian_networks)} Bayesian Networks")
        
        # Step 5: Likelihood Calculation
        print("📈 Step 5: Computing likelihoods...")
        self.likelihood_scores = self.bn_learner.compute_likelihoods(
            self.processed_data, self.bayesian_networks, self.feature_groups
        )
        print(f"   Computed likelihood matrix: {self.likelihood_scores.shape}")
        
        # Step 6: Anomaly Detection
        print("🔍 Step 6: Detecting anomalies...")
        self.anomaly_scores, self.anomalies = self.anomaly_detector.detect_anomalies(
            self.likelihood_scores
        )
        print(f"   Detected {len(self.anomalies)} anomalies")
        
        # Step 7: Optimization (Optional)
        if self.config['optimization'].get('use_optimization', True):
            optimizer_type = self.config['optimization'].get('algorithm', 'genetic')
            
            if optimizer_type == 'genetic':
                print("🧬 Step 7: Optimizing with Genetic Algorithm...")
                
                # Set the GA optimizer to use the same execution folder
                self.genetic_optimizer.set_results_dir(self.visualizer.get_execution_folder_path())
                
                optimized_params = self.genetic_optimizer.optimize(
                    self.likelihood_scores, self.processed_data
                )
                print(f"   Optimized threshold percentile: {optimized_params['threshold_percentile']:.2f}%")
                print(f"   Optimized aggregation method: {optimized_params['aggregation_method']}")
                
            elif optimizer_type == 'cmaes':
                print("🎯 Step 7: Optimizing with CMA-ES...")
                
                # Set the CMA-ES optimizer to use the same execution folder
                self.cmaes_optimizer.set_results_dir(self.visualizer.get_execution_folder_path())
                
                optimized_params = self.cmaes_optimizer.optimize(
                    self.likelihood_scores, self.processed_data
                )
                print(f"   Optimized threshold percentile: {optimized_params['threshold_percentile']:.2f}%")
                print(f"   Optimized aggregation method: {optimized_params['aggregation_method']}")
                
            else:
                raise ValueError(f"Unknown optimizer type: {optimizer_type}. Use 'genetic' or 'cmaes'.")
            
            # Re-detect with optimized parameters
            self.anomaly_detector.update_parameters(optimized_params)
            self.anomaly_scores, self.anomalies = self.anomaly_detector.detect_anomalies(
                self.likelihood_scores
            )
            print(f"   Optimized detection: {len(self.anomalies)} anomalies")
        
        # Step 8: Results Visualization
        print("📊 Step 8: Generating visualizations...")
        self.visualizer.create_visualizations(
            self.anomaly_scores, self.anomalies, self.likelihood_scores
        )
        
        # Step 9: Save comprehensive results
        print("💾 Step 9: Saving results...")
        self._save_comprehensive_results()
        
        print("✅ Pipeline completed successfully!")
        print("=" * 60)
        
        return {
            'anomalies': self.anomalies,
            'anomaly_scores': self.anomaly_scores,
            'likelihood_scores': self.likelihood_scores,
            'feature_groups': self.feature_groups,
            'n_anomalies': len(self.anomalies) if self.anomalies is not None else 0
        }
    
    def plot_bayesian_network(self, group_id: int):
        """
        Plot the Bayesian Network for a specific group.
        
        Args:
            group_id (int): The ID of the feature group to plot.
        """
        print(f"📊 Plotting Bayesian Network for group {group_id}...")
        if not self.bayesian_networks:
            print("     ❌ Bayesian networks have not been learned yet. Run the pipeline first.")
            return
            
        if group_id not in self.bayesian_networks:
            print(f"     ❌ Group ID {group_id} not found. Available groups: {list(self.bayesian_networks.keys())}")
            return
            
        # Create a figure and axes for the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use the learner to plot the network
        self.bn_learner.plot_network(group_id, ax=ax)
        
        # Save the plot to the execution folder
        plot_path = os.path.join(self.visualizer.get_execution_folder_path(), f"bayesian_network_group_{group_id}.png")
        fig.savefig(plot_path)
        plt.close(fig)
        
        print(f"     ✅ Network plot saved to {plot_path}")

    def get_anomaly_analysis(self):
        """
        Get detailed analysis of detected anomalies.
        
        Returns:
            dict: Detailed analysis results
        """
        if self.anomalies is None or self.processed_data is None or self.anomaly_scores is None:
            raise ValueError("No anomalies detected yet. Run the pipeline first.")
        
        analysis = {
            'total_samples': len(self.processed_data),
            'total_anomalies': len(self.anomalies),
            'anomaly_percentage': len(self.anomalies) / len(self.processed_data) * 100,
            'anomaly_scores_stats': {
                'mean': np.mean(self.anomaly_scores),
                'std': np.std(self.anomaly_scores),
                'min': np.min(self.anomaly_scores),
                'max': np.max(self.anomaly_scores),
                'median': np.median(self.anomaly_scores)
            },
            'top_anomalies': self.anomalies.tolist()[:10]
        }
        
        return analysis
    
    def _save_comprehensive_results(self):
        """Save all results to the results folder."""
        try:
            import json
            from datetime import datetime
            
            # Use the visualizer's execution folder instead of creating separate timestamped files
            results_dir = self.visualizer.get_execution_folder_path()
            
            # 1. Save main results
            if self.anomaly_scores is not None and self.anomalies is not None and self.likelihood_scores is not None:
                self.visualizer.export_results(
                    self.anomaly_scores, 
                    self.anomalies, 
                    self.likelihood_scores
                )
            
            # 2. Save summary report
            if self.anomaly_scores is not None and self.anomalies is not None and self.likelihood_scores is not None and self.feature_groups is not None:
                summary_report = self.visualizer.create_summary_report(
                    self.anomaly_scores,
                    self.anomalies,
                    self.likelihood_scores,
                    self.feature_groups
                )
                
                report_file = os.path.join(results_dir, "summary_report.txt")
                with open(report_file, 'w') as f:
                    f.write(summary_report)
                print(f"     Summary report saved to {report_file}")
            
            # 3. Save configuration
            config_file = os.path.join(results_dir, "pipeline_config.json")
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"     Configuration saved to {config_file}")
            
            # 4. Save feature groups
            groups_file = os.path.join(results_dir, "feature_groups.json")
            group_data = {
                'groups': self.feature_groups,
                'group_info': self.feature_grouper.get_group_info(),
                'execution_timestamp': os.path.basename(results_dir)
            }
            with open(groups_file, 'w') as f:
                json.dump(group_data, f, indent=2)
            print(f"     Feature groups saved to {groups_file}")
            
            # 5. Save analysis summary
            try:
                analysis = self.get_anomaly_analysis()
                analysis_file = os.path.join(results_dir, "anomaly_analysis.json")
                with open(analysis_file, 'w') as f:
                    # Convert numpy types to native Python types for JSON serialization
                    analysis_serializable = self._make_json_serializable(analysis)
                    json.dump(analysis_serializable, f, indent=2)
                print(f"     Analysis summary saved to {analysis_file}")
            except Exception as e:
                print(f"     Warning: Could not save analysis summary: {str(e)}")
            
            # 6. Create master summary file
            execution_timestamp = os.path.basename(results_dir)
            master_summary = {
                'execution_timestamp': execution_timestamp,
                'data_info': {
                    'original_shape': list(self.raw_data.shape) if self.raw_data is not None else None,
                    'processed_shape': list(self.processed_data.shape) if self.processed_data is not None else None,
                    'n_feature_groups': len(self.feature_groups) if self.feature_groups is not None else 0
                },
                'results': {
                    'total_anomalies': len(self.anomalies) if self.anomalies is not None else 0,
                    'anomaly_rate': (len(self.anomalies) / len(self.processed_data) * 100) if (self.anomalies is not None and self.processed_data is not None) else 0,
                    'threshold_used': float(self.anomaly_detector.threshold) if (hasattr(self.anomaly_detector, 'threshold') and self.anomaly_detector.threshold is not None) else None
                },
                'files_created': {
                    'results_csv': "anomaly_results.csv",
                    'summary_report': "summary_report.txt",
                    'config': "pipeline_config.json",
                    'feature_groups': "feature_groups.json",
                    'analysis': "anomaly_analysis.json"
                }
            }
            
            # Add optimization results if available
            optimizer_type = self.config['optimization'].get('algorithm', 'genetic')
            
            if optimizer_type == 'genetic' and hasattr(self.genetic_optimizer, 'best_individual') and self.genetic_optimizer.best_individual is not None:
                ga_summary = self.genetic_optimizer.get_optimization_summary()
                master_summary['optimization'] = {
                    'algorithm': 'genetic',
                    'results': self._make_json_serializable(ga_summary)
                }
            elif optimizer_type == 'cmaes' and hasattr(self.cmaes_optimizer, 'best_solution') and self.cmaes_optimizer.best_solution is not None:
                cmaes_summary = self.cmaes_optimizer.get_optimization_summary()
                master_summary['optimization'] = {
                    'algorithm': 'cmaes',
                    'results': self._make_json_serializable(cmaes_summary)
                }
            
            master_file = os.path.join(results_dir, "master_summary.json")
            with open(master_file, 'w') as f:
                json.dump(master_summary, f, indent=2)
            print(f"     Master summary saved to {master_file}")
            
        except Exception as e:
            print(f"     Error saving comprehensive results: {str(e)}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj


def main():
    """Main execution function with YAML configuration support."""
    print("*** BAYESIAN ANOMALY DETECTION SYSTEM")
    print("=" * 50)
    
    # Option 1: Use default config.yaml file
    print(">>> Loading configuration...")
    
    # config_path = "custom_config.yaml"  # Optional: use custom config file
    config_path = None  # Uses default config.yaml
    
    # Optional: Override specific configuration parameters
    # This will be merged with the YAML configuration
    config_overrides = {
        'data': {
            'path': "data/Dati_wallbox_aggregati.csv"
        },
        'feature_grouping': {
            'group_size': 10,  # Smaller groups for better BN learning
            'strategy': 'correlation'
        },
        'bayesian_network': {
            'structure_learning': 'hc',  # More stable for large datasets
            'discretization_bins': 3  # Fewer bins for better learning
        },
        'anomaly_detection': {
            'threshold_percentile': 4.0,  # Top n% as anomalies
            'threshold_method': 'percentile',
            'aggregation_method': 'mean',
            'use_zscore_transformation': True
        },
        'genetic_algorithm': {
            'population_size': 100,   # Reasonable population for good optimization
            'generations': 100,       # Sufficient generations for convergence
            'mutation_rate': 0.2,
            'crossover_rate': 0.7,
            'fitness_weights': {
                'detection_quality': 0.65,      # Increase focus on detection
                'statistical_coherence': 0.25,  # Reduce quality weight
                'diversity_bonus': 0.10         # Keep exploration
            }
        },
        'cmaes_algorithm': {
            'population_size': None,  # Let CMA-ES decide
            'generations': 150,       # More iterations for enhanced exploration
            'initial_sigma': 0.8,      # Higher initial exploration for better diversity
            'fitness_weights': {
                'rate_score': 0.20,           # Anomaly rate importance
                'separation_score': 0.35,     # Most critical - increase weight
                'distribution_score': 0.15,   # Statistical quality
                'stability_score': 0.15,      # Robustness
                'domain_score': 0.10,         # Domain patterns
                'exploration_score': 0.03,    # Parameter diversity
                'convergence_score': 0.02     # Anti-stagnation
            }
        },
        'optimization': {
            'algorithm': 'genetic',    # 'genetic' or 'cmaes'
            'use_optimization': True
        }
    }
    
    # Initialize and run system with YAML configuration
    system = BayesianAnomalyDetectionSystem(
        config=config_overrides,  # Optional overrides
        config_path=config_path   # Path to YAML config (None = default config.yaml)
    )
    
    print(f">>> Data source: {system.data_path}")
    print(f">>> Optimizer: {system.config['optimization']['algorithm'].upper()}")
    print(f">>> Group size: {system.config['feature_grouping']['group_size']}")
    print()
    
    # Run
    results = system.run_full_pipeline()
    
    # Plot a representative Bayesian Network (e.g., for group 4)
    if system.bayesian_networks:
        system.plot_bayesian_network(group_id=4)

    # Print analysis
    print("\n*** ANOMALY DETECTION ANALYSIS")
    print("=" * 40)
    analysis = system.get_anomaly_analysis()
    
    print(f"Total samples: {analysis['total_samples']}")
    print(f"Anomalies detected: {analysis['total_anomalies']}")
    print(f"Anomaly rate: {analysis['anomaly_percentage']:.2f}%")
    print(f"Score range: [{analysis['anomaly_scores_stats']['min']:.4f}, {analysis['anomaly_scores_stats']['max']:.4f}]")
    print(f"Mean score: {analysis['anomaly_scores_stats']['mean']:.4f}")
    
    if len(analysis['top_anomalies']) > 0:
        print(f"\nTop anomalous samples (indices): {list(analysis['top_anomalies'][:5])}")
    
    # Show the specific execution folder where results were saved
    execution_folder = system.visualizer.get_execution_folder_path()
    optimizer_type = system.config['optimization'].get('algorithm', 'genetic')
    print(f"\n>>> All results saved to: {execution_folder}")
    print("   - CSV with anomaly scores and classifications")
    print("   - PNG plots showing distributions and timelines") 
    print("   - JSON files with parameters and summaries")
    print("   - Text report with detailed analysis")
    print(f"   - {optimizer_type.upper()} optimization results (if enabled)")
    print("   - Master summary file with execution metadata")
    print(f"   - Pipeline configuration: pipeline_config.json")
    
    print(f"\n*** Configuration was loaded from: config.yaml")
    print("   You can modify config.yaml to change system behavior")
    print("   Or create custom config files and specify them in the code")


def main_with_custom_config(custom_config_path: str):
    """Run the system with a custom configuration file."""
    print("*** BAYESIAN ANOMALY DETECTION SYSTEM (Custom Config)")
    print("=" * 60)
    
    # Minimal overrides (if any)
    minimal_overrides = {
        'logging': {
            'verbose': True
        }
    }
    
    try:
        system = BayesianAnomalyDetectionSystem(
            config=minimal_overrides,
            config_path=custom_config_path
        )
        results = system.run_full_pipeline()
        
        # Plot a representative Bayesian Network (e.g., for group 4)
        if system.bayesian_networks:
            system.plot_bayesian_network(group_id=4)
        
        # Print analysis
        print("\n*** ANOMALY DETECTION ANALYSIS")
        print("=" * 40)
        analysis = system.get_anomaly_analysis()
        
        print(f"Total samples: {analysis['total_samples']}")
        print(f"Anomalies detected: {analysis['total_anomalies']}")
        print(f"Anomaly rate: {analysis['anomaly_percentage']:.2f}%")
        print(f"Score range: [{analysis['anomaly_scores_stats']['min']:.4f}, {analysis['anomaly_scores_stats']['max']:.4f}]")
        print(f"Mean score: {analysis['anomaly_scores_stats']['mean']:.4f}")
        
        if len(analysis['top_anomalies']) > 0:
            print(f"\nTop anomalous samples (indices): {list(analysis['top_anomalies'][:5])}")
        
        # Show the specific execution folder where results were saved
        execution_folder = system.visualizer.get_execution_folder_path()
        optimizer_type = system.config['optimization'].get('algorithm', 'genetic')
        print(f"\n>>> All results saved to: {execution_folder}")
        print("   - CSV with anomaly scores and classifications")
        print("   - PNG plots showing distributions and timelines") 
        print("   - JSON files with parameters and summaries")
        print("   - Text report with detailed analysis")
        print(f"   - {optimizer_type.upper()} optimization results (if enabled)")
        print("   - Master summary file with execution metadata")
        print(f"   - Pipeline configuration: pipeline_config.json")
        
        print(f"\n*** Experiment completed with config: {custom_config_path}")
        
    except FileNotFoundError:
        print(f"ERROR: Custom config file not found: {custom_config_path}")
        print("   Falling back to default configuration...")
        system = BayesianAnomalyDetectionSystem()
        results = system.run_full_pipeline()
        
    except Exception as e:
        print(f"ERROR: Error loading config file {custom_config_path}: {str(e)}")
        print("   This might be because the file is empty or has invalid format.")
        print("   Falling back to default configuration...")
        system = BayesianAnomalyDetectionSystem()
        results = system.run_full_pipeline()


def display_experiment_info():
    """Display detailed information about available experiments."""
    print("\n*** DETAILED EXPERIMENT INFORMATION")
    print("=" * 60)
    
    experiments = {
        1: {
            'name': 'Default Configuration',
            'description': 'Built-in optimized settings with balanced parameters',
            'optimizer': 'Genetic Algorithm',
            'best_for': 'General purpose, first-time users'
        },
        2: {
            'name': 'High-Performance GA',
            'description': 'Thorough exploration with enhanced genetic algorithm',
            'optimizer': 'Genetic Algorithm',
            'best_for': 'When you need comprehensive parameter exploration'
        },
        3: {
            'name': 'CMA-ES Optimization',
            'description': 'Advanced parameter tuning with CMA-ES algorithm',
            'optimizer': 'CMA-ES',
            'best_for': 'Continuous parameter optimization, research purposes'
        },
        4: {
            'name': 'Balanced Detection',
            'description': 'Equal focus on detection quality and robustness',
            'optimizer': 'CMA-ES',
            'best_for': 'Production environments, balanced approach'
        },
        5: {
            'name': 'Conservative Detection',
            'description': 'Minimize false positives, high separation quality',
            'optimizer': 'CMA-ES',
            'best_for': 'When false positives are costly'
        },
        6: {
            'name': 'Aggressive Detection',
            'description': 'Catch more anomalies with parameter exploration',
            'optimizer': 'Genetic Algorithm',
            'best_for': 'When missing anomalies is costly'
        },
        7: {
            'name': 'Custom Weights',
            'description': 'Experimental fitness weight configurations',
            'optimizer': 'Varies',
            'best_for': 'Experimental purposes, custom scenarios'
        },
        8: {
            'name': 'Standard Aggressive',
            'description': 'Basic aggressive detection settings',
            'optimizer': 'Varies',
            'best_for': 'Simple aggressive detection'
        },
        9: {
            'name': 'Standard Conservative',
            'description': 'Basic conservative detection settings',
            'optimizer': 'Varies',
            'best_for': 'Simple conservative detection'
        }
    }
    
    for exp_id, info in experiments.items():
        print(f"{exp_id}. {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Optimizer: {info['optimizer']}")
        print(f"   Best for: {info['best_for']}")
        print()
    
    print("*** To run an experiment, change the 'experiment_to_run' value in main.py")
    print("   and execute the script again.")
    print("=" * 60)


if __name__ == "__main__":
    # --- CHOOSE YOUR EXPERIMENT ---
    # Select an experiment by changing the experiment_to_run value below
    
    print("*** AVAILABLE EXPERIMENTS:")
    print("=" * 60)
    print("1. Default Configuration - Built-in optimized settings")
    print("2. High-Performance GA - Thorough exploration with Genetic Algorithm")
    print("3. CMA-ES Optimization - Advanced CMA-ES parameter tuning")
    print("4. Balanced Detection - Equal focus on quality and robustness (CMA-ES)")
    print("5. Conservative Detection - Fewer false positives, high separation (CMA-ES)")
    print("6. Aggressive Detection - More anomalies, higher exploration (GA)")
    print("7. Custom Weights - Experimental weight configurations")
    print("8. Standard Aggressive - Basic aggressive detection settings")
    print("9. Standard Conservative - Basic conservative detection settings")
    print("=" * 60)
    
    experiment_to_run = 7  # <--- CHANGE THIS VALUE TO SELECT AN EXPERIMENT

    if experiment_to_run == 1:
        print(">>> Running default configuration...")
        main()
    else:
        experiment_files = {
            2: "experiments/experiment_1.yaml",                    # High-Performance GA
            3: "experiments/experiment_2_cmaes.yaml",              # CMA-ES Optimization
            4: "experiments/experiment_weighted_balanced.yaml",    # Balanced Detection
            5: "experiments/experiment_weighted_conservative.yaml", # Conservative Detection
            6: "experiments/experiment_weighted_aggressive.yaml",   # Aggressive Detection
            7: "experiments/experiment_custom_weights.yaml",       # Custom Weights
            8: "experiments/experiment_aggressive.yaml",           # Standard Aggressive
            9: "experiments/experiment_conservative.yaml",         # Standard Conservative
        }
        
        experiment_descriptions = {
            2: "High-Performance Genetic Algorithm - Thorough exploration and quality results",
            3: "CMA-ES Optimization - Advanced parameter tuning with CMA-ES algorithm",
            4: "Balanced Detection - Equal focus on detection quality and robustness",
            5: "Conservative Detection - Minimize false positives, high separation quality",
            6: "Aggressive Detection - Catch more anomalies with parameter exploration",
            7: "Custom Weights - Experimental fitness weight configurations",
            8: "Standard Aggressive - Basic aggressive detection settings",
            9: "Standard Conservative - Basic conservative detection settings",
        }
        
        config_file = experiment_files.get(experiment_to_run)
        
        if config_file:
            description = experiment_descriptions.get(experiment_to_run, "No description available")
            print(f">>> Running experiment {experiment_to_run}: {description}")
            print(f">>> Config file: {config_file}")
            print()
            main_with_custom_config(config_file)
        else:
            print(f"ERROR: Invalid choice: {experiment_to_run}. Please choose from 1-9.")
            print("\nAvailable experiments:")
            print("  1: Default Configuration")
            for key, value in experiment_files.items():
                desc = experiment_descriptions.get(key, "No description")
                print(f"  {key}: {desc}")
                print(f"      -> {value}")
            print("\n*** Change 'experiment_to_run' value in main.py to select different experiment")
