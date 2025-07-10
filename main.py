"""
Bayesian Anomaly Detection with Genetic Algorithm Optimization

This module implements a scalable Bayesian Network-based anomaly detection system
that handles large feature sets by dividing them into smaller groups.
"""

import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List

warnings.filterwarnings('ignore')

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.feature_grouper import FeatureGrouper
from src.bayesian_network import BayesianNetworkLearner
from src.anomaly_detector import AnomalyDetector
from src.genetic_optimizer import GeneticOptimizer
from src.cmaes_optimizer import CMAESOptimizer
from src.visualizer import ResultVisualizer

class BayesianAnomalyDetectionSystem:
    """
    Main system for Bayesian Network-based anomaly detection.
    """
    
    def __init__(self, data_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the anomaly detection system.
        
        Args:
            data_path (str): Path to the CSV data file
            config (dict): Configuration parameters
        """
        self.data_path = data_path
        self.config = self._merge_configs(self._default_config(), config or {})
        
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
        
    def _merge_configs(self, default_config: dict, user_config: dict) -> dict:
        """
        Merge user configuration with default configuration.
        
        Args:
            default_config (dict): Default configuration
            user_config (dict): User-provided configuration
            
        Returns:
            dict: Merged configuration
        """
        merged = default_config.copy()
        for key, value in user_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value
        return merged
    
    def _default_config(self):
        """Default configuration parameters."""
        return {
            'preprocessing': {
                'missing_threshold': 0.5,  # Drop columns with >50% missing
                'scale_features': True,
                'handle_categorical': True
            },
            'feature_grouping': {
                'group_size': 15,  # Features per group
                'strategy': 'correlation'  # 'random', 'correlation', 'domain'
            },
            'bayesian_network': {
                'structure_learning': 'naive_bayes',  # 'naive_bayes', 'pc', 'hc'
                'discretization_bins': 5,
                'max_parents': 3
            },
            'anomaly_detection': {
                'aggregation_method': 'mean',  # 'mean', 'min', 'weighted'
                'threshold_percentile': 5  # Bottom 5% as anomalies
            },
            'genetic_algorithm': {
                'population_size': 50,
                'generations': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8
            },
            'cmaes_algorithm': {
                'population_size': None,  # Let CMA-ES decide
                'generations': 100,
                'initial_sigma': 0.3
            },
            'optimization': {
                'algorithm': 'genetic',  # 'genetic' or 'cmaes'
                'use_optimization': True
            }
        }
    
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
    """Main execution function."""
    # Configuration
    data_path = "data/Dati_wallbox_aggregati.csv"
    
    # Custom configuration
    custom_config = {
        'feature_grouping': {
            'group_size': 10,  # Smaller groups for better BN learning
            'strategy': 'correlation'
        },
        'bayesian_network': {
            'structure_learning': 'hc',  # More stable for large datasets
            'discretization_bins': 3  # Fewer bins for better learning
        },
        'anomaly_detection': {
            'threshold_percentile': 4,  # Top n% as anomalies
            'threshold_method': 'percentile',
            'aggregation_method': 'mean',
            'use_zscore_transformation': True
        },
        'genetic_algorithm': {
            'population_size': 100,   # Reasonable population for good optimization
            'generations': 100,       # Sufficient generations for convergence
            'mutation_rate': 0.2,
            'crossover_rate': 0.7
        },
        'cmaes_algorithm': {
            'population_size': None,  # Let CMA-ES decide
            'generations': 150,       # More iterations for enhanced exploration
            'initial_sigma': 0.8      # Higher initial exploration for better diversity
        },
        'optimization': {
            'algorithm': 'genetic',    # 'genetic' or 'cmaes'
            'use_optimization': True
        }
    }
    
    # Initialize and run system
    system = BayesianAnomalyDetectionSystem(data_path, custom_config)
    results = system.run_full_pipeline()
    
    # Plot a representative Bayesian Network (e.g., for group 0)
    if system.bayesian_networks:
        system.plot_bayesian_network(group_id=4)

    # Print analysis
    print("\n📋 ANOMALY DETECTION ANALYSIS")
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
    print(f"\n📁 All results saved to: {execution_folder}")
    print("   - CSV with anomaly scores and classifications")
    print("   - PNG plots showing distributions and timelines") 
    print("   - JSON files with parameters and summaries")
    print("   - Text report with detailed analysis")
    print(f"   - {optimizer_type.upper()} optimization results (if enabled)")
    print("   - Master summary file with execution metadata")

if __name__ == "__main__":
    main()
