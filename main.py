"""
Bayesian Anomaly Detection with Genetic Algorithm Optimization

This module implements a scalable Bayesian Network-based anomaly detection system
that handles large feature sets by dividing them into smaller groups.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.feature_grouper import FeatureGrouper
from src.bayesian_network import BayesianNetworkLearner
from src.anomaly_detector import AnomalyDetector
from src.genetic_optimizer import GeneticOptimizer
from src.visualizer import ResultVisualizer

class BayesianAnomalyDetectionSystem:
    """
    Main system for Bayesian Network-based anomaly detection.
    """
    
    def __init__(self, data_path: str, config: dict = None):
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
        self.visualizer = ResultVisualizer()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.feature_groups = None
        self.bayesian_networks = {}
        self.likelihood_scores = None
        self.anomaly_scores = None
        self.anomalies = None
        
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
        
        # Step 7: GA Optimization (Optional)
        if self.config.get('use_genetic_optimization', True):
            print("🧬 Step 7: Optimizing with Genetic Algorithm...")
            optimized_params = self.genetic_optimizer.optimize(
                self.likelihood_scores, self.processed_data
            )
            print(f"   Optimized threshold percentile: {optimized_params['threshold_percentile']:.2f}%")
            print(f"   Optimized aggregation method: {optimized_params['aggregation_method']}")
            
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
        
        print("✅ Pipeline completed successfully!")
        print("=" * 60)
        
        return {
            'anomalies': self.anomalies,
            'anomaly_scores': self.anomaly_scores,
            'likelihood_scores': self.likelihood_scores,
            'feature_groups': self.feature_groups,
            'n_anomalies': len(self.anomalies)
        }
    
    def get_anomaly_analysis(self):
        """
        Get detailed analysis of detected anomalies.
        
        Returns:
            dict: Detailed analysis results
        """
        if self.anomalies is None:
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
            'top_anomalies': self.anomalies[:10] if len(self.anomalies) > 10 else self.anomalies
        }
        
        return analysis

def main():
    """Main execution function."""
    # Configuration
    data_path = "data/Dati_wallbox_aggregati.csv"
    
    # Custom configuration (optional)
    custom_config = {
        'feature_grouping': {
            'group_size': 10,  # Smaller groups for better BN learning
            'strategy': 'correlation'
        },
        'bayesian_network': {
            'structure_learning': 'naive_bayes',  # More stable for large datasets
            'discretization_bins': 3  # Fewer bins for better learning
        },
        'anomaly_detection': {
            'threshold_percentile': 5,  # Top 5% as anomalies
            'threshold_method': 'percentile',
            'aggregation_method': 'mean',
            'use_zscore_transformation': True
        },
        'genetic_algorithm': {
            'population_size': 100,  # Smaller population for faster optimization
            'generations': 100,      # Fewer generations for faster optimization
            'mutation_rate': 0.2,
            'crossover_rate': 0.8
        },
        'use_genetic_optimization': True
    }
    
    # Initialize and run system
    system = BayesianAnomalyDetectionSystem(data_path, custom_config)
    results = system.run_full_pipeline()
    
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

if __name__ == "__main__":
    main()
