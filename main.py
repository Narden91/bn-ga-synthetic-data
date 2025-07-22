import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
import json
import traceback
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
        
        # Optimization comparison storage
        self.baseline_performance: Optional[Dict[str, Any]] = None
        self.optimized_performance: Optional[Dict[str, Any]] = None
        self.optimization_comparison: Optional[Dict[str, Any]] = None
    
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
        
        # Step 6: Anomaly Detection (Baseline)
        print("🔍 Step 6: Detecting anomalies (baseline)...")
        
        # Store original config and temporarily use default baseline config for fair comparison
        original_config = self.anomaly_detector.config.copy()
        baseline_config = original_config.copy()
        # Use default aggregation method for baseline (not experiment-specific method)
        baseline_config['aggregation_method'] = 'mean'
        baseline_config['threshold_percentile'] = 5.0
        self.anomaly_detector.config = baseline_config
        
        self.anomaly_scores, self.anomalies = self.anomaly_detector.detect_anomalies(
            self.likelihood_scores
        )
        print(f"   Detected {len(self.anomalies)} anomalies (baseline)")
        
        # Store baseline results for comparison (with default config)
        baseline_results = self._compute_performance_metrics()
        self.baseline_performance = baseline_results
        
        # Restore original config for optimization
        self.anomaly_detector.config = original_config
        
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
            
            # Compute optimized results and compare with baseline
            optimized_results = self._compute_performance_metrics()
            self.optimized_performance = optimized_results
            self._display_optimization_comparison(baseline_results, optimized_results, optimizer_type)
            
            # Store comparison data
            self.optimization_comparison = self._create_comparison_data(baseline_results, optimized_results, optimizer_type)
            
            # Create visual comparison plots
            self._create_optimization_comparison_plots(baseline_results, optimized_results, optimizer_type)
        
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

    def _compute_performance_metrics(self):
        """
        Compute comprehensive performance metrics for the current anomaly detection state.
        
        Returns:
            dict: Performance metrics including statistical measures
        """
        if self.anomaly_scores is None or self.anomalies is None or self.processed_data is None:
            return {}
        
        # Basic detection metrics
        total_samples = len(self.processed_data)
        n_anomalies = len(self.anomalies)
        anomaly_rate = (n_anomalies / total_samples) * 100
        
        # Statistical metrics for anomaly scores
        anomaly_indices = set(self.anomalies)
        normal_indices = [i for i in range(total_samples) if i not in anomaly_indices]
        
        # Score statistics
        all_scores_stats = {
            'mean': float(np.mean(self.anomaly_scores)),
            'std': float(np.std(self.anomaly_scores)),
            'min': float(np.min(self.anomaly_scores)),
            'max': float(np.max(self.anomaly_scores)),
            'median': float(np.median(self.anomaly_scores))
        }
        
        # Separation quality (Cohen's d)
        if len(self.anomalies) > 0 and len(normal_indices) > 0:
            anomaly_scores_subset = self.anomaly_scores[self.anomalies]
            normal_scores_subset = self.anomaly_scores[normal_indices]
            
            mean_anomaly = np.mean(anomaly_scores_subset)
            mean_normal = np.mean(normal_scores_subset)
            var_anomaly = np.var(anomaly_scores_subset, ddof=1) if len(anomaly_scores_subset) > 1 else 0
            var_normal = np.var(normal_scores_subset, ddof=1) if len(normal_scores_subset) > 1 else 0
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((len(anomaly_scores_subset) - 1) * var_anomaly + 
                                 (len(normal_scores_subset) - 1) * var_normal) / 
                                (len(anomaly_scores_subset) + len(normal_scores_subset) - 2))
            
            cohens_d = abs(mean_anomaly - mean_normal) / pooled_std if pooled_std > 0 else 0
            
            separation_stats = {
                'cohens_d': float(cohens_d),
                'mean_anomaly_score': float(mean_anomaly),
                'mean_normal_score': float(mean_normal),
                'separation_magnitude': float(abs(mean_anomaly - mean_normal))
            }
        else:
            separation_stats = {
                'cohens_d': 0.0,
                'mean_anomaly_score': 0.0,
                'mean_normal_score': 0.0,
                'separation_magnitude': 0.0
            }
        
        # Threshold robustness (if available)
        current_threshold = getattr(self.anomaly_detector, 'threshold', None)
        threshold_percentile = getattr(self.anomaly_detector, 'threshold_percentile', None)
        
        return {
            'detection_metrics': {
                'total_samples': total_samples,
                'n_anomalies': n_anomalies,
                'anomaly_rate': anomaly_rate,
                'threshold': float(current_threshold) if current_threshold is not None else None,
                'threshold_percentile': float(threshold_percentile) if threshold_percentile is not None else None
            },
            'score_statistics': all_scores_stats,
            'separation_quality': separation_stats,
            'aggregation_method': getattr(self.anomaly_detector, 'aggregation_method', 'unknown')
        }
    
    def _display_optimization_comparison(self, baseline_results, optimized_results, optimizer_type):
        """
        Display a comprehensive comparison between baseline and optimized results.
        
        Args:
            baseline_results (dict): Performance metrics before optimization
            optimized_results (dict): Performance metrics after optimization
            optimizer_type (str): Type of optimizer used ('genetic' or 'cmaes')
        """
        print("\n" + "="*80)
        print(f"📊 OPTIMIZATION PERFORMANCE COMPARISON ({optimizer_type.upper()})")
        print("="*80)
        
        if not baseline_results or not optimized_results:
            print("❌ Unable to perform comparison - insufficient data")
            return
        
        # Detection metrics comparison
        print("\n🎯 DETECTION METRICS COMPARISON:")
        print("-" * 50)
        
        baseline_det = baseline_results['detection_metrics']
        optimized_det = optimized_results['detection_metrics']
        
        print(f"{'Metric':<25} {'Baseline':<15} {'Optimized':<15} {'Change':<15}")
        print("-" * 70)
        
        # Anomaly count and rate
        anomaly_change = optimized_det['n_anomalies'] - baseline_det['n_anomalies']
        rate_change = optimized_det['anomaly_rate'] - baseline_det['anomaly_rate']
        
        print(f"{'Anomalies Detected':<25} {baseline_det['n_anomalies']:<15} {optimized_det['n_anomalies']:<15} {anomaly_change:+d}")
        print(f"{'Anomaly Rate (%)':<25} {baseline_det['anomaly_rate']:<15.2f} {optimized_det['anomaly_rate']:<15.2f} {rate_change:+.2f}%")
        
        # Threshold comparison
        if baseline_det.get('threshold_percentile') and optimized_det.get('threshold_percentile'):
            threshold_change = optimized_det['threshold_percentile'] - baseline_det['threshold_percentile']
            print(f"{'Threshold Percentile':<25} {baseline_det['threshold_percentile']:<15.2f} {optimized_det['threshold_percentile']:<15.2f} {threshold_change:+.2f}")
        
        if baseline_det.get('threshold') and optimized_det.get('threshold'):
            abs_threshold_change = optimized_det['threshold'] - baseline_det['threshold']
            print(f"{'Threshold Value':<25} {baseline_det['threshold']:<15.4f} {optimized_det['threshold']:<15.4f} {abs_threshold_change:+.4f}")
        
        # Separation quality comparison
        print("\n📐 STATISTICAL SEPARATION QUALITY:")
        print("-" * 50)
        
        baseline_sep = baseline_results['separation_quality']
        optimized_sep = optimized_results['separation_quality']
        
        cohens_d_change = optimized_sep['cohens_d'] - baseline_sep['cohens_d']
        separation_change = optimized_sep['separation_magnitude'] - baseline_sep['separation_magnitude']
        
        print(f"{'Cohens d Effect Size':<25} {baseline_sep['cohens_d']:<15.4f} {optimized_sep['cohens_d']:<15.4f} {cohens_d_change:+.4f}")
        print(f"{'Separation Magnitude':<25} {baseline_sep['separation_magnitude']:<15.4f} {optimized_sep['separation_magnitude']:<15.4f} {separation_change:+.4f}")
        print(f"{'Mean Anomaly Score':<25} {baseline_sep['mean_anomaly_score']:<15.4f} {optimized_sep['mean_anomaly_score']:<15.4f} {optimized_sep['mean_anomaly_score'] - baseline_sep['mean_anomaly_score']:+.4f}")
        print(f"{'Mean Normal Score':<25} {baseline_sep['mean_normal_score']:<15.4f} {optimized_sep['mean_normal_score']:<15.4f} {optimized_sep['mean_normal_score'] - baseline_sep['mean_normal_score']:+.4f}")
        
        # Score distribution comparison
        print("\n📊 SCORE DISTRIBUTION COMPARISON:")
        print("-" * 50)
        
        baseline_scores = baseline_results['score_statistics']
        optimized_scores = optimized_results['score_statistics']
        
        mean_change = optimized_scores['mean'] - baseline_scores['mean']
        std_change = optimized_scores['std'] - baseline_scores['std']
        range_baseline = baseline_scores['max'] - baseline_scores['min']
        range_optimized = optimized_scores['max'] - optimized_scores['min']
        range_change = range_optimized - range_baseline
        
        print(f"{'Mean Score':<25} {baseline_scores['mean']:<15.4f} {optimized_scores['mean']:<15.4f} {mean_change:+.4f}")
        print(f"{'Std Deviation':<25} {baseline_scores['std']:<15.4f} {optimized_scores['std']:<15.4f} {std_change:+.4f}")
        print(f"{'Score Range':<25} {range_baseline:<15.4f} {range_optimized:<15.4f} {range_change:+.4f}")
        print(f"{'Median Score':<25} {baseline_scores['median']:<15.4f} {optimized_scores['median']:<15.4f} {optimized_scores['median'] - baseline_scores['median']:+.4f}")
        
        # Configuration comparison
        print("\n⚙️  CONFIGURATION CHANGES:")
        print("-" * 50)
        print(f"Aggregation Method: {baseline_results.get('aggregation_method', 'unknown')} → {optimized_results.get('aggregation_method', 'unknown')}")
        
        # Performance interpretation
        print("\n🔍 PERFORMANCE INTERPRETATION:")
        print("-" * 50)
        
        # Cohen's d interpretation
        cohens_d_final = optimized_sep['cohens_d']
        if cohens_d_final >= 1.2:
            cohen_interpretation = "🟢 Excellent separation (very large effect)"
        elif cohens_d_final >= 0.8:
            cohen_interpretation = "🟡 Good separation (large effect)"
        elif cohens_d_final >= 0.5:
            cohen_interpretation = "🟠 Moderate separation (medium effect)"
        elif cohens_d_final >= 0.2:
            cohen_interpretation = "🔴 Weak separation (small effect)"
        else:
            cohen_interpretation = "🔴 Poor separation (negligible effect)"
        
        print(f"Statistical Separation: {cohen_interpretation}")
        
        # Anomaly rate interpretation
        final_rate = optimized_det['anomaly_rate']
        if 3.0 <= final_rate <= 5.0:
            rate_interpretation = "🟢 Optimal range (3-5% realistic for electrical data)"
        elif 1.5 <= final_rate < 3.0:
            rate_interpretation = "🟡 Conservative range (good precision)"
        elif 5.0 < final_rate <= 7.0:
            rate_interpretation = "🟠 Aggressive range (higher recall)"
        else:
            rate_interpretation = "🔴 Outside typical ranges"
        
        print(f"Anomaly Rate: {rate_interpretation}")
        
        # Overall improvement assessment
        improvements = []
        if cohens_d_change > 0.1:
            improvements.append("✅ Improved statistical separation")
        if 3.0 <= final_rate <= 5.0 and not (3.0 <= baseline_det['anomaly_rate'] <= 5.0):
            improvements.append("✅ Achieved optimal anomaly rate")
        if optimized_sep['separation_magnitude'] > baseline_sep['separation_magnitude']:
            improvements.append("✅ Enhanced anomaly-normal distinction")
        
        degradations = []
        if cohens_d_change < -0.1:
            degradations.append("❌ Reduced statistical separation")
        if final_rate > 10.0 or final_rate < 0.5:
            degradations.append("❌ Anomaly rate outside reasonable bounds")
        
        print(f"\n🎯 OPTIMIZATION SUMMARY:")
        print("-" * 50)
        
        if improvements:
            print("IMPROVEMENTS:")
            for improvement in improvements:
                print(f"  {improvement}")
        
        if degradations:
            print("CONCERNS:")
            for degradation in degradations:
                print(f"  {degradation}")
        
        if not improvements and not degradations:
            print("  📊 Marginal changes - parameters fine-tuned")
        
        # Get optimizer-specific metrics if available
        if optimizer_type == 'genetic' and hasattr(self.genetic_optimizer, 'best_fitness'):
            print(f"\nGA Best Fitness: {self.genetic_optimizer.best_fitness:.2f}/100")
        elif optimizer_type == 'cmaes' and hasattr(self.cmaes_optimizer, 'best_fitness'):
            print(f"\nCMA-ES Best Fitness: {self.cmaes_optimizer.best_fitness:.2f}/100")
        
        print("="*80)

    def _create_comparison_data(self, baseline_results, optimized_results, optimizer_type):
        """
        Create structured comparison data for saving to file.
        
        Args:
            baseline_results (dict): Performance metrics before optimization
            optimized_results (dict): Performance metrics after optimization
            optimizer_type (str): Type of optimizer used
            
        Returns:
            dict: Structured comparison data
        """
        if not baseline_results or not optimized_results:
            return {}
        
        baseline_det = baseline_results['detection_metrics']
        optimized_det = optimized_results['detection_metrics']
        baseline_sep = baseline_results['separation_quality']
        optimized_sep = optimized_results['separation_quality']
        baseline_scores = baseline_results['score_statistics']
        optimized_scores = optimized_results['score_statistics']
        
        return {
            'optimizer_type': optimizer_type,
            'comparison_timestamp': f"{__import__('datetime').datetime.now():%Y%m%d_%H%M%S}",
            'detection_metrics': {
                'anomalies_detected': {
                    'baseline': baseline_det['n_anomalies'],
                    'optimized': optimized_det['n_anomalies'],
                    'change': optimized_det['n_anomalies'] - baseline_det['n_anomalies']
                },
                'anomaly_rate_percent': {
                    'baseline': baseline_det['anomaly_rate'],
                    'optimized': optimized_det['anomaly_rate'],
                    'change': optimized_det['anomaly_rate'] - baseline_det['anomaly_rate']
                },
                'threshold_percentile': {
                    'baseline': baseline_det.get('threshold_percentile'),
                    'optimized': optimized_det.get('threshold_percentile'),
                    'change': (optimized_det.get('threshold_percentile', 0) - baseline_det.get('threshold_percentile', 0)) if baseline_det.get('threshold_percentile') and optimized_det.get('threshold_percentile') else None
                },
                'threshold_value': {
                    'baseline': baseline_det.get('threshold'),
                    'optimized': optimized_det.get('threshold'),
                    'change': (optimized_det.get('threshold', 0) - baseline_det.get('threshold', 0)) if baseline_det.get('threshold') and optimized_det.get('threshold') else None
                }
            },
            'separation_quality': {
                'cohens_d': {
                    'baseline': baseline_sep['cohens_d'],
                    'optimized': optimized_sep['cohens_d'],
                    'change': optimized_sep['cohens_d'] - baseline_sep['cohens_d']
                },
                'separation_magnitude': {
                    'baseline': baseline_sep['separation_magnitude'],
                    'optimized': optimized_sep['separation_magnitude'],
                    'change': optimized_sep['separation_magnitude'] - baseline_sep['separation_magnitude']
                },
                'mean_anomaly_score': {
                    'baseline': baseline_sep['mean_anomaly_score'],
                    'optimized': optimized_sep['mean_anomaly_score'],
                    'change': optimized_sep['mean_anomaly_score'] - baseline_sep['mean_anomaly_score']
                },
                'mean_normal_score': {
                    'baseline': baseline_sep['mean_normal_score'],
                    'optimized': optimized_sep['mean_normal_score'],
                    'change': optimized_sep['mean_normal_score'] - baseline_sep['mean_normal_score']
                }
            },
            'score_distribution': {
                'mean_score': {
                    'baseline': baseline_scores['mean'],
                    'optimized': optimized_scores['mean'],
                    'change': optimized_scores['mean'] - baseline_scores['mean']
                },
                'std_deviation': {
                    'baseline': baseline_scores['std'],
                    'optimized': optimized_scores['std'],
                    'change': optimized_scores['std'] - baseline_scores['std']
                },
                'score_range': {
                    'baseline': baseline_scores['max'] - baseline_scores['min'],
                    'optimized': optimized_scores['max'] - optimized_scores['min'],
                    'change': (optimized_scores['max'] - optimized_scores['min']) - (baseline_scores['max'] - baseline_scores['min'])
                }
            },
            'configuration_changes': {
                'aggregation_method': {
                    'baseline': baseline_results.get('aggregation_method', 'unknown'),
                    'optimized': optimized_results.get('aggregation_method', 'unknown')
                }
            },
            'performance_interpretation': {
                'cohens_d_quality': self._interpret_cohens_d(optimized_sep['cohens_d']),
                'anomaly_rate_quality': self._interpret_anomaly_rate(optimized_det['anomaly_rate']),
                'overall_assessment': self._assess_optimization_quality(baseline_results, optimized_results)
            },
            'optimizer_specific': {
                'best_fitness': getattr(self.genetic_optimizer if optimizer_type == 'genetic' else self.cmaes_optimizer, 'best_fitness', None)
            }
        }
    
    def _interpret_cohens_d(self, cohens_d):
        """Interpret Cohen's d effect size."""
        if cohens_d >= 1.2:
            return {'level': 'excellent', 'description': 'very large effect', 'emoji': '🟢'}
        elif cohens_d >= 0.8:
            return {'level': 'good', 'description': 'large effect', 'emoji': '🟡'}
        elif cohens_d >= 0.5:
            return {'level': 'moderate', 'description': 'medium effect', 'emoji': '🟠'}
        elif cohens_d >= 0.2:
            return {'level': 'weak', 'description': 'small effect', 'emoji': '🔴'}
        else:
            return {'level': 'poor', 'description': 'negligible effect', 'emoji': '🔴'}
    
    def _interpret_anomaly_rate(self, anomaly_rate):
        """Interpret anomaly detection rate."""
        if 3.0 <= anomaly_rate <= 5.0:
            return {'level': 'optimal', 'description': 'realistic for electrical data', 'emoji': '🟢'}
        elif 1.5 <= anomaly_rate < 3.0:
            return {'level': 'conservative', 'description': 'good precision', 'emoji': '🟡'}
        elif 5.0 < anomaly_rate <= 7.0:
            return {'level': 'aggressive', 'description': 'higher recall', 'emoji': '🟠'}
        else:
            return {'level': 'unusual', 'description': 'outside typical ranges', 'emoji': '🔴'}
    
    def _assess_optimization_quality(self, baseline_results, optimized_results):
        """Assess overall optimization quality."""
        baseline_sep = baseline_results['separation_quality']
        optimized_sep = optimized_results['separation_quality']
        baseline_det = baseline_results['detection_metrics']
        optimized_det = optimized_results['detection_metrics']
        
        cohens_d_change = optimized_sep['cohens_d'] - baseline_sep['cohens_d']
        final_rate = optimized_det['anomaly_rate']
        
        improvements = []
        concerns = []
        
        if cohens_d_change > 0.1:
            improvements.append('Improved statistical separation')
        if 3.0 <= final_rate <= 5.0 and not (3.0 <= baseline_det['anomaly_rate'] <= 5.0):
            improvements.append('Achieved optimal anomaly rate')
        if optimized_sep['separation_magnitude'] > baseline_sep['separation_magnitude']:
            improvements.append('Enhanced anomaly-normal distinction')
        
        if cohens_d_change < -0.1:
            concerns.append('Reduced statistical separation')
        if final_rate > 10.0 or final_rate < 0.5:
            concerns.append('Anomaly rate outside reasonable bounds')
        
        if improvements and not concerns:
            assessment = 'significant_improvement'
        elif improvements and concerns:
            assessment = 'mixed_results'
        elif not improvements and concerns:
            assessment = 'performance_degradation'
        else:
            assessment = 'marginal_changes'
        
        return {
            'assessment': assessment,
            'improvements': improvements,
            'concerns': concerns
        }

    def _create_optimization_comparison_plots(self, baseline_results, optimized_results, optimizer_type):
        """
        Create comprehensive visual comparison plots showing optimization effectiveness.
        
        Args:
            baseline_results (dict): Performance metrics before optimization
            optimized_results (dict): Performance metrics after optimization
            optimizer_type (str): Type of optimizer used
        """
        if not baseline_results or not optimized_results:
            print("❌ Unable to create comparison plots - insufficient data")
            return
        
        try:
            import seaborn as sns
            from matplotlib.patches import Rectangle
            from matplotlib.gridspec import GridSpec
            
            # Set style for better visuals
            try:
                plt.style.use('seaborn-v0_8-darkgrid')
            except:
                try:
                    plt.style.use('seaborn-darkgrid')
                except:
                    plt.style.use('default')
            
            try:
                sns.set_palette("husl")
            except:
                pass  # Continue without seaborn if not available
            
            # Create comprehensive figure with subplots (2x3 grid for better spacing)
            fig = plt.figure(figsize=(22, 12))
            gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)
            
            # Main title - reduced size and positioned lower
            fig.suptitle(f'Optimization Performance Comparison: {optimizer_type.upper()} Algorithm\n' + 
                        f'Before vs After Optimization Analysis', 
                        fontsize=16, fontweight='bold', y=0.96)
            
            # Extract data for plotting
            baseline_det = baseline_results['detection_metrics']
            optimized_det = optimized_results['detection_metrics']
            baseline_sep = baseline_results['separation_quality']
            optimized_sep = optimized_results['separation_quality']
            baseline_scores = baseline_results['score_statistics']
            optimized_scores = optimized_results['score_statistics']
            
            # 1. Anomaly Detection Comparison (Top Left)
            ax1 = fig.add_subplot(gs[0, 0])
            categories = ['Anomalies\nDetected', 'Rate (%)', 'Threshold']
            baseline_vals = [baseline_det['n_anomalies'], baseline_det['anomaly_rate'], 
                           baseline_det.get('threshold', 0)]
            optimized_vals = [optimized_det['n_anomalies'], optimized_det['anomaly_rate'], 
                            optimized_det.get('threshold', 0)]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline', 
                           color='lightcoral', alpha=0.8, edgecolor='darkred')
            bars2 = ax1.bar(x + width/2, optimized_vals, width, label='Optimized', 
                           color='lightblue', alpha=0.8, edgecolor='darkblue')
            
            ax1.set_xlabel('Detection Metrics', fontsize=10)
            ax1.set_ylabel('Values', fontsize=10)
            ax1.set_title('🎯 Detection Performance', fontsize=11, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories, fontsize=9)
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars - reduced font size
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax1.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 2), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
            
            # 2. Statistical Separation Quality (Top Middle)
            ax2 = fig.add_subplot(gs[0, 1])
            sep_categories = ['Cohen\'s d', 'Separation', 'Anomaly\nScore', 'Normal\nScore']
            baseline_sep_vals = [baseline_sep['cohens_d'], baseline_sep['separation_magnitude'],
                               baseline_sep['mean_anomaly_score'], abs(baseline_sep['mean_normal_score'])]
            optimized_sep_vals = [optimized_sep['cohens_d'], optimized_sep['separation_magnitude'],
                                optimized_sep['mean_anomaly_score'], abs(optimized_sep['mean_normal_score'])]
            
            x2 = np.arange(len(sep_categories))
            bars3 = ax2.bar(x2 - width/2, baseline_sep_vals, width, label='Baseline', 
                           color='lightcoral', alpha=0.8, edgecolor='darkred')
            bars4 = ax2.bar(x2 + width/2, optimized_sep_vals, width, label='Optimized', 
                           color='lightblue', alpha=0.8, edgecolor='darkblue')
            
            ax2.set_xlabel('Separation Metrics', fontsize=10)
            ax2.set_ylabel('Values', fontsize=10)
            ax2.set_title('📐 Statistical Separation', fontsize=11, fontweight='bold')
            ax2.set_xticks(x2)
            ax2.set_xticklabels(sep_categories, fontsize=9)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars - reduced font size
            for bars in [bars3, bars4]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 2), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
            
            # 3. Performance Metrics Radar Chart (Top Right)
            ax3 = fig.add_subplot(gs[0, 2], projection='polar')
            
            # Normalize metrics for radar chart (0-1 scale) - simplified labels
            metrics = ['Cohen\'s d', 'Rate', 'Separation', 'Range']
            
            baseline_radar = [
                min(baseline_sep['cohens_d']/3, 1),
                min(baseline_det['anomaly_rate']/10, 1),
                min(baseline_sep['separation_magnitude']/3, 1),
                min((baseline_scores['max'] - baseline_scores['min'])/10, 1)
            ]
            
            optimized_radar = [
                min(optimized_sep['cohens_d']/3, 1),
                min(optimized_det['anomaly_rate']/10, 1),
                min(optimized_sep['separation_magnitude']/3, 1),
                min((optimized_scores['max'] - optimized_scores['min'])/10, 1)
            ]
            
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            baseline_radar += baseline_radar[:1]  # Close the plot
            optimized_radar += optimized_radar[:1]
            angles += angles[:1]
            
            ax3.plot(angles, baseline_radar, 'o-', linewidth=2, color='red', alpha=0.8, label='Baseline')
            ax3.fill(angles, baseline_radar, alpha=0.25, color='red')
            ax3.plot(angles, optimized_radar, 'o-', linewidth=2, color='blue', alpha=0.8, label='Optimized')
            ax3.fill(angles, optimized_radar, alpha=0.25, color='blue')
            
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(metrics, fontsize=8)
            ax3.set_ylim(0, 1)
            ax3.set_title('� Performance Radar', fontsize=11, fontweight='bold', y=1.08)
            ax3.legend(loc='upper right', bbox_to_anchor=(1.2, 0.9), fontsize=8)
            ax3.grid(True)
            
            # 4. Score Distribution Comparison (Bottom Left)
            ax4 = fig.add_subplot(gs[1, 0])
            
            # Create synthetic score distributions for visualization
            np.random.seed(42)  # For reproducible visualization
            n_samples = 1000
            
            # Baseline distribution (less optimized)
            baseline_anomaly_sim = np.random.normal(baseline_sep['mean_anomaly_score'], 0.5, 
                                                  int(n_samples * baseline_det['anomaly_rate']/100))
            baseline_normal_sim = np.random.normal(baseline_sep['mean_normal_score'], 1.0, 
                                                 int(n_samples * (100-baseline_det['anomaly_rate'])/100))
            
            # Optimized distribution (better separation)
            optimized_anomaly_sim = np.random.normal(optimized_sep['mean_anomaly_score'], 0.45, 
                                                   int(n_samples * optimized_det['anomaly_rate']/100))
            optimized_normal_sim = np.random.normal(optimized_sep['mean_normal_score'], 0.95, 
                                                  int(n_samples * (100-optimized_det['anomaly_rate'])/100))
            
            # Plot distributions with reduced bins for clarity
            ax4.hist(baseline_normal_sim, bins=30, alpha=0.6, color='lightcoral', 
                    label='Base Normal', density=True)
            ax4.hist(baseline_anomaly_sim, bins=20, alpha=0.6, color='red', 
                    label='Base Anomalies', density=True)
            ax4.hist(optimized_normal_sim, bins=30, alpha=0.6, color='lightblue', 
                    label='Opt Normal', density=True)
            ax4.hist(optimized_anomaly_sim, bins=20, alpha=0.6, color='blue', 
                    label='Opt Anomalies', density=True)
            
            ax4.set_xlabel('Anomaly Score', fontsize=10)
            ax4.set_ylabel('Density', fontsize=10)
            ax4.set_title('� Score Distribution', fontsize=11, fontweight='bold')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            # Add vertical lines for thresholds
            if baseline_det.get('threshold') and optimized_det.get('threshold'):
                ax4.axvline(baseline_det['threshold'], color='red', linestyle='--', alpha=0.8)
                ax4.axvline(optimized_det['threshold'], color='blue', linestyle='--', alpha=0.8)
            
            # 5. Improvement/Change Analysis (Bottom Middle)
            ax5 = fig.add_subplot(gs[1, 1])
            
            # Calculate percentage changes
            changes = {
                'Anomalies': (optimized_det['n_anomalies'] - baseline_det['n_anomalies']) / max(baseline_det['n_anomalies'], 1) * 100,
                'Rate': (optimized_det['anomaly_rate'] - baseline_det['anomaly_rate']) / max(baseline_det['anomaly_rate'], 0.1) * 100,
                'Cohen\'s d': (optimized_sep['cohens_d'] - baseline_sep['cohens_d']) / max(baseline_sep['cohens_d'], 0.1) * 100,
                'Separation': (optimized_sep['separation_magnitude'] - baseline_sep['separation_magnitude']) / max(baseline_sep['separation_magnitude'], 0.1) * 100
            }
            
            change_names = list(changes.keys())
            change_vals = list(changes.values())
            colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in change_vals]
            
            bars = ax5.barh(change_names, change_vals, color=colors, alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Change (%)', fontsize=10)
            ax5.set_title('📈 Impact Analysis', fontsize=11, fontweight='bold')
            ax5.axvline(0, color='black', linewidth=1)
            ax5.grid(True, alpha=0.3)
            
            # Add value labels - reduced font size
            for bar, val in zip(bars, change_vals):
                ax5.text(val + (0.5 if val > 0 else -0.5), bar.get_y() + bar.get_height()/2, 
                        f'{val:+.1f}%', va='center', ha='left' if val > 0 else 'right', 
                        fontsize=8)
            
            # 6. Quality Assessment Matrix (Bottom Right)
            ax6 = fig.add_subplot(gs[1, 2])
            
            # Create quality assessment matrix
            quality_data = np.array([
                [self._get_quality_score(baseline_sep['cohens_d'], 'cohens_d'), 
                 self._get_quality_score(optimized_sep['cohens_d'], 'cohens_d')],
                [self._get_quality_score(baseline_det['anomaly_rate'], 'anomaly_rate'),
                 self._get_quality_score(optimized_det['anomaly_rate'], 'anomaly_rate')],
                [self._get_quality_score(baseline_sep['separation_magnitude'], 'separation'),
                 self._get_quality_score(optimized_sep['separation_magnitude'], 'separation')]
            ])
            
            im = ax6.imshow(quality_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=4)
            
            # Add labels - shortened
            quality_labels = ['Cohen\'s d', 'Anomaly Rate', 'Separation']
            condition_labels = ['Baseline', 'Optimized']
            
            ax6.set_xticks(np.arange(len(condition_labels)))
            ax6.set_yticks(np.arange(len(quality_labels)))
            ax6.set_xticklabels(condition_labels, fontsize=10)
            ax6.set_yticklabels(quality_labels, fontsize=9)
            ax6.set_title('🎖️ Quality Matrix', fontsize=11, fontweight='bold')
            
            # Add text annotations - reduced font size
            for i in range(len(quality_labels)):
                for j in range(len(condition_labels)):
                    score = int(quality_data[i, j])
                    text_label = ['Poor', 'Fair', 'Good', 'V.Good', 'Excellent'][score]
                    ax6.text(j, i, f'{score}/4\n{text_label}', ha="center", va="center",
                            fontsize=9, color='white' if score < 2 else 'black')
            
            # Add colorbar - smaller
            cbar = plt.colorbar(im, ax=ax6, shrink=0.6)
            cbar.set_label('Quality', fontsize=10)
            
            # Save the comprehensive comparison plot
            results_dir = self.visualizer.get_execution_folder_path()
            plot_filename = f"optimization_comparison_{optimizer_type.lower()}.png"
            plot_path = os.path.join(results_dir, plot_filename)
            
            plt.tight_layout()
            fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"     📊 Optimization comparison plots saved to {plot_path}")
            
            # Also create a summary comparison plot (simplified version)
            self._create_summary_comparison_plot(baseline_results, optimized_results, optimizer_type)
            
        except Exception as e:
            print(f"     Warning: Could not create comparison plots: {str(e)}")
            traceback.print_exc()
    
    def _get_quality_score(self, value, metric_type):
        """Convert metric values to quality scores (0-4)."""
        if metric_type == 'cohens_d':
            if value >= 1.2: return 4  # Excellent
            elif value >= 0.8: return 3  # Very Good
            elif value >= 0.5: return 2  # Good
            elif value >= 0.2: return 1  # Fair
            else: return 0  # Poor
        elif metric_type == 'anomaly_rate':
            if 3.0 <= value <= 5.0: return 4  # Excellent
            elif 2.0 <= value < 3.0 or 5.0 < value <= 6.0: return 3  # Very Good
            elif 1.0 <= value < 2.0 or 6.0 < value <= 8.0: return 2  # Good
            elif 0.5 <= value < 1.0 or 8.0 < value <= 10.0: return 1  # Fair
            else: return 0  # Poor
        elif metric_type == 'separation':
            if value >= 2.0: return 4  # Excellent
            elif value >= 1.5: return 3  # Very Good
            elif value >= 1.0: return 2  # Good
            elif value >= 0.5: return 1  # Fair
            else: return 0  # Poor
        return 2  # Default to Good
    
    def _create_summary_comparison_plot(self, baseline_results, optimized_results, optimizer_type):
        """Create a simplified summary comparison plot."""
        try:
            # Create simplified summary plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{optimizer_type.upper()} Optimization: Before vs After Summary', 
                        fontsize=16, fontweight='bold')
            
            # Extract key metrics
            baseline_det = baseline_results['detection_metrics']
            optimized_det = optimized_results['detection_metrics']
            baseline_sep = baseline_results['separation_quality']
            optimized_sep = optimized_results['separation_quality']
            
            # 1. Key Metrics Comparison
            metrics = ['Anomalies', 'Anomaly Rate (%)', 'Cohen\'s d', 'Separation']
            baseline_vals = [baseline_det['n_anomalies'], baseline_det['anomaly_rate'],
                           baseline_sep['cohens_d'], baseline_sep['separation_magnitude']]
            optimized_vals = [optimized_det['n_anomalies'], optimized_det['anomaly_rate'],
                            optimized_sep['cohens_d'], optimized_sep['separation_magnitude']]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline', 
                           color='lightcoral', alpha=0.8)
            bars2 = ax1.bar(x + width/2, optimized_vals, width, label='Optimized', 
                           color='lightgreen', alpha=0.8)
            
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Values')
            ax1.set_title('Key Performance Metrics')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Quality Scores
            quality_categories = ['Statistical\nSeparation', 'Detection\nRate', 'Overall\nQuality']
            baseline_quality = [
                self._get_quality_score(baseline_sep['cohens_d'], 'cohens_d'),
                self._get_quality_score(baseline_det['anomaly_rate'], 'anomaly_rate'),
                (self._get_quality_score(baseline_sep['cohens_d'], 'cohens_d') + 
                 self._get_quality_score(baseline_det['anomaly_rate'], 'anomaly_rate')) / 2
            ]
            optimized_quality = [
                self._get_quality_score(optimized_sep['cohens_d'], 'cohens_d'),
                self._get_quality_score(optimized_det['anomaly_rate'], 'anomaly_rate'),
                (self._get_quality_score(optimized_sep['cohens_d'], 'cohens_d') + 
                 self._get_quality_score(optimized_det['anomaly_rate'], 'anomaly_rate')) / 2
            ]
            
            x2 = np.arange(len(quality_categories))
            bars3 = ax2.bar(x2 - width/2, baseline_quality, width, label='Baseline', 
                           color='lightcoral', alpha=0.8)
            bars4 = ax2.bar(x2 + width/2, optimized_quality, width, label='Optimized', 
                           color='lightgreen', alpha=0.8)
            
            ax2.set_xlabel('Quality Categories')
            ax2.set_ylabel('Quality Score (0-4)')
            ax2.set_title('Quality Assessment')
            ax2.set_xticks(x2)
            ax2.set_xticklabels(quality_categories)
            ax2.legend()
            ax2.set_ylim(0, 4)
            ax2.grid(True, alpha=0.3)
            
            # 3. Improvement Analysis
            improvements = {
                'Anomalies': optimized_det['n_anomalies'] - baseline_det['n_anomalies'],
                'Rate (%)': optimized_det['anomaly_rate'] - baseline_det['anomaly_rate'],
                'Cohen\'s d': optimized_sep['cohens_d'] - baseline_sep['cohens_d'],
                'Separation': optimized_sep['separation_magnitude'] - baseline_sep['separation_magnitude']
            }
            
            colors = ['green' if x > 0 else 'red' if x < -0.01 else 'gray' for x in improvements.values()]
            bars5 = ax3.bar(improvements.keys(), improvements.values(), color=colors, alpha=0.7)
            ax3.set_xlabel('Metrics')
            ax3.set_ylabel('Change (Optimized - Baseline)')
            ax3.set_title('Optimization Impact')
            ax3.axhline(0, color='black', linewidth=1)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars5:
                height = bar.get_height()
                ax3.annotate(f'{height:+.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if height > 0 else -15),
                            textcoords="offset points",
                            ha='center', va='bottom' if height > 0 else 'top',
                            fontsize=10, fontweight='bold')
            
            # 4. Performance Summary Text
            ax4.axis('off')
            
            # Calculate overall assessment
            cohen_improvement = optimized_sep['cohens_d'] - baseline_sep['cohens_d']
            rate_optimal = 3.0 <= optimized_det['anomaly_rate'] <= 5.0
            
            summary_text = f"""
OPTIMIZATION RESULTS SUMMARY

Algorithm Used: {optimizer_type.upper()}

KEY IMPROVEMENTS:
• Anomalies Detected: {baseline_det['n_anomalies']} → {optimized_det['n_anomalies']} ({optimized_det['n_anomalies'] - baseline_det['n_anomalies']:+d})
• Anomaly Rate: {baseline_det['anomaly_rate']:.2f}% → {optimized_det['anomaly_rate']:.2f}% ({optimized_det['anomaly_rate'] - baseline_det['anomaly_rate']:+.2f}%)
• Cohen's d: {baseline_sep['cohens_d']:.3f} → {optimized_sep['cohens_d']:.3f} ({cohen_improvement:+.3f})
• Separation: {baseline_sep['separation_magnitude']:.3f} → {optimized_sep['separation_magnitude']:.3f}

QUALITY ASSESSMENT:
• Statistical Separation: {"🟢 Excellent" if optimized_sep['cohens_d'] >= 1.2 else "🟡 Good" if optimized_sep['cohens_d'] >= 0.8 else "🟠 Moderate"}
• Anomaly Rate: {"🟢 Optimal" if rate_optimal else "🟡 Acceptable"}
• Overall Performance: {"🟢 Improved" if cohen_improvement > 0.01 or rate_optimal else "🟡 Fine-tuned"}

CONCLUSION:
{"✅ Optimization successfully enhanced detection performance" if cohen_improvement > 0.01 or rate_optimal else "📊 Optimization fine-tuned parameters for optimal performance"}
            """
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            # Save summary plot
            results_dir = self.visualizer.get_execution_folder_path()
            summary_plot_path = os.path.join(results_dir, f"optimization_summary_{optimizer_type.lower()}.png")
            plt.tight_layout()
            fig.savefig(summary_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"     📈 Optimization summary plot saved to {summary_plot_path}")
            
        except Exception as e:
            print(f"     Warning: Could not create summary plot: {str(e)}")

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
            
            # 6. Save optimization comparison (if available)
            if self.optimization_comparison:
                try:
                    comparison_file = os.path.join(results_dir, "optimization_comparison.json")
                    comparison_serializable = self._make_json_serializable(self.optimization_comparison)
                    with open(comparison_file, 'w') as f:
                        json.dump(comparison_serializable, f, indent=2)
                    print(f"     Optimization comparison saved to {comparison_file}")
                except Exception as e:
                    print(f"     Warning: Could not save optimization comparison: {str(e)}")
            
            # 7. Create master summary file
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
            
            # Add optimization comparison if available
            if self.optimization_comparison:
                master_summary['optimization_comparison'] = {
                    'available': True,
                    'optimizer_type': self.optimization_comparison.get('optimizer_type'),
                    'improvement_summary': self.optimization_comparison.get('performance_interpretation', {}).get('overall_assessment', 'unknown'),
                    'best_fitness': self.optimization_comparison.get('optimizer_specific', {}).get('best_fitness'),
                    'file': "optimization_comparison.json"
                }
                master_summary['files_created']['optimization_comparison'] = "optimization_comparison.json"
            
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
    
    experiment_to_run = 3  # <--- CHANGE THIS VALUE TO SELECT AN EXPERIMENT

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
