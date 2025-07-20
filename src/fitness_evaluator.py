"""
Shared fitness function evaluation for both GA and CMA-ES optimizers.

This module provides a unified fitness evaluation system that both optimizers
can use, eliminating code duplication and ensuring consistency.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats


class FitnessEvaluator:
    """
    Unified fitness evaluation system for BN weight optimization.
    
    Uses a 4-component fitness function:
    1. Separation Quality (45%) - Cohen's d effect size
    2. Detection Rate (25%) - Multi-modal Gaussian targeting  
    3. Threshold Robustness (20%) - Coefficient of variation
    4. Weight Diversity (10%) - Shannon entropy
    """
    
    def __init__(self, fitness_components: Dict[str, float]):
        """
        Initialize fitness evaluator with component weights.
        
        Args:
            fitness_components: Dictionary with fitness component weights
        """
        self.fitness_components = fitness_components
        self._validate_weights()
    
    def _validate_weights(self):
        """Validate and normalize fitness component weights."""
        total_weight = sum(self.fitness_components.values())
        if abs(total_weight - 1.0) > 1e-6:
            print(f"⚠️  Fitness weights sum to {total_weight:.3f}, normalizing...")
            for key in self.fitness_components:
                self.fitness_components[key] /= total_weight
    
    def evaluate_fitness(self, anomaly_scores: np.ndarray, 
                        anomaly_indices: np.ndarray, params: Dict) -> float:
        """
        Calculate complete fitness score using 4-component function.
        
        Args:
            anomaly_scores: Computed anomaly scores
            anomaly_indices: Indices of detected anomalies
            params: Dictionary with 'bn_weights' and 'threshold_percentile'
            
        Returns:
            Fitness score in range [0, 100]
        """
        n_samples = len(anomaly_scores)
        n_anomalies = len(anomaly_indices)
        
        # Handle degenerate cases
        if n_anomalies == 0:
            return 0.0  # No anomalies detected
        if n_anomalies >= n_samples * 0.5:
            return 0.0  # Too many anomalies (likely poor threshold)
            
        # Split data into anomaly and normal groups
        normal_indices = np.setdiff1d(np.arange(n_samples), anomaly_indices)
        anomaly_values = anomaly_scores[anomaly_indices]
        normal_values = anomaly_scores[normal_indices]
        
        # Component scores
        components = {}
        
        # 1. SEPARATION QUALITY (45 points) - Most important component
        components['separation'] = self._evaluate_separation_quality(normal_values, anomaly_values)
        
        # 2. DETECTION RATE (25 points) - Target realistic anomaly rates
        anomaly_rate = (n_anomalies / n_samples) * 100
        components['detection_rate'] = self._evaluate_detection_rate(anomaly_rate)
        
        # 3. THRESHOLD ROBUSTNESS (20 points) - Stability across thresholds
        components['robustness'] = self._evaluate_threshold_robustness(
            anomaly_scores, params['threshold_percentile']
        )
        
        # 4. WEIGHT DIVERSITY (10 points) - Encourage diverse weight usage
        components['diversity'] = self._evaluate_weight_diversity(params['bn_weights'])
        
        # Calculate weighted total fitness
        total_fitness = (
            components['separation'] * self.fitness_components['separation_quality'] +
            components['detection_rate'] * self.fitness_components['detection_rate'] +
            components['robustness'] * self.fitness_components['threshold_robustness'] +
            components['diversity'] * self.fitness_components['weight_diversity']
        ) * 100
        
        return min(100.0, max(0.0, total_fitness))
    
    def _evaluate_separation_quality(self, normal_values: np.ndarray, 
                                   anomaly_values: np.ndarray) -> float:
        """
        Evaluate statistical separation using Cohen's d effect size.
        
        Args:
            normal_values: Normal sample scores
            anomaly_values: Anomaly sample scores
            
        Returns:
            Separation quality score [0, 1]
        """
        try:
            if len(normal_values) < 2 or len(anomaly_values) < 2:
                return 0.0
            
            # Cohen's d calculation
            mean_normal = np.mean(normal_values)
            mean_anomaly = np.mean(anomaly_values)
            std_normal = np.std(normal_values, ddof=1)
            std_anomaly = np.std(anomaly_values, ddof=1)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(
                ((len(anomaly_values) - 1) * std_anomaly**2 + 
                 (len(normal_values) - 1) * std_normal**2) /
                (len(anomaly_values) + len(normal_values) - 2)
            )
            
            if pooled_std < 1e-10:
                return 0.0
            
            cohens_d = abs(mean_anomaly - mean_normal) / pooled_std
            
            # Transform Cohen's d to [0,1] score with bonus for excellent separation
            if cohens_d > 1.5:
                return min(1.0, cohens_d / 2.0 + 0.1)  # Bonus for very large effect
            else:
                return min(1.0, cohens_d / 2.0)
            
        except Exception:
            return 0.0
    
    def _evaluate_detection_rate(self, anomaly_rate: float) -> float:
        """
        Evaluate anomaly detection rate using multi-modal Gaussian functions.
        
        Args:
            anomaly_rate: Percentage of samples detected as anomalies
            
        Returns:
            Detection rate quality score [0, 1]
        """
        try:
            # Multi-modal Gaussian functions for different target rates
            
            # Primary target: 4% (σ=1.0) - Realistic for electrical data
            primary_score = np.exp(-0.5 * ((anomaly_rate - 4.0) / 1.0) ** 2)
            
            # Secondary target: 1.5% (σ=0.5) - Conservative detection  
            secondary_score = 0.8 * np.exp(-0.5 * ((anomaly_rate - 1.5) / 0.5) ** 2)
            
            # Tertiary target: 7% (σ=1.5) - Aggressive detection
            tertiary_score = 0.6 * np.exp(-0.5 * ((anomaly_rate - 7.0) / 1.5) ** 2)
            
            # Take maximum of the three target functions
            return max(primary_score, secondary_score, tertiary_score)
            
        except Exception:
            return 0.0
    
    def _evaluate_threshold_robustness(self, anomaly_scores: np.ndarray, 
                                     target_percentile: float) -> float:
        """
        Evaluate threshold robustness using coefficient of variation.
        
        Args:
            anomaly_scores: All anomaly scores
            target_percentile: Target threshold percentile
            
        Returns:
            Robustness score [0, 1]
        """
        try:
            # Test neighboring thresholds
            test_percentiles = [
                max(1.0, target_percentile - 0.5),
                target_percentile,
                min(10.0, target_percentile + 0.5)
            ]
            
            anomaly_counts = []
            for percentile in test_percentiles:
                threshold = np.percentile(anomaly_scores, 100 - percentile)
                count = np.sum(anomaly_scores > threshold)
                anomaly_counts.append(count)
            
            # Calculate coefficient of variation
            mean_count = np.mean(anomaly_counts)
            std_count = np.std(anomaly_counts)
            
            if mean_count == 0:
                return 0.0
                
            cv = std_count / mean_count
            
            # Transform CV to robustness score (lower CV = higher robustness)
            return min(1.0, np.exp(-5.0 * cv))
            
        except Exception:
            return 0.0
    
    def _evaluate_weight_diversity(self, bn_weights: np.ndarray) -> float:
        """
        Evaluate weight diversity using Shannon entropy.
        
        Args:
            bn_weights: BN weight vector
            
        Returns:
            Diversity score [0, 1]
        """
        try:
            # Shannon entropy calculation
            epsilon = 1e-8
            entropy = -np.sum(bn_weights * np.log(bn_weights + epsilon))
            
            # Normalize by maximum possible entropy
            k = len(bn_weights)
            max_entropy = np.log(k)
            
            if max_entropy == 0:
                return 0.0
                
            normalized_entropy = entropy / max_entropy
            
            # Bonus for balanced weight distribution
            effective_weights = np.sum(bn_weights > 0.05)  # Count meaningful weights
            if 3 <= effective_weights <= 0.7 * k:
                return min(1.0, normalized_entropy + 0.1)  # 10% bonus
            else:
                return normalized_entropy
            
        except Exception:
            return 0.0
    
    def compute_weighted_anomaly_scores(self, likelihood_scores: pd.DataFrame, 
                                      bn_weights: np.ndarray) -> np.ndarray:
        """
        Compute weighted anomaly scores from likelihood matrix.
        
        Args:
            likelihood_scores: Likelihood matrix (N x K)
            bn_weights: BN weight vector (K,)
            
        Returns:
            Weighted anomaly scores (N,)
        """
        # Compute weighted scores (negative log-likelihood)
        weighted_scores = -np.dot(likelihood_scores.values, bn_weights)
        
        # Standardize scores (z-score normalization)
        epsilon = 1e-8
        mean_score = np.mean(weighted_scores)
        std_score = np.std(weighted_scores)
        
        if std_score < epsilon:
            return np.zeros_like(weighted_scores)  # All scores identical
        
        standardized_scores = (weighted_scores - mean_score) / (std_score + epsilon)
        return standardized_scores
    
    def determine_threshold(self, anomaly_scores: np.ndarray, 
                          threshold_percentile: float) -> float:
        """
        Determine anomaly threshold from scores and percentile.
        
        Args:
            anomaly_scores: Computed anomaly scores
            threshold_percentile: Percentile for threshold (1-10%)
            
        Returns:
            Threshold value for anomaly detection
        """
        return np.percentile(anomaly_scores, 100 - threshold_percentile)
