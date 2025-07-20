import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import List, Tuple, Dict, Optional
import warnings

class AnomalyDetector:
    """
    Detects anomalies by aggregating likelihood scores and applying thresholds.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the anomaly detector.
        
        Args:
            config (Dict): Configuration for anomaly detection
        """
        self.config = config
        self.threshold = None
        self.scaler = None
        self.isolation_forest = None
        
    def detect_anomalies(self, likelihood_scores: pd.DataFrame, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies based on likelihood scores.
        
        Args:
            likelihood_scores (pd.DataFrame): Likelihood matrix from BN learning
            verbose (bool): Whether to print detailed information
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (anomaly_scores, anomaly_indices)
        """
        if verbose:
            print("     Detecting anomalies...")
        
        # Step 1: Aggregate likelihood scores
        aggregated_scores = self._aggregate_scores(likelihood_scores, verbose)
        
        # Step 2: Compute anomaly scores (lower likelihood = higher anomaly score)
        anomaly_scores = self._compute_anomaly_scores(aggregated_scores, verbose)
        
        # Step 3: Determine threshold and detect anomalies
        threshold = self._determine_threshold(anomaly_scores, verbose)
        anomaly_indices = self._identify_anomalies(anomaly_scores, threshold)
        
        if verbose:
            self._print_detection_stats(anomaly_scores, threshold, anomaly_indices, likelihood_scores)
        
        return anomaly_scores, anomaly_indices
    
    def _aggregate_scores(self, likelihood_scores: pd.DataFrame, verbose: bool = True) -> np.ndarray:
        """
        Aggregate likelihood scores across groups.
        
        Args:
            likelihood_scores (pd.DataFrame): Likelihood matrix
            verbose (bool): Whether to print information
            
        Returns:
            np.ndarray: Aggregated scores
        """
        method = self.config.get('aggregation_method', 'mean')
        
        if method == 'mean':
            # Average likelihood across all groups
            aggregated = likelihood_scores.mean(axis=1).values
        elif method == 'min':
            # Minimum likelihood (most pessimistic)
            aggregated = likelihood_scores.min(axis=1).values
        elif method == 'max':
            # Maximum likelihood (most optimistic)
            aggregated = likelihood_scores.max(axis=1).values
        elif method == 'median':
            # Median likelihood
            aggregated = likelihood_scores.median(axis=1).values
        elif method == 'weighted':
            # Use custom BN weights if available, otherwise uniform weights
            if 'bn_weights' in self.config and self.config['bn_weights'] is not None:
                weights = self.config['bn_weights']
                if len(weights) != likelihood_scores.shape[1]:
                    print(f"     Warning: BN weights length ({len(weights)}) doesn't match likelihood matrix ({likelihood_scores.shape[1]})")
                    weights = np.ones(likelihood_scores.shape[1]) / likelihood_scores.shape[1]
            else:
                weights = np.ones(likelihood_scores.shape[1]) / likelihood_scores.shape[1]
            
            # Ensure weights are normalized
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            aggregated = np.average(likelihood_scores.values, axis=1, weights=weights)
            
            if verbose:
                print(f"     Using BN weights - Max: {np.max(weights):.3f}, Min: {np.min(weights):.3f}")
        elif method == 'sum':
            # Sum of log-likelihoods
            aggregated = likelihood_scores.sum(axis=1).values
        else:
            if verbose:
                print(f"     Unknown aggregation method '{method}', using mean")
            aggregated = likelihood_scores.mean(axis=1).values
        
        if verbose:
            print(f"     Aggregated scores using method: {method}")
        return aggregated
    
    def _compute_anomaly_scores(self, aggregated_scores: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Convert aggregated likelihood scores to anomaly scores.
        
        Args:
            aggregated_scores (np.ndarray): Aggregated likelihood scores
            verbose (bool): Whether to print information
            
        Returns:
            np.ndarray: Anomaly scores (higher = more anomalous)
        """
        # Convert log-likelihoods to anomaly scores
        # Lower likelihood -> Higher anomaly score
        
        # Method 1: Negative log-likelihood
        anomaly_scores = -aggregated_scores
        
        # Method 2: Z-score based transformation
        if self.config.get('use_zscore_transformation', True):
            mean_score = np.mean(anomaly_scores)
            std_score = np.std(anomaly_scores)
            if std_score > 0:
                anomaly_scores = (anomaly_scores - mean_score) / std_score
        
        # Method 3: Rank-based transformation
        if self.config.get('use_rank_transformation', False):
            from scipy.stats import rankdata
            anomaly_scores = rankdata(anomaly_scores) / len(anomaly_scores)
        
        return anomaly_scores
    
    def _determine_threshold(self, anomaly_scores: np.ndarray, verbose: bool = True) -> float:
        """
        Determine threshold for anomaly detection.
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            verbose (bool): Whether to print information
            
        Returns:
            float: Threshold value
        """
        method = self.config.get('threshold_method', 'percentile')
        
        if method == 'percentile':
            # Use percentile-based threshold
            percentile = self.config.get('threshold_percentile', 5)
            threshold = np.percentile(anomaly_scores, 100 - percentile)
            
        elif method == 'std':
            # Use standard deviation based threshold
            n_std = self.config.get('threshold_std', 2)
            mean_score = np.mean(anomaly_scores)
            std_score = np.std(anomaly_scores)
            threshold = mean_score + n_std * std_score
            
        elif method == 'iqr':
            # Use IQR-based threshold (outlier detection)
            Q1 = np.percentile(anomaly_scores, 25)
            Q3 = np.percentile(anomaly_scores, 75)
            IQR = Q3 - Q1
            threshold = Q3 + 1.5 * IQR
            
        elif method == 'isolation_forest':
            # Use Isolation Forest for threshold determination
            threshold = self._isolation_forest_threshold(anomaly_scores)
            
        elif method == 'adaptive':
            # Adaptive threshold based on score distribution
            threshold = self._adaptive_threshold(anomaly_scores)
            
        else:
            if verbose:
                print(f"     Unknown threshold method '{method}', using percentile")
            percentile = self.config.get('threshold_percentile', 5)
            threshold = np.percentile(anomaly_scores, 100 - percentile)
        
        self.threshold = threshold
        return float(threshold)
    
    def _isolation_forest_threshold(self, anomaly_scores: np.ndarray) -> float:
        """
        Use Isolation Forest to determine threshold.
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            
        Returns:
            float: Threshold value
        """
        try:
            # Reshape for sklearn
            scores_reshaped = anomaly_scores.reshape(-1, 1)
            
            # Fit Isolation Forest
            contamination = self.config.get('threshold_percentile', 5) / 100
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            predictions = iso_forest.fit_predict(scores_reshaped)
            
            # Get threshold from decision function
            decision_scores = iso_forest.decision_function(scores_reshaped)
            threshold_idx = np.where(predictions == -1)[0]
            
            if len(threshold_idx) > 0:
                threshold = np.min(anomaly_scores[threshold_idx])
            else:
                # Fallback to percentile method
                percentile = self.config.get('threshold_percentile', 5)
                threshold = np.percentile(anomaly_scores, 100 - percentile)
            
            self.isolation_forest = iso_forest
            return threshold
            
        except Exception as e:
            print(f"     Isolation Forest threshold failed: {str(e)}, using percentile")
            percentile = self.config.get('threshold_percentile', 5)
            return np.percentile(anomaly_scores, 100 - percentile)
    
    def _adaptive_threshold(self, anomaly_scores: np.ndarray) -> float:
        """
        Adaptive threshold based on score distribution.
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            
        Returns:
            float: Threshold value
        """
        try:
            # Test for normality
            _, p_value = stats.normaltest(anomaly_scores)
            
            if p_value > 0.05:  # Approximately normal
                # Use z-score based threshold
                mean_score = np.mean(anomaly_scores)
                std_score = np.std(anomaly_scores)
                threshold = mean_score + 2 * std_score
            else:
                # Use robust percentile-based threshold
                percentile = self.config.get('threshold_percentile', 5)
                threshold = np.percentile(anomaly_scores, 100 - percentile)
            
            return float(threshold)
            
        except Exception:
            # Fallback to percentile method
            percentile = self.config.get('threshold_percentile', 5)
            return float(np.percentile(anomaly_scores, 100 - percentile))
    
    def _identify_anomalies(self, anomaly_scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Identify anomalies based on threshold.
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            threshold (float): Threshold value
            
        Returns:
            np.ndarray: Indices of anomalous samples
        """
        anomaly_mask = anomaly_scores > threshold
        anomaly_indices = np.where(anomaly_mask)[0]
        
        return anomaly_indices
    
    def update_parameters(self, new_params: Dict) -> None:
        """
        Update detection parameters (for GA optimization).
        
        Args:
            new_params (Dict): New parameters
        """
        if 'threshold' in new_params:
            self.threshold = new_params['threshold']
        
        if 'aggregation_method' in new_params:
            self.config['aggregation_method'] = new_params['aggregation_method']
        
        if 'threshold_percentile' in new_params:
            self.config['threshold_percentile'] = new_params['threshold_percentile']
    
    def get_anomaly_statistics(self, anomaly_scores: np.ndarray, 
                             anomaly_indices: np.ndarray) -> Dict:
        """
        Get detailed statistics about detected anomalies.
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            anomaly_indices (np.ndarray): Anomaly indices
            
        Returns:
            Dict: Anomaly statistics
        """
        stats_dict = {
            'total_samples': len(anomaly_scores),
            'total_anomalies': len(anomaly_indices),
            'anomaly_rate': len(anomaly_indices) / len(anomaly_scores) * 100,
            'threshold': self.threshold,
            'score_statistics': {
                'mean': np.mean(anomaly_scores),
                'std': np.std(anomaly_scores),
                'min': np.min(anomaly_scores),
                'max': np.max(anomaly_scores),
                'median': np.median(anomaly_scores),
                'q25': np.percentile(anomaly_scores, 25),
                'q75': np.percentile(anomaly_scores, 75)
            }
        }
        
        if len(anomaly_indices) > 0:
            anomaly_score_values = anomaly_scores[anomaly_indices]
            stats_dict['anomaly_score_statistics'] = {
                'mean': np.mean(anomaly_score_values),
                'std': np.std(anomaly_score_values),
                'min': np.min(anomaly_score_values),
                'max': np.max(anomaly_score_values),
                'median': np.median(anomaly_score_values)
            }
            
            # Top anomalies
            top_indices = anomaly_indices[np.argsort(anomaly_scores[anomaly_indices])[-10:]]
            stats_dict['top_anomalies'] = {
                'indices': top_indices.tolist(),
                'scores': anomaly_scores[top_indices].tolist()
            }
        
        return stats_dict
    
    def classify_new_samples(self, new_likelihood_scores: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify new samples as anomalous or normal.
        
        Args:
            new_likelihood_scores (pd.DataFrame): Likelihood scores for new samples
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (anomaly_scores, is_anomaly)
        """
        if self.threshold is None:
            raise ValueError("Model not trained. Run detect_anomalies first.")
        
        # Aggregate scores for new samples
        aggregated_scores = self._aggregate_scores(new_likelihood_scores)
        
        # Compute anomaly scores
        anomaly_scores = self._compute_anomaly_scores(aggregated_scores)
        
        # Classify based on learned threshold
        is_anomaly = anomaly_scores > self.threshold
        
        return anomaly_scores, is_anomaly
    
    def explain_anomalies(self, likelihood_scores: pd.DataFrame, 
                         anomaly_indices: np.ndarray, top_k: int = 5) -> Dict:
        """
        Provide explanations for detected anomalies.
        
        Args:
            likelihood_scores (pd.DataFrame): Original likelihood scores
            anomaly_indices (np.ndarray): Indices of anomalies
            top_k (int): Number of top contributing groups to report
            
        Returns:
            Dict: Explanations for anomalies
        """
        explanations = {}
        
        for idx in anomaly_indices[:10]:  # Limit to first 10 for performance
            sample_scores = likelihood_scores.iloc[idx]
            
            # Find groups with lowest likelihoods (highest contribution to anomaly)
            sorted_groups = sample_scores.sort_values()
            top_contributing_groups = sorted_groups.head(top_k)
            
            explanations[idx] = {
                'overall_score': sample_scores.mean(),
                'contributing_groups': top_contributing_groups.to_dict(),
                'group_ranks': {col: rank for rank, col in enumerate(sorted_groups.index)}
            }
        
        return explanations
    
    def _print_detection_stats(self, anomaly_scores: np.ndarray, threshold: float, 
                              anomaly_indices: np.ndarray, likelihood_scores: pd.DataFrame) -> None:
        """
        Print detailed statistics about anomaly detection.
        
        Args:
            anomaly_scores (np.ndarray): Computed anomaly scores
            threshold (float): Applied threshold
            anomaly_indices (np.ndarray): Detected anomaly indices
            likelihood_scores (pd.DataFrame): Original likelihood scores
        """
        print(f"     Likelihood scores range: [{likelihood_scores.values.min():.4f}, {likelihood_scores.values.max():.4f}]")
        print(f"     Likelihood scores mean: {likelihood_scores.values.mean():.4f}")
        print(f"     Anomaly scores range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
        print(f"     Anomaly scores mean: {anomaly_scores.mean():.4f}")
        print(f"     Threshold: {threshold:.4f}")
        print(f"     Anomalies detected: {len(anomaly_indices)} ({len(anomaly_indices)/len(anomaly_scores)*100:.1f}%)")
        
        if len(anomaly_indices) > 0:
            top_scores = anomaly_scores[anomaly_indices]
            top_scores_sorted = np.sort(top_scores)[::-1]  # Descending order
            print(f"     Top anomaly scores: {top_scores_sorted[:5]}")
        else:
            # Investigate why no anomalies were found
            print(f"     🔍 DEBUG: No anomalies found")
            print(f"     Max anomaly score: {anomaly_scores.max():.4f}")
            print(f"     95th percentile: {np.percentile(anomaly_scores, 95):.4f}")
            print(f"     90th percentile: {np.percentile(anomaly_scores, 90):.4f}")
            print(f"     Scores above threshold: {np.sum(anomaly_scores > threshold)}")
            
            # Try a lower threshold as fallback
            lower_threshold = np.percentile(anomaly_scores, 97)  # Top 3%
            anomalies_with_lower = np.sum(anomaly_scores > lower_threshold)
            print(f"     Would detect {anomalies_with_lower} anomalies with 97th percentile threshold ({lower_threshold:.4f})")
