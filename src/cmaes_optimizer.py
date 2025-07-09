import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cma
from typing import Dict, List, Tuple
import warnings
import os
import json
import traceback
from scipy import stats
import random

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class CMAESOptimizer:
    """
    Optimizes anomaly detection parameters using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    Uses CMA-ES to find optimal hyperparameters for the AnomalyDetector with the same
    multi-objective fitness function as the genetic algorithm.
    """

    def __init__(self, config: Dict):
        """
        Initializes the CMA-ES optimizer.
        
        Args:
            config (Dict): Configuration for the CMA-ES algorithm.
        """
        self.config = config
        self.es = None  # CMA-ES object
        self.best_solution = None
        self.best_fitness = -np.inf
        self.best_components = {}
        self.convergence_data = {}
        self.base_results_dir = "results"
        self.execution_results_dir = None
        os.makedirs(self.base_results_dir, exist_ok=True)
        self._define_parameters()
    
    def set_results_dir(self, results_dir: str):
        """Set the execution results directory."""
        self.execution_results_dir = results_dir
    
    def get_results_dir(self) -> str:
        """Get the current results directory."""
        if self.execution_results_dir is None:
            return self.base_results_dir
        return self.execution_results_dir
    
    @property
    def results_dir(self) -> str:
        """Property to get the current results directory."""
        return self.get_results_dir()

    def _define_parameters(self):
        """Defines the parameter space for optimization."""
        # Same parameter space as genetic optimizer
        self.param_bounds = {
            'threshold_percentile': (1.0, 10.0),
            'aggregation_method_idx': (0, 4),
            'use_zscore': (0, 1),
            'threshold_method_idx': (0, 3)
        }
        self.aggregation_methods = ['mean', 'min', 'median', 'weighted', 'sum']
        self.threshold_methods = ['percentile', 'std', 'iqr', 'adaptive']
        
        # Parameter bounds for CMA-ES (continuous space)
        self.lower_bounds = [bounds[0] for bounds in self.param_bounds.values()]
        self.upper_bounds = [bounds[1] for bounds in self.param_bounds.values()]
        self.dimension = len(self.param_bounds)

    def _create_initial_solution(self) -> List[float]:
        """Creates an initial solution in the center of the parameter space."""
        initial = []
        for param, (min_val, max_val) in self.param_bounds.items():
            if 'idx' in param or param == 'use_zscore':
                # For discrete parameters, start in the middle
                initial.append((min_val + max_val) / 2.0)
            else:
                # For continuous parameters, start in the middle
                initial.append((min_val + max_val) / 2.0)
        return initial

    def _decode_solution(self, solution: List[float]) -> Dict:
        """Converts a CMA-ES solution into human-readable parameters."""
        # Enforce bounds first
        bounded_solution = self._enforce_bounds(solution.copy())
        
        return {
            'threshold_percentile': bounded_solution[0],
            'aggregation_method': self.aggregation_methods[int(round(bounded_solution[1]))],
            'use_zscore_transformation': bool(round(bounded_solution[2])),
            'threshold_method': self.threshold_methods[int(round(bounded_solution[3]))]
        }

    def _enforce_bounds(self, solution: List[float]) -> List[float]:
        """Ensures that all parameters are within their defined bounds."""
        bounded = []
        for i, (param, (min_val, max_val)) in enumerate(self.param_bounds.items()):
            val = max(min_val, min(max_val, solution[i]))
            if 'idx' in param or param == 'use_zscore':
                val = round(val)  # Round discrete parameters
            bounded.append(val)
        return bounded

    def _fitness_function(self, solution: List[float]) -> float:
        """
        CMA-ES fitness function (to be minimized, so we return negative fitness).
        
        Args:
            solution: Parameter vector from CMA-ES
            
        Returns:
            Negative fitness value (CMA-ES minimizes)
        """
        try:
            params = self._decode_solution(solution)
            from .anomaly_detector import AnomalyDetector  # Local import to avoid circular dependency
            
            temp_detector = AnomalyDetector(params)
            anomaly_scores, anomaly_indices = temp_detector.detect_anomalies(
                self.likelihood_scores, verbose=False
            )
            
            fitness, components = self._calculate_fitness(anomaly_scores, anomaly_indices, params)
            
            # Update best solution if this is better
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution.copy()
                self.best_components = components.copy()
            
            return -fitness  # CMA-ES minimizes, so return negative
            
        except Exception:
            return 1000.0  # Return high value (bad fitness) for invalid solutions

    def optimize(self, likelihood_scores: pd.DataFrame, data: pd.DataFrame) -> Dict:
        """
        Runs the  CMA-ES optimization process with improved convergence strategies.
        
        Args:
            likelihood_scores: Likelihood scores from the Bayesian Network.
            data: Original data for validation.
            
        Returns:
            A dictionary containing the best found parameters.
        """
        print("     Starting  CMA-ES optimization...")
        self.likelihood_scores = likelihood_scores
        self.original_data = data
        
        # Reset best tracking
        self.best_fitness = -np.inf
        self.best_solution = None
        self.best_components = {}
        
        #  CMA-ES configuration
        initial_solution = self._create_initial_solution()
        initial_sigma = self.config.get('initial_sigma', 0.8)  # Increased for better exploration
        population_size = self.config.get('population_size', None)
        
        # Calculate adaptive population size if not specified
        if population_size is None:
            population_size = 4 + int(3 * np.log(self.dimension))  # CMA-ES default formula
            population_size = max(8, min(50, population_size))  # Reasonable bounds
        
        #  CMA-ES options with improved convergence control
        options = {
            'maxiter': self.config.get('generations', 150),  # Increased iterations
            'popsize': population_size,
            'bounds': [self.lower_bounds, self.upper_bounds],
            'tolfun': 1e-8,  # Tighter function tolerance  
            'tolx': 1e-10,   # Tighter parameter tolerance
            'tolstagnation': 30,  # Allow more stagnation before stopping
            'verbose': -1,
            'AdaptSigma': True,  # Enable sigma adaptation
            'CMA_stds': initial_solution,  # Initialize step sizes
        }

        try:
            # Initialize CMA-ES with  strategy
            self.es = cma.CMAEvolutionStrategy(initial_solution, initial_sigma, options)
            
            # Multi-phase evolution with restart capability
            iteration = 0
            fitness_history = []
            sigma_history = []
            best_fitness_history = []
            restart_count = 0
            max_restarts = 2
            
            # Stagnation detection parameters
            stagnation_window = 20
            stagnation_threshold = 0.01
            
            print(f"     Initial solution: {self._decode_solution(initial_solution)}")
            print(f"     Population size: {population_size}, Initial sigma: {initial_sigma:.3f}")
            
            while not self.es.stop() and restart_count <= max_restarts:
                iteration += 1
                
                # Generate new population
                solutions = self.es.ask()
                
                # Evaluate fitness for all solutions with  diversity
                fitness_values = []
                for i, sol in enumerate(solutions):
                    base_fitness = self._fitness_function(sol)
                    
                    # Add diversity pressure based on solution uniqueness
                    diversity_bonus = self._calculate_diversity_bonus(sol, solutions, i)
                    adjusted_fitness = base_fitness - diversity_bonus  # CMA-ES minimizes
                    fitness_values.append(adjusted_fitness)
                
                # Tell CMA-ES the fitness values
                self.es.tell(solutions, fitness_values)
                
                # Track convergence data
                current_best_fitness = -min(fitness_values)  # Convert back to positive
                current_sigma = self.es.sigma
                current_mean_fitness = -np.mean(fitness_values)
                
                fitness_history.append(current_mean_fitness)
                sigma_history.append(current_sigma)
                best_fitness_history.append(current_best_fitness)
                
                #  stagnation detection and restart mechanism
                if iteration >= stagnation_window:
                    recent_improvement = (best_fitness_history[-1] - 
                                        best_fitness_history[-stagnation_window])
                    
                    if (recent_improvement < stagnation_threshold and 
                        current_sigma < 0.01 and restart_count < max_restarts):
                        
                        print(f"     Detected stagnation at iteration {iteration}. Restarting CMA-ES...")
                        restart_count += 1
                        
                        # Restart with increased sigma and perturbed initial solution  
                        new_initial = self._create_perturbed_solution()
                        new_sigma = initial_sigma * (1.5 ** restart_count)  # Increase exploration
                        
                        options['maxiter'] = self.config.get('generations', 150) - iteration
                        self.es = cma.CMAEvolutionStrategy(new_initial, new_sigma, options)
                        
                        print(f"     Restart #{restart_count}: New sigma = {new_sigma:.3f}")
                        continue
                
                # Progress reporting with  information
                if iteration % 15 == 0 or iteration == 1:
                    improvement_rate = (best_fitness_history[-1] - best_fitness_history[max(0, iteration-15)]) if iteration > 15 else 0
                    print(f"     Iter {iteration}: Best={current_best_fitness:.3f}, "
                          f"Mean={current_mean_fitness:.3f}, Sigma={current_sigma:.4f}, "
                          f"Improvement={improvement_rate:.3f}")
            
            # Store  convergence data
            self.convergence_data = {
                'iterations': list(range(1, iteration + 1)),
                'best_fitness': best_fitness_history,
                'mean_fitness': fitness_history,
                'sigma': sigma_history,
                'final_sigma': self.es.sigma,
                'total_iterations': iteration,
                'restart_count': restart_count,
                'stop_condition': str(self.es.stop())
            }
            
            # Get best parameters
            if self.best_solution is not None:
                best_params = self._decode_solution(self.best_solution)
                self._save_optimization_results(best_params)
                self._create_fitness_plots()
                
                print(f"     ✅ CMA-ES completed. Best fitness: {self.best_fitness:.4f}")
                print(f"     Total iterations: {iteration}, Restarts: {restart_count}")
                print(f"     Final sigma: {self.es.sigma:.6f}")
                print(f"     Results saved to '{self.results_dir}/'")
                
                return best_params
            else:
                raise Exception("No valid solution found during optimization")
                
        except Exception as e:
            print(f"     ❌ CMA-ES optimization failed: {e}")
            # Return a reasonable default
            default_solution = self._create_initial_solution()
            return self._decode_solution(default_solution)
    
    def _create_perturbed_solution(self) -> List[float]:
        """Creates a perturbed initial solution for restart."""
        base_solution = self._create_initial_solution()
        perturbed = []
        
        for i, (param, (min_val, max_val)) in enumerate(self.param_bounds.items()):
            # Add controlled perturbation
            noise = random.uniform(-0.3, 0.3) * (max_val - min_val)
            perturbed_val = base_solution[i] + noise
            perturbed_val = max(min_val, min(max_val, perturbed_val))
            perturbed.append(perturbed_val)
        
        return perturbed
    
    def _calculate_diversity_bonus(self, solution: List[float], 
                                   all_solutions: List[List[float]], 
                                   current_idx: int) -> float:
        """Calculate diversity bonus to encourage exploration."""
        try:
            min_distance = float('inf')
            for i, other_sol in enumerate(all_solutions):
                if i != current_idx:
                    distance = np.linalg.norm(np.array(solution) - np.array(other_sol))
                    min_distance = min(min_distance, distance)
            
            # Convert distance to bonus (closer solutions get less bonus)
            if min_distance < 0.1:
                return 0.5  # Penalty for very similar solutions
            elif min_distance > 1.0:
                return -0.2  # Bonus for diverse solutions
            else:
                return 0.0
        except:
            return 0.0

    def _calculate_fitness(self, anomaly_scores: np.ndarray, 
                           anomaly_indices: np.ndarray, params: Dict) -> Tuple[float, Dict]:
        """
         multi-objective fitness function for CMA-ES optimization designed for 
        electrical anomaly detection in time-series wallbox data.
        
        The fitness score (0-100) integrates seven sophisticated components:
        1. Anomaly Rate Quality (20 pts): Multi-modal target distribution for rate optimization
        2. Score Separation Quality (25 pts):  Cohen's d with non-linear scaling  
        3. Distribution Quality (15 pts): Multi-metric statistical coherence assessment
        4. Stability & Robustness (15 pts): Temporal consistency and noise resistance
        5. Domain-Specific Quality (10 pts): Electrical measurement pattern analysis
        6. Exploration Incentive (10 pts): Rewards parameter space exploration
        7. Convergence Control (5 pts): Dynamic convergence prevention mechanism
        """
        components = {
            'rate_score': 0.0, 'separation_score': 0.0, 'distribution_score': 0.0,
            'stability_score': 0.0, 'domain_score': 0.0, 'exploration_score': 0.0, 
            'convergence_score': 0.0
        }
        
        num_samples = len(anomaly_scores)
        num_anomalies = len(anomaly_indices)

        # Early exit for degenerate cases
        if num_anomalies == 0 or num_anomalies >= num_samples * 0.4:
            return 0.0, components

        anomaly_rate = (num_anomalies / num_samples) * 100
        normal_indices = np.setdiff1d(np.arange(num_samples), anomaly_indices)
        anomaly_values = anomaly_scores[anomaly_indices]
        normal_values = anomaly_scores[normal_indices]

        # Component 1:  Anomaly Rate Quality (20 points)
        # Multi-modal target: prefer 3-5% (primary) or 1-2% (secondary) or 6-8% (tertiary)
        primary_target, primary_spread = 4.0, 1.2
        secondary_target, secondary_spread = 1.5, 0.8  
        tertiary_target, tertiary_spread = 7.0, 1.5
        
        primary_score = 20 * np.exp(-0.5 * ((anomaly_rate - primary_target) / primary_spread) ** 2)
        secondary_score = 15 * np.exp(-0.5 * ((anomaly_rate - secondary_target) / secondary_spread) ** 2)
        tertiary_score = 12 * np.exp(-0.5 * ((anomaly_rate - tertiary_target) / tertiary_spread) ** 2)
        
        components['rate_score'] = max(primary_score, secondary_score, tertiary_score)

        # Component 2: Advanced Score Separation Quality (25 points)
        if len(normal_values) > 2 and len(anomaly_values) > 1:
            mean_anomaly, std_anomaly = np.mean(anomaly_values), np.std(anomaly_values)
            mean_normal, std_normal = np.mean(normal_values), np.std(normal_values)
            
            #  Cohen's d with robust pooled standard deviation
            pooled_std = np.sqrt(((num_anomalies - 1) * std_anomaly**2 + (len(normal_values) - 1) * std_normal**2) / (num_samples - 2))
            
            if pooled_std > 1e-6:
                cohens_d = abs(mean_anomaly - mean_normal) / pooled_std
                
                # Non-linear scaling: steep reward for d > 0.8 (large effect size)
                if cohens_d >= 0.8:
                    base_score = 20
                    bonus = 5 * (1 - np.exp(-(cohens_d - 0.8) * 2))  # Exponential bonus
                    components['separation_score'] = min(25, base_score + bonus)
                else:
                    # Sigmoid scaling for moderate effect sizes
                    k, offset = 6.0, 0.5
                    components['separation_score'] = 25 / (1 + np.exp(-k * (cohens_d - offset)))
                
                # Additional separation metrics
                overlap_coefficient = self._calculate_overlap_coefficient(normal_values, anomaly_values)
                separation_bonus = (1 - overlap_coefficient) * 3
                components['separation_score'] = min(25, components['separation_score'] + separation_bonus)

        # Component 3: Multi-Metric Distribution Quality (15 points)
        if len(normal_values) > 5:
            dist_scores = []
            
            # Shapiro-Wilk normality test
            try:
                _, shapiro_p = stats.shapiro(normal_values[:min(len(normal_values), 5000)])
                dist_scores.append(min(shapiro_p * 8, 4))  # Max 4 points
            except:
                dist_scores.append(0)
            
            # Kurtosis assessment (prefer normal kurtosis ~3)
            try:
                kurt = stats.kurtosis(normal_values, fisher=True)  # Fisher kurtosis (normal=0)
                kurtosis_score = 3 * np.exp(-0.5 * (kurt / 2) ** 2)  # Max 3 points
                dist_scores.append(kurtosis_score)
            except:
                dist_scores.append(0)
            
            # Skewness assessment (prefer symmetric ~0)  
            try:
                skew = stats.skew(normal_values)
                skewness_score = 3 * np.exp(-0.5 * (skew / 1.5) ** 2)  # Max 3 points
                dist_scores.append(skewness_score)
            except:
                dist_scores.append(0)
                
            # Anderson-Darling test for normality
            try:
                ad_stat, critical_vals, significance_levels = stats.anderson(normal_values, dist='norm')
                # Convert to p-value approximation
                ad_p_approx = 1.0 / (1.0 + ad_stat)
                dist_scores.append(min(ad_p_approx * 5, 5))  # Max 5 points
            except:
                dist_scores.append(0)
            
            components['distribution_score'] = sum(dist_scores)

        # Component 4: Stability & Robustness Analysis (15 points)
        stability_scores = []
        
        # Coefficient of variation for anomaly scores
        cv_normal = std_normal / abs(mean_normal) if abs(mean_normal) > 1e-6 else 10
        cv_stability = 5 * np.exp(-cv_normal * 3)  # Prefer low CV (stable detection)
        stability_scores.append(cv_stability)
        
        # Outlier sensitivity test
        try:
            # Remove extreme values and recalculate 
            q1, q3 = np.percentile(normal_values, [25, 75])
            iqr = q3 - q1
            filtered_normal = normal_values[(normal_values >= q1 - 1.5*iqr) & (normal_values <= q3 + 1.5*iqr)]
            
            if len(filtered_normal) > len(normal_values) * 0.7:  # Most values retained
                mean_stability = abs(np.mean(filtered_normal) - mean_normal)
                stability_score = 5 * np.exp(-mean_stability * 20)  # Prefer stable means
                stability_scores.append(stability_score)
            else:
                stability_scores.append(2)  # Moderate score for high outlier sensitivity
        except:
            stability_scores.append(0)
        
        # Parameter robustness: penalize extreme parameter combinations
        param_penalty = 0
        if params['threshold_percentile'] < 1.5 or params['threshold_percentile'] > 9.5:
            param_penalty += 2
        
        robustness_score = max(0, 5 - param_penalty)
        stability_scores.append(robustness_score)
        
        components['stability_score'] = sum(stability_scores)

        # Component 5: Domain-Specific Electrical Pattern Analysis (10 points)
        domain_scores = []
        
        # Time-series analysis (if data has temporal structure)
        if hasattr(self, 'original_data') and self.original_data is not None:
            try:
                # Analyze electrical measurement patterns
                domain_scores.append(self._analyze_electrical_patterns(anomaly_indices))
            except:
                domain_scores.append(2)  # Default moderate score
        else:
            domain_scores.append(2)
            
        # Anomaly clustering quality
        if len(anomaly_values) >= 3:
            cluster_score = self._evaluate_anomaly_clustering(anomaly_values)
            domain_scores.append(cluster_score)
        else:
            domain_scores.append(1)
            
        components['domain_score'] = sum(domain_scores)

        # Component 6: Exploration Incentive (10 points) 
        # Reward parameter diversity to prevent premature convergence
        exploration_score = 0
        
        # Threshold percentile diversity reward
        if 2.5 <= params['threshold_percentile'] <= 7.5:
            exploration_score += 3
        elif params['threshold_percentile'] < 2 or params['threshold_percentile'] > 8:
            exploration_score += 5  # Reward exploration of extreme values
        
        # Method diversity bonus
        if params['aggregation_method'] in ['weighted', 'median']:
            exploration_score += 2
        elif params['aggregation_method'] in ['min', 'max']:
            exploration_score += 3  # Higher reward for less common methods
            
        # Transformation combination bonus
        if params['use_zscore_transformation'] and params['threshold_method'] != 'percentile':
            exploration_score += 2
            
        components['exploration_score'] = min(10, exploration_score)

        # Component 7: Convergence Control (5 points)
        # Dynamic mechanism to prevent stagnation
        convergence_score = 0
        
        # Add small controlled randomness based on fitness components
        base_randomness = random.uniform(-0.1, 0.1)
        
        # Increase randomness when fitness is very high (near convergence)
        current_total = sum([components[k] for k in components.keys() if k != 'convergence_score'])
        if current_total > 85:  # High fitness -> increase exploration
            convergence_score = random.uniform(-2, 3)
        elif current_total > 75:
            convergence_score = random.uniform(-1, 2)
        else:
            convergence_score = random.uniform(-0.5, 1)
            
        components['convergence_score'] = convergence_score + base_randomness

        # Final fitness calculation with non-linear combination
        total_fitness = sum(components.values())
        
        # Apply final transformations
        total_fitness = max(0, min(100, total_fitness))
        
        # Add small perturbation to break ties and maintain diversity
        total_fitness += random.uniform(-0.05, 0.05)
        
        return total_fitness, components
    
    def _calculate_overlap_coefficient(self, normal_values: np.ndarray, anomaly_values: np.ndarray) -> float:
        """Calculate overlap coefficient between normal and anomaly score distributions."""
        try:
            # Use histogram-based approach for overlap calculation
            all_values = np.concatenate([normal_values, anomaly_values])
            min_val, max_val = np.min(all_values), np.max(all_values)
            
            if max_val - min_val < 1e-6:
                return 1.0  # Complete overlap
            
            bins = np.linspace(min_val, max_val, 50)
            normal_hist, _ = np.histogram(normal_values, bins=bins, density=True)
            anomaly_hist, _ = np.histogram(anomaly_values, bins=bins, density=True)
            
            # Calculate overlap as minimum of densities
            overlap = np.sum(np.minimum(normal_hist, anomaly_hist)) / len(bins)
            return min(1.0, overlap)
        except:
            return 0.5  # Default moderate overlap
    
    def _analyze_electrical_patterns(self, anomaly_indices: np.ndarray) -> float:
        """Analyze domain-specific electrical measurement patterns."""
        try:
            if not hasattr(self, 'original_data') or self.original_data is None:
                return 2.0
            
            score = 0
            data = self.original_data
            
            # Check for electrical measurement coherence in anomalies
            voltage_cols = [col for col in data.columns if 'V_' in col and not 'phase' in col]
            current_cols = [col for col in data.columns if 'I_' in col and not 'phase' in col]
            power_cols = [col for col in data.columns if 'P' in col and not 'phase' in col]
            
            if len(voltage_cols) > 0 and len(current_cols) > 0:
                # Analyze V-I relationship in anomalies
                anomaly_data = data.iloc[anomaly_indices]
                normal_data = data.drop(data.index[anomaly_indices])
                
                # Check if anomalies show unusual V-I patterns
                if len(voltage_cols) > 0 and len(current_cols) > 0:
                    v_mean_anom = anomaly_data[voltage_cols[0]].mean()
                    i_mean_anom = anomaly_data[current_cols[0]].mean()
                    v_mean_norm = normal_data[voltage_cols[0]].mean()
                    i_mean_norm = normal_data[current_cols[0]].mean()
                    
                    # Reward if anomalies show distinct electrical patterns
                    v_diff = abs(v_mean_anom - v_mean_norm) / (abs(v_mean_norm) + 1e-6)
                    i_diff = abs(i_mean_anom - i_mean_norm) / (abs(i_mean_norm) + 1e-6)
                    
                    pattern_score = min(3, (v_diff + i_diff) * 3)
                    score += pattern_score
            
            # Check for harmonic distortion patterns
            harmonic_cols = [col for col in data.columns if 'THD' in col or '_H' in col]
            if len(harmonic_cols) > 0 and len(anomaly_indices) > 0:
                try:
                    anomaly_harmonic = data.iloc[anomaly_indices][harmonic_cols[0]].mean()
                    normal_harmonic = data.drop(data.index[anomaly_indices])[harmonic_cols[0]].mean()
                    harmonic_diff = abs(anomaly_harmonic - normal_harmonic) / (abs(normal_harmonic) + 1e-6)
                    score += min(2, harmonic_diff * 2)
                except:
                    score += 1
            
            return min(5, score)
        except:
            return 2.0  # Default score on error
    
    def _evaluate_anomaly_clustering(self, anomaly_values: np.ndarray) -> float:
        """Evaluate the clustering quality of anomaly scores."""
        try:
            if len(anomaly_values) < 3:
                return 1.0
            
            # Prefer anomalies that are somewhat clustered (not too spread out)
            anomaly_std = np.std(anomaly_values)
            anomaly_range = np.max(anomaly_values) - np.min(anomaly_values)
            
            # Ideal clustering: moderate std relative to range
            if anomaly_range > 1e-6:
                clustering_ratio = anomaly_std / anomaly_range
                # Prefer clustering_ratio around 0.2-0.4 (moderate clustering)
                optimal_ratio = 0.3
                clustering_score = 5 * np.exp(-10 * (clustering_ratio - optimal_ratio) ** 2)
                return min(5, clustering_score)
            else:
                return 2  # All anomalies have same score (perfect clustering)
        except:
            return 1.0

    def _find_convergence_iteration(self) -> int:
        """Finds the iteration where the best fitness score plateaus."""
        if len(self.convergence_data['best_fitness']) < 10:
            return len(self.convergence_data['best_fitness'])
        
        best_fitness = self.convergence_data['best_fitness']
        window = 10
        for i in range(window, len(best_fitness)):
            recent_window = best_fitness[i-window:i]
            if (np.max(recent_window) - np.min(recent_window)) < 0.1:
                return i - window
        return len(best_fitness)

    def _assess_convergence_quality(self) -> str:
        """Provides a qualitative assessment of the convergence."""
        if not self.convergence_data: 
            return "Unknown"
        
        best_fitness = self.convergence_data['best_fitness']
        if len(best_fitness) < 20: 
            return "Insufficient Data"

        total_improvement = best_fitness[-1] - best_fitness[0]
        last_quarter_idx = int(len(best_fitness) * 0.75)
        late_improvement = best_fitness[-1] - best_fitness[last_quarter_idx]

        if total_improvement <= 0.1: 
            return "Stalled"
        if late_improvement / total_improvement < 0.05: 
            return "Excellent"
        if late_improvement / total_improvement < 0.15: 
            return "Good"
        return "Fair"

    def _create_fitness_plots(self):
        """Creates and displays CMA-ES optimization plots."""
        if not self.convergence_data:
            print("     No convergence data available for plotting.")
            return

        try:
            plt.style.use('default')
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            iterations = self.convergence_data['iterations']
            best_f = self.convergence_data['best_fitness']
            mean_f = self.convergence_data['mean_fitness']
            sigma = self.convergence_data['sigma']
            
            # Plot 1: Fitness Evolution
            ax1 = axes[0]
            ax1.plot(iterations, best_f, 'r-', linewidth=3, label='Best Fitness', marker='o', markersize=4)
            ax1.plot(iterations, mean_f, 'b-', linewidth=2, label='Mean Fitness', alpha=0.8)
            
            conv_iter = self._find_convergence_iteration()
            ax1.axvline(x=conv_iter, color='red', linestyle='--', linewidth=2, 
                       label=f'Convergence (Iter {conv_iter})')
            
            ax1.set_title('CMA-ES Fitness Evolution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Fitness Value (0-100)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 105)

            # Plot 2:  Fitness Breakdown
            ax2 = axes[1]
            if self.best_components:
                names = ['Rate', 'Separation', 'Distribution', 'Stability', 'Domain', 'Exploration', 'Convergence']
                values = [self.best_components.get(k, 0) for k in 
                         ['rate_score', 'separation_score', 'distribution_score', 
                          'stability_score', 'domain_score', 'exploration_score', 'convergence_score']]
                colors = ['#440154', '#31688e', '#35b779', '#fde725', '#ff6b6b', '#4ecdc4', '#95a5a6'][:len(names)]
                bars = ax2.bar(names, values, color=colors, alpha=0.8)
                ax2.bar_label(bars, fmt='%.1f')
                ax2.set_title(' Fitness Breakdown (7 Components)', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Fitness Components')
                ax2.set_ylabel('Component Score')
                ax2.grid(True, axis='y', alpha=0.3)
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            else:
                ax2.text(0.5, 0.5, 'No fitness breakdown available', ha='center', va='center', 
                        transform=ax2.transAxes)
                ax2.set_title(' Fitness Breakdown', fontsize=14, fontweight='bold')

            # Plot 3: CMA-ES Sigma Evolution
            ax3 = axes[2]
            ax3.semilogy(iterations, sigma, 'g-', linewidth=2, label='Step Size (σ)', marker='s', markersize=3)
            ax3.set_title('CMA-ES Step Size Evolution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Step Size (σ) [log scale]')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.results_dir, 'cmaes_fitness_evolution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"     CMA-ES plots saved to {plot_path}")
            
            # Display the plot
            plt.show()

        except Exception as e:
            print(f"     Error creating fitness plots: {e}")
            traceback.print_exc()

    def get_optimization_summary(self) -> Dict:
        """Returns a dictionary summarizing the optimization results."""
        if self.best_solution is None:
            return {'status': 'Not optimized'}
        
        return {
            'status': 'Optimized',
            'algorithm': 'CMA-ES',
            'best_fitness': self.best_fitness,
            'best_parameters': self._decode_solution(self.best_solution),
            'iterations_run': len(self.convergence_data.get('iterations', [])),
            'convergence_iteration': self._find_convergence_iteration(),
            'convergence_quality': self._assess_convergence_quality(),
            'final_sigma': self.convergence_data.get('final_sigma', None),
            'stop_condition': self.convergence_data.get('stop_condition', 'Unknown')
        }

    def _save_optimization_results(self, best_params: Dict):
        """Saves optimization artifacts like parameters and history to files."""
        try:
            # Save comprehensive JSON summary
            summary_data = self.get_optimization_summary()
            summary_data['convergence_data'] = self._make_json_serializable(self.convergence_data)
            summary_path = os.path.join(self.results_dir, 'cmaes_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)

            # Save fitness history
            history_path = os.path.join(self.results_dir, 'cmaes_fitness_history.csv')
            pd.DataFrame(self.convergence_data).to_csv(history_path, index=False)
            
            print(f"     CMA-ES results saved to: {summary_path}")
            print(f"     CMA-ES fitness history saved to: {history_path}")
            
        except Exception as e:
            print(f"     Error saving CMA-ES results: {e}")

    def _make_json_serializable(self, obj):
        """Recursively converts numpy types to native Python types for JSON."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_json_serializable(i) for i in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj
