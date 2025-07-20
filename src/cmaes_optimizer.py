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
    CMA-ES optimizer for Bayesian Network weight optimization in anomaly detection.
    
    This optimizer uses CMA-ES to find optimal BN weights and threshold percentiles
    with the same focused fitness function as the genetic algorithm, providing an
    alternative optimization approach with different exploration characteristics.
    """

    def __init__(self, config: Dict):
        """
        Initialize CMA-ES optimizer for BN weight optimization.
        
        Args:
            config (Dict): Configuration parameters for CMA-ES
        """
        self.config = config
        self.es = None  # CMA-ES evolution strategy object
        self.best_solution = None
        self.best_fitness = -np.inf
        self.convergence_data = {}
        self.base_results_dir = "results"
        self.execution_results_dir = None
        
        # Data placeholders
        self.likelihood_scores = None
        self.processed_data = None
        self.n_bn_groups = None
        
        os.makedirs(self.base_results_dir, exist_ok=True)
        
        # Load and validate fitness component weights
        self.fitness_components = config.get('fitness_components', {
            'separation_quality': 0.45,    # Cohen's d between anomaly/normal
            'detection_rate': 0.25,        # Target anomaly rate optimization
            'threshold_robustness': 0.20,  # Stability across thresholds
            'weight_diversity': 0.10       # Encourage diverse BN usage
        })
        
        # Validate and normalize weights
        total_weight = sum(self.fitness_components.values())
        if abs(total_weight - 1.0) > 1e-6:
            print(f"⚠️  CMA-ES fitness weights sum to {total_weight:.3f}, normalizing...")
            self.fitness_components = {k: v/total_weight for k, v in self.fitness_components.items()}
        
        print(f"🎯 CMA-ES Optimizer for BN Weight Optimization")
        print(f"   Fitness components: {self.fitness_components}")
    
    def set_results_dir(self, results_dir: str):
        """Set the execution results directory for saving results."""
        self.execution_results_dir = results_dir
    
    def get_results_dir(self) -> str:
        """Get the current results directory path."""
        return self.execution_results_dir if self.execution_results_dir else self.base_results_dir
    
    @property
    def results_dir(self) -> str:
        """Property to access current results directory."""
        return self.get_results_dir()

    def _setup_parameter_space(self):
        """Setup parameter space for BN weights + threshold percentile optimization."""
        # Parameter space: n_bn_groups weights + 1 threshold percentile
        self.dimension = self.n_bn_groups + 1
        
        # Bounds: weights [0.01, 1.0], threshold [1.0, 10.0]
        self.lower_bounds = [0.01] * self.n_bn_groups + [1.0]
        self.upper_bounds = [1.0] * self.n_bn_groups + [10.0]
        
        print(f"📊 CMA-ES parameter space: {self.n_bn_groups} BN weights + 1 threshold = {self.dimension} parameters")
    def _create_initial_solution(self) -> List[float]:
        """Create initial solution with equal BN weights and moderate threshold."""
        # Start with equal weights and middle threshold
        initial_weights = [1.0 / self.n_bn_groups] * self.n_bn_groups
        initial_threshold = 5.0  # Middle of 1-10 range
        
        return initial_weights + [initial_threshold]

    def _decode_individual(self, solution: List[float]) -> Dict:
        """
        Convert CMA-ES solution into interpretable parameters.
        
        Args:
            solution (List[float]): CMA-ES solution vector
            
        Returns:
            Dict: Decoded parameters with normalized weights
        """
        # Enforce bounds
        bounded_solution = self._enforce_bounds(solution.copy())
        
        # Extract BN weights and normalize them to sum to 1
        raw_weights = np.array(bounded_solution[:self.n_bn_groups])
        bn_weights = raw_weights / np.sum(raw_weights)  # Normalize to sum to 1
        
        # Extract threshold percentile
        threshold_percentile = bounded_solution[self.n_bn_groups]
        
        return {
            'bn_weights': bn_weights,
            'threshold_percentile': threshold_percentile,
            'n_bn_groups': self.n_bn_groups,
            'aggregation_method': 'weighted'  # CMA-ES uses weighted aggregation
        }

    def _enforce_bounds(self, solution: List[float]) -> List[float]:
        """Enforce parameter bounds on solution vector."""
        bounded = []
        for i in range(len(solution)):
            min_val = self.lower_bounds[i]
            max_val = self.upper_bounds[i]
            bounded.append(max(min_val, min(max_val, solution[i])))
        return bounded

    def _fitness_function(self, solution: List[float]) -> float:
        """
        CMA-ES fitness function (minimization) - returns negative fitness.
        
        Args:
            solution (List[float]): CMA-ES parameter vector
            
        Returns:
            float: Negative fitness value (CMA-ES minimizes)
        """
        try:
            params = self._decode_individual(solution)
            
            # Compute weighted anomaly scores using optimized BN weights
            anomaly_scores = self._compute_weighted_anomaly_scores(
                self.likelihood_scores, params['bn_weights']
            )
            
            # Determine threshold and identify anomalies
            threshold = self._determine_threshold(anomaly_scores, params['threshold_percentile'])
            anomaly_indices = np.where(anomaly_scores > threshold)[0]
            
            # Calculate fitness components
            fitness = self._calculate_fitness_score(
                anomaly_scores, anomaly_indices, params
            )
            
            # Update best solution tracking
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution.copy()
            
            return -fitness  # CMA-ES minimizes, so return negative fitness
            
        except Exception as e:
            # Return high penalty for invalid solutions
            return 1000.0

    def _compute_weighted_anomaly_scores(self, likelihood_scores: pd.DataFrame, 
                                        bn_weights: np.ndarray) -> np.ndarray:
        """
        Compute weighted anomaly scores using optimized BN weights.
        
        Args:
            likelihood_scores (pd.DataFrame): Likelihood matrix (samples x BN_groups)
            bn_weights (np.ndarray): Normalized weights for each BN group
            
        Returns:
            np.ndarray: Weighted anomaly scores (higher = more anomalous)
        """
        # Convert likelihood scores to anomaly scores (negative log-likelihood)
        anomaly_matrix = -likelihood_scores.values
        
        # Apply weighted combination
        weighted_scores = np.dot(anomaly_matrix, bn_weights)
        
        # Standardize scores (mean=0, std=1)
        weighted_scores = (weighted_scores - np.mean(weighted_scores)) / (np.std(weighted_scores) + 1e-8)
        
        return weighted_scores

    def _determine_threshold(self, anomaly_scores: np.ndarray, threshold_percentile: float) -> float:
        """Determine anomaly threshold from percentile."""
        return np.percentile(anomaly_scores, 100 - threshold_percentile)

    def _calculate_fitness_score(self, anomaly_scores: np.ndarray, 
                                anomaly_indices: np.ndarray, params: Dict) -> float:
        """
        Calculate comprehensive fitness score for BN weight optimization.
        
        This function implements the same focused fitness evaluation as the GA,
        specifically designed for optimizing Bayesian Network weights.
        
        Fitness Components (0-100 scale):
        1. Separation Quality (45%): Cohen's d effect size between anomaly/normal distributions
        2. Detection Rate (25%): Reward for target anomaly rates (2-6%)
        3. Threshold Robustness (20%): Stability across neighboring threshold values
        4. Weight Diversity (10%): Prevent degenerate solutions with few active weights
        
        Args:
            anomaly_scores (np.ndarray): All anomaly scores from weighted BN combination
            anomaly_indices (np.ndarray): Indices of detected anomalies
            params (Dict): Current parameters including BN weights
            
        Returns:
            float: Overall fitness score (0-100)
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
        
        return max(0.0, min(100.0, total_fitness))

    def _evaluate_separation_quality(self, normal_values: np.ndarray, 
                                   anomaly_values: np.ndarray) -> float:
        """
        Evaluate separation quality using Cohen's d effect size.
        
        Cohen's d interpretation:
        - 0.2: Small effect
        - 0.5: Medium effect  
        - 0.8: Large effect
        - 1.2+: Very large effect (ideal for anomaly detection)
        
        Returns:
            float: Separation score (0.0 to 1.0)
        """
        try:
            if len(normal_values) < 2 or len(anomaly_values) < 2:
                return 0.0
                
            # Calculate means
            normal_mean = np.mean(normal_values)
            anomaly_mean = np.mean(anomaly_values)
            
            # Calculate pooled standard deviation
            normal_var = np.var(normal_values, ddof=1) if len(normal_values) > 1 else np.var(normal_values)
            anomaly_var = np.var(anomaly_values, ddof=1) if len(anomaly_values) > 1 else np.var(anomaly_values)
            
            pooled_std = np.sqrt(
                ((len(normal_values) - 1) * normal_var + (len(anomaly_values) - 1) * anomaly_var) /
                (len(normal_values) + len(anomaly_values) - 2)
            )
            
            if pooled_std < 1e-8:
                return 0.0
                
            # Calculate Cohen's d
            cohens_d = abs(anomaly_mean - normal_mean) / pooled_std
            
            # Convert to score (sigmoid-like function targeting d > 1.2)
            # Score approaches 1.0 as Cohen's d approaches 2.0+
            separation_score = min(1.0, cohens_d / 2.0)
            
            # Bonus for very large effect sizes
            if cohens_d > 1.5:
                separation_score = min(1.0, separation_score + 0.1)
                
            return separation_score
            
        except Exception:
            return 0.0

    def _evaluate_detection_rate(self, anomaly_rate: float) -> float:
        """
        Evaluate anomaly detection rate against target ranges.
        
        Target rates for electrical data:
        - Primary target: 3-5% (realistic for real anomalies)
        - Secondary target: 1-2% (conservative detection)
        - Tertiary target: 6-8% (aggressive but acceptable)
        
        Args:
            anomaly_rate (float): Percentage of samples detected as anomalies
            
        Returns:
            float: Detection rate score (0.0 to 1.0)
        """
        # Multi-modal scoring to reward different useful ranges
        primary_score = np.exp(-0.5 * ((anomaly_rate - 4.0) / 1.0) ** 2)      # Target: 4% ± 1%
        secondary_score = 0.8 * np.exp(-0.5 * ((anomaly_rate - 1.5) / 0.5) ** 2)  # Target: 1.5% ± 0.5%  
        tertiary_score = 0.6 * np.exp(-0.5 * ((anomaly_rate - 7.0) / 1.5) ** 2)   # Target: 7% ± 1.5%
        
        return max(primary_score, secondary_score, tertiary_score)

    def _evaluate_threshold_robustness(self, anomaly_scores: np.ndarray, 
                                     target_percentile: float) -> float:
        """
        Evaluate stability of anomaly detection across nearby threshold values.
        
        A robust solution should produce similar anomaly counts when the threshold
        is slightly perturbed, indicating a stable separation.
        
        Args:
            anomaly_scores (np.ndarray): All anomaly scores
            target_percentile (float): Target threshold percentile
            
        Returns:
            float: Robustness score (0.0 to 1.0)
        """
        try:
            # Test neighboring thresholds
            test_percentiles = [
                max(1.0, target_percentile - 0.5),
                target_percentile,
                min(10.0, target_percentile + 0.5)
            ]
            
            anomaly_counts = []
            for perc in test_percentiles:
                threshold = np.percentile(anomaly_scores, 100 - perc)
                count = np.sum(anomaly_scores > threshold)
                anomaly_counts.append(count)
            
            if len(anomaly_counts) < 2:
                return 0.5
                
            # Calculate coefficient of variation (lower = more stable)
            mean_count = np.mean(anomaly_counts)
            std_count = np.std(anomaly_counts)
            
            if mean_count < 1e-8:
                return 0.0
                
            cv = std_count / mean_count
            
            # Convert to robustness score (lower CV = higher robustness)
            robustness_score = np.exp(-cv * 5)  # Exponential decay for CV
            
            return min(1.0, robustness_score)
            
        except Exception:
            return 0.5

    def _evaluate_weight_diversity(self, bn_weights: np.ndarray) -> float:
        """
        Evaluate diversity of BN weights to prevent degenerate solutions.
        
        Encourages solutions that use multiple BNs rather than focusing on just one.
        
        Args:
            bn_weights (np.ndarray): Normalized BN weights
            
        Returns:
            float: Diversity score (0.0 to 1.0)
        """
        try:
            # Calculate entropy of weight distribution
            # Add small epsilon to avoid log(0)
            weights_with_epsilon = bn_weights + 1e-8
            weights_normalized = weights_with_epsilon / np.sum(weights_with_epsilon)
            
            # Shannon entropy
            entropy = -np.sum(weights_normalized * np.log(weights_normalized))
            max_entropy = np.log(len(bn_weights))  # Maximum possible entropy
            
            if max_entropy < 1e-8:
                return 1.0
                
            # Normalize entropy to 0-1 range
            diversity_score = entropy / max_entropy
            
            # Bonus for reasonably balanced weights (not too uniform, not too concentrated)
            effective_weights = np.sum(bn_weights > 0.05)  # Count significant weights
            if 3 <= effective_weights <= len(bn_weights) * 0.7:  # Sweet spot
                diversity_score = min(1.0, diversity_score + 0.1)
                
            return diversity_score
            
        except Exception:
            return 0.5

    def optimize(self, likelihood_scores: pd.DataFrame, data: pd.DataFrame) -> Dict:
        """
        Run CMA-ES optimization to find optimal BN weights and threshold.
        
        Args:
            likelihood_scores (pd.DataFrame): Likelihood matrix (samples x BN_groups)
            data (pd.DataFrame): Original processed data
            
        Returns:
            Dict: Optimized parameters including BN weights and threshold
        """
        print("🎯 Starting CMA-ES BN Weight Optimization...")
        
        # Store data references
        self.likelihood_scores = likelihood_scores
        self.processed_data = data
        self.n_bn_groups = likelihood_scores.shape[1]
        
        # Setup parameter space
        self._setup_parameter_space()
        
        # Reset tracking variables
        self.best_fitness = -np.inf
        self.best_solution = None
        
        # CMA-ES configuration
        initial_solution = self._create_initial_solution()
        initial_sigma = self.config.get('initial_sigma', 0.3)  # Moderate exploration
        population_size = self.config.get('population_size', None)
        n_generations = self.config.get('generations', 100)
        
        # Calculate population size if not specified
        if population_size is None:
            population_size = 4 + int(3 * np.log(self.dimension))
            population_size = max(12, min(50, population_size))
        
        print(f"📊 Optimizing weights for {self.n_bn_groups} Bayesian Networks")
        print(f"📈 Likelihood matrix shape: {likelihood_scores.shape}")
        print(f"🔧 CMA-ES Config: Pop={population_size}, Gen={n_generations}, Sigma={initial_sigma}")
        
        # CMA-ES options
        options = {
            'maxiter': n_generations,
            'popsize': population_size,
            'bounds': [self.lower_bounds, self.upper_bounds],
            'tolfun': 1e-9,
            'tolx': 1e-11,
            'tolstagnation': 25,
            'verbose': -1,
            'AdaptSigma': True,
        }

        try:
            # Initialize CMA-ES
            self.es = cma.CMAEvolutionStrategy(initial_solution, initial_sigma, options)
            
            # Evolution tracking
            iteration = 0
            fitness_history = []
            sigma_history = []
            best_fitness_history = []
            
            print(f"📊 Individual structure: {self.n_bn_groups} BN weights + 1 threshold = {self.dimension} genes")
            
            while not self.es.stop():
                iteration += 1
                
                # Generate and evaluate population
                solutions = self.es.ask()
                fitness_values = [self._fitness_function(sol) for sol in solutions]
                
                # Update CMA-ES
                self.es.tell(solutions, fitness_values)
                
                # Track progress
                current_best_fitness = -min(fitness_values)  # Convert back to positive
                current_mean_fitness = -np.mean(fitness_values)
                current_sigma = self.es.sigma
                
                fitness_history.append(current_mean_fitness)
                sigma_history.append(current_sigma)
                best_fitness_history.append(current_best_fitness)
                
                # Progress reporting
                if iteration % 10 == 0 or iteration == 1:
                    print(f"Gen {iteration}: Best={current_best_fitness:.2f}, Avg={current_mean_fitness:.2f}, Sigma={current_sigma:.4f}")
            
            # Store convergence data
            self.convergence_data = {
                'iterations': list(range(1, iteration + 1)),
                'best_fitness': best_fitness_history,
                'mean_fitness': fitness_history,
                'sigma': sigma_history,
                'final_sigma': self.es.sigma,
                'total_iterations': iteration,
                'stop_condition': str(self.es.stop())
            }
            
            # Extract best solution
            if self.best_solution is not None:
                best_params = self._decode_individual(self.best_solution)
                best_fitness = self.best_fitness
                
                print(f"\n✅ CMA-ES Optimization completed!")
                print(f"🏆 Best fitness: {best_fitness:.2f}/100")
                print(f"🎯 Best threshold: {best_params['threshold_percentile']:.2f}%")
                print(f"⚖️  Weight distribution:")
                
                # Show top 5 weights
                weight_indices = np.argsort(best_params['bn_weights'])[::-1]
                for i in range(min(5, len(weight_indices))):
                    idx = weight_indices[i]
                    weight = best_params['bn_weights'][idx]
                    print(f"   BN_{idx:2d}: {weight:.3f} ({weight*100:.1f}%)")
                
                # Save results if results directory is available
                if hasattr(self, 'execution_results_dir') and self.execution_results_dir:
                    self._save_optimization_results(best_params)
                    self._create_optimization_plots()
                
                return best_params
            else:
                raise Exception("No valid solution found during optimization")
                
        except Exception as e:
            print(f"❌ CMA-ES optimization failed: {e}")
            # Return reasonable defaults
            default_solution = self._create_initial_solution()
            return self._decode_individual(default_solution)

    def _save_optimization_results(self, best_params: Dict):
        """Save CMA-ES optimization results to files."""
        try:
            # Save comprehensive summary
            summary_data = {
                'algorithm': 'CMA-ES',
                'best_fitness': self.best_fitness,
                'best_parameters': self._make_json_serializable(best_params),
                'fitness_components': self.fitness_components,
                'total_iterations': len(self.convergence_data.get('iterations', [])),
                'final_sigma': self.convergence_data.get('final_sigma'),
                'stop_condition': self.convergence_data.get('stop_condition', 'Unknown'),
                'convergence_data': self._make_json_serializable(self.convergence_data)
            }
            
            # Save JSON summary
            summary_path = os.path.join(self.results_dir, 'cmaes_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            # Save fitness history CSV
            if self.convergence_data:
                history_df = pd.DataFrame({
                    'iteration': self.convergence_data['iterations'],
                    'best_fitness': self.convergence_data['best_fitness'],
                    'mean_fitness': self.convergence_data['mean_fitness'],
                    'sigma': self.convergence_data['sigma']
                })
                history_path = os.path.join(self.results_dir, 'cmaes_fitness_history.csv')
                history_df.to_csv(history_path, index=False)
                
                print(f"📁 Results saved to {summary_path}")
            
        except Exception as e:
            print(f"⚠️  Error saving CMA-ES results: {e}")

    def _create_optimization_plots(self):
        """Create CMA-ES optimization progress plots."""
        if not self.convergence_data:
            print("⚠️  No convergence data available for plotting.")
            return

        try:
            plt.style.use('default')
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            iterations = self.convergence_data['iterations']
            best_fitness = self.convergence_data['best_fitness']
            mean_fitness = self.convergence_data['mean_fitness']
            sigma = self.convergence_data['sigma']
            
            # Plot 1: Fitness Evolution
            ax1 = axes[0]
            ax1.plot(iterations, best_fitness, 'r-', linewidth=3, label='Best Fitness', marker='o', markersize=3)
            ax1.plot(iterations, mean_fitness, 'b-', linewidth=2, label='Mean Fitness', alpha=0.8)
            ax1.set_title('CMA-ES Fitness Evolution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness Score (0-100)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 105)
            
            # Add final fitness annotation
            final_fitness = best_fitness[-1]
            ax1.annotate(f'Final: {final_fitness:.1f}', 
                        xy=(iterations[-1], final_fitness), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=10, fontweight='bold')
            
            # Plot 2: Sigma Evolution (Step Size)
            ax2 = axes[1]
            ax2.semilogy(iterations, sigma, 'g-', linewidth=2, label='Step Size (σ)', marker='s', markersize=2)
            ax2.set_title('CMA-ES Step Size Evolution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Step Size (σ) [log scale]')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.results_dir, 'cmaes_fitness_evolution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Error creating CMA-ES plots: {e}")
            traceback.print_exc()

    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization results."""
        if self.best_solution is None:
            return {'status': 'Not optimized'}
        
        return {
            'status': 'Optimized',
            'algorithm': 'CMA-ES',
            'best_fitness': self.best_fitness,
            'best_parameters': self._decode_individual(self.best_solution),
            'total_iterations': len(self.convergence_data.get('iterations', [])),
            'final_sigma': self.convergence_data.get('final_sigma'),
            'stop_condition': self.convergence_data.get('stop_condition', 'Unknown')
        }
