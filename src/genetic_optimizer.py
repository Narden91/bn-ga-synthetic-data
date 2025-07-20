import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deap import creator, base, tools, algorithms
from typing import Dict, List, Tuple, Optional
import random
import warnings
import os
import json
import traceback
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class GeneticOptimizer:
    """
    Genetic Algorithm optimizer for Bayesian Network weight optimization.
    
    Optimizes weights for 17 BN outputs to find the best weighted combination
    that maximizes anomaly detection quality through threshold optimization.
    
    Architecture:
    - Individual: [w1, w2, ..., w17, threshold_percentile] where weights sum to 1.0
    - Fitness: Multi-objective function balancing separation quality and detection rate
    - Simplified design focused specifically on weight optimization
    """

    def __init__(self, config: Dict):
        """Initialize the genetic optimizer."""
        self.config = config
        self.toolbox = None
        self.best_individual = None
        self.best_fitness = 0.0
        self.best_params = None
        self.convergence_data = []
        
        # Results directory
        self.base_results_dir = "results"
        self.execution_results_dir = None
        
        # Data storage
        self.likelihood_scores = None
        self.original_data = None
        self.n_bn_groups = None
        
        os.makedirs(self.base_results_dir, exist_ok=True)
        
        # Focused fitness components for BN weight optimization
        self.fitness_components = {
            'separation_quality': 0.45,     # Cohen's d between anomaly and normal distributions
            'detection_rate': 0.25,         # Target anomaly detection rate (2-6%)  
            'threshold_robustness': 0.20,   # Stability across different threshold values
            'weight_diversity': 0.10        # Encourage diverse weight distributions
        }
        
        # Validate fitness weights
        total_weight = sum(self.fitness_components.values())
        if abs(total_weight - 1.0) > 1e-6:
            print(f"⚠️  Normalizing fitness weights from {total_weight:.3f} to 1.0")
            self.fitness_components = {k: v/total_weight for k, v in self.fitness_components.items()}
        
        print(f"🧬 GA Optimizer for BN Weight Optimization")
        print(f"   Fitness components: {self.fitness_components}")
        
        self._deap_initialized = False
    
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

    def _setup_deap(self, n_bn_groups: int):
        """
        Sets up DEAP genetic algorithm components based on number of BN groups.
        
        Args:
            n_bn_groups (int): Number of Bayesian Network groups (likelihood matrix columns)
        """
        if self._deap_initialized:
            return
            
        self.n_bn_groups = n_bn_groups
        print(f"🔧 Setting up DEAP for {n_bn_groups} BN groups")
        
        # Create DEAP types - Clear any existing types first
        try:
            del creator.FitnessMax
            del creator.Individual
        except AttributeError:
            pass  # Types don't exist yet
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        
        # Individual structure: [bn_weight_1, bn_weight_2, ..., bn_weight_n, threshold_percentile]
        # Weights will be normalized to sum to 1.0, threshold_percentile in range [1.0, 10.0]
        self.individual_size = n_bn_groups + 1

        # Register genetic operators
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self._evaluate_fitness)
        
        print(f"📊 Individual structure: {n_bn_groups} BN weights + 1 threshold = {self.individual_size} genes")
        self._deap_initialized = True

    def _create_individual(self):
        """
        Create a random individual representing BN weights + threshold percentile.
        
        Returns:
            Individual with BN weights (will be normalized) + threshold_percentile
        """
        genes = []
        
        # BN weights (0.1 to 1.0, will be normalized to sum to 1)
        for _ in range(self.n_bn_groups):
            genes.append(random.uniform(0.1, 1.0))
            
        # Threshold percentile (1.0 to 10.0)
        genes.append(random.uniform(1.0, 10.0))
        
        return creator.Individual(genes)

    def _decode_individual(self, individual: List[float]) -> Dict:
        """
        Convert individual genes into interpretable parameters.
        
        Args:
            individual (List[float]): Individual genome
            
        Returns:
            Dict: Decoded parameters with normalized weights
        """
        # Extract BN weights and normalize them to sum to 1
        raw_weights = np.array(individual[:self.n_bn_groups])
        bn_weights = raw_weights / np.sum(raw_weights)  # Normalize to sum to 1
        
        # Extract threshold percentile
        threshold_percentile = individual[self.n_bn_groups]
        
        return {
            'bn_weights': bn_weights,
            'threshold_percentile': threshold_percentile,
            'n_bn_groups': self.n_bn_groups,
            'aggregation_method': 'weighted'  # GA uses weighted aggregation
        }

    def _crossover(self, ind1: List[float], ind2: List[float]) -> Tuple[List[float], List[float]]:
        """
        Perform crossover between two individuals.
        
        Args:
            ind1, ind2 (List[float]): Parent individuals
            
        Returns:
            Tuple[List[float], List[float]]: Offspring individuals
        """
        # Blend crossover for BN weights (continuous values)
        alpha = 0.3  # Blend parameter
        
        for i in range(self.n_bn_groups):
            if random.random() < 0.5:
                v1, v2 = ind1[i], ind2[i]
                range_val = abs(v1 - v2)
                
                # Create blended offspring
                low = min(v1, v2) - alpha * range_val
                high = max(v1, v2) + alpha * range_val
                
                ind1[i] = np.clip(random.uniform(low, high), 0.1, 1.0)
                ind2[i] = np.clip(random.uniform(low, high), 0.1, 1.0)
        
        # Uniform crossover for threshold percentile
        thresh_idx = self.n_bn_groups
        if random.random() < 0.5:
            ind1[thresh_idx], ind2[thresh_idx] = ind2[thresh_idx], ind1[thresh_idx]
        
        return ind1, ind2

    def _mutate(self, individual: List[float]) -> Tuple[List[float]]:
        """
        Perform mutation on an individual.
        
        Args:
            individual (List[float]): Individual to mutate
            
        Returns:
            Tuple[List[float]]: Mutated individual
        """
        mutation_rate = self.config.get('mutation_rate', 0.15)
        
        # Gaussian mutation for BN weights
        for i in range(self.n_bn_groups):
            if random.random() < mutation_rate:
                # Gaussian mutation with adaptive sigma
                sigma = 0.1  # Standard deviation for mutation
                mutation_value = random.gauss(0, sigma)
                individual[i] = np.clip(individual[i] + mutation_value, 0.1, 1.0)
        
        # Gaussian mutation for threshold percentile
        thresh_idx = self.n_bn_groups
        if random.random() < mutation_rate:
            mutation_value = random.gauss(0, 0.5)  # Smaller sigma for threshold
            individual[thresh_idx] = np.clip(individual[thresh_idx] + mutation_value, 1.0, 10.0)
        
        return (individual,)

    def _evaluate_fitness(self, individual: List[float]) -> Tuple[float]:
        """
        Evaluate fitness of an individual representing BN weights + threshold.
        
        Focuses on finding optimal weights that create the best separation
        between anomaly and normal data points.
        
        Args:
            individual (List[float]): Individual genome [bn_weights..., threshold_percentile]
            
        Returns:
            Tuple[float]: Fitness score (higher is better)
        """
        try:
            # Decode individual to get parameters
            params = self._decode_individual(individual)
            
            # Calculate weighted anomaly scores using custom BN weights
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
            
            # Update best individual for tracking
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_params = params.copy()
            
            return (fitness,)
            
        except Exception as e:
            # Return very low fitness for invalid individuals
            print(f"⚠️  Fitness evaluation error: {e}")
            return (0.0,)

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
        # Convert to numpy for efficient computation
        likelihood_matrix = likelihood_scores.values
        
        # Compute weighted average of negative log-likelihoods
        # Each BN produces log-likelihood values, we want to combine them weighted
        weighted_log_likelihoods = np.average(likelihood_matrix, axis=1, weights=bn_weights)
        
        # Convert to anomaly scores (negative log-likelihood = higher anomaly score)
        anomaly_scores = -weighted_log_likelihoods
        
        return anomaly_scores

    def _determine_threshold(self, anomaly_scores: np.ndarray, threshold_percentile: float) -> float:
        """Determine threshold using percentile method."""
        return np.percentile(anomaly_scores, 100 - threshold_percentile)

    def _calculate_fitness_score(self, anomaly_scores: np.ndarray, 
                                anomaly_indices: np.ndarray, params: Dict) -> float:
        """
        Calculate comprehensive fitness score for BN weight optimization.
        
        This function implements a focused fitness evaluation specifically designed
        for optimizing Bayesian Network weights to improve anomaly detection.
        
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
        Run genetic algorithm optimization to find optimal BN weights.
        
        Args:
            likelihood_scores (pd.DataFrame): Likelihood matrix (samples x BN_groups)
            data (pd.DataFrame): Original data for validation
            
        Returns:
            Dict: Best parameters found including optimized BN weights
        """
        print("🧬 Starting BN Weight Optimization with Genetic Algorithm...")
        
        # Store data for fitness evaluation
        self.likelihood_scores = likelihood_scores
        self.original_data = data
        
        # Setup DEAP based on likelihood matrix dimensions
        n_bn_groups = likelihood_scores.shape[1]
        self._setup_deap(n_bn_groups)
        
        print(f"📊 Optimizing weights for {n_bn_groups} Bayesian Networks")
        print(f"📈 Likelihood matrix shape: {likelihood_scores.shape}")
        
        # GA parameters from config
        population_size = self.config.get('population_size', 100)
        n_generations = self.config.get('n_generations', 50)
        crossover_prob = self.config.get('crossover_probability', 0.8)
        mutation_prob = self.config.get('mutation_probability', 0.15)
        
        print(f"🔧 GA Config: Pop={population_size}, Gen={n_generations}, "
              f"Cx={crossover_prob}, Mut={mutation_prob}")
        
        try:
            # Initialize population
            population = self.toolbox.population(n=population_size)
            
            # Statistics tracking
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Hall of fame to store best individuals
            hall_of_fame = tools.HallOfFame(5)
            
            # Initialize tracking
            self.convergence_data = []
            
            # Run evolution
            final_pop, logbook = self._run_evolution(
                population, crossover_prob, mutation_prob, n_generations, 
                stats, hall_of_fame
            )
            
            # Extract best solution
            best_individual = hall_of_fame[0]
            best_params = self._decode_individual(best_individual)
            best_fitness = best_individual.fitness.values[0]
            
            print(f"\n✅ GA Optimization completed!")
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
                self._save_optimization_results(best_params, hall_of_fame, logbook)
                self._create_optimization_plots()
            
            return best_params
            
        except Exception as e:
            print(f"❌ GA optimization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return default equal weights
            return self._get_default_weights(n_bn_groups)

    def _run_evolution(self, population: List, crossover_prob: float, mutation_prob: float,
                      n_generations: int, stats: tools.Statistics, 
                      hall_of_fame: tools.HallOfFame) -> Tuple[List, tools.Logbook]:
        """Execute the genetic algorithm evolution process."""
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields
        
        # Initial evaluation
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Record initial statistics
        hall_of_fame.update(population)
        record = stats.compile(population)
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        
        # Track convergence
        self.convergence_data.append({
            'generation': 0,
            'best_fitness': record['max'],
            'avg_fitness': record['avg'],
            'std_fitness': record['std']
        })
        
        print(f"Gen 0: Best={record['max']:.2f}, Avg={record['avg']:.2f}, Std={record['std']:.2f}")

        # Evolution loop
        for generation in range(1, n_generations + 1):
            # Selection and variation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population
            population[:] = offspring
            hall_of_fame.update(population)
            
            # Record statistics
            record = stats.compile(population)
            logbook.record(gen=generation, nevals=len(invalid_ind), **record)
            
            # Track convergence
            self.convergence_data.append({
                'generation': generation,
                'best_fitness': record['max'],
                'avg_fitness': record['avg'], 
                'std_fitness': record['std']
            })
            
            # Progress reporting
            if generation % 10 == 0 or generation == n_generations:
                print(f"Gen {generation}: Best={record['max']:.2f}, Avg={record['avg']:.2f}, "
                      f"Std={record['std']:.2f}")

        return population, logbook

    def _get_default_weights(self, n_bn_groups: int) -> Dict:
        """Return default equal weights as fallback."""
        equal_weights = np.ones(n_bn_groups) / n_bn_groups
        return {
            'bn_weights': equal_weights,
            'threshold_percentile': 5.0,
            'n_bn_groups': n_bn_groups
        }

    def _save_optimization_results(self, best_params: Dict, hall_of_fame: tools.HallOfFame, 
                                 logbook: tools.Logbook):
        """Save optimization results to files."""
        if not self.execution_results_dir:
            return
            
        try:
            import os
            import json
            
            # Save best parameters
            results = {
                'best_fitness': float(hall_of_fame[0].fitness.values[0]),
                'best_bn_weights': best_params['bn_weights'].tolist(),
                'best_threshold_percentile': float(best_params['threshold_percentile']),
                'n_bn_groups': int(best_params['n_bn_groups']),
                'convergence_data': self.convergence_data,
                'ga_config': {
                    'population_size': self.config.get('population_size', 100),
                    'n_generations': self.config.get('n_generations', 50),
                    'crossover_probability': self.config.get('crossover_probability', 0.8),
                    'mutation_probability': self.config.get('mutation_probability', 0.15)
                }
            }
            
            # Save to JSON file
            results_path = os.path.join(self.execution_results_dir, 'ga_summary.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            print(f"📁 Results saved to {results_path}")
            
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")

    def _create_optimization_plots(self):
        """Create optimization convergence plots."""
        if not self.convergence_data:
            return
            
        try:
            import matplotlib.pyplot as plt
            import os
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            generations = [d['generation'] for d in self.convergence_data]
            best_fitness = [d['best_fitness'] for d in self.convergence_data]
            avg_fitness = [d['avg_fitness'] for d in self.convergence_data]
            std_fitness = [d['std_fitness'] for d in self.convergence_data]
            
            # Plot 1: Fitness evolution
            ax1.plot(generations, best_fitness, 'r-', linewidth=3, label='Best Fitness', marker='o')
            ax1.plot(generations, avg_fitness, 'b-', linewidth=2, label='Average Fitness', alpha=0.8)
            ax1.fill_between(generations, 
                           np.array(avg_fitness) - np.array(std_fitness),
                           np.array(avg_fitness) + np.array(std_fitness),
                           alpha=0.2, color='blue', label='±1 Std Dev')
            
            ax1.set_title('GA Fitness Evolution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness Score')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 105)
            
            # Plot 2: BN Weight Distribution  
            if self.best_params and 'bn_weights' in self.best_params:
                weights = self.best_params['bn_weights']
                bn_indices = np.arange(len(weights))
                
                bars = ax2.bar(bn_indices, weights, color='steelblue', alpha=0.7)
                ax2.set_title('Optimized BN Weights', fontsize=14, fontweight='bold')
                ax2.set_xlabel('BN Index')
                ax2.set_ylabel('Weight')
                ax2.grid(True, axis='y', alpha=0.3)
                
                # Highlight top weights
                top_indices = np.argsort(weights)[-3:]
                for idx in top_indices:
                    bars[idx].set_color('coral')
                    bars[idx].set_alpha(0.9)
            else:
                ax2.text(0.5, 0.5, 'No weight data available', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title('BN Weight Distribution', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            if self.execution_results_dir:
                plot_path = os.path.join(self.execution_results_dir, 'ga_fitness_evolution.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"📊 Plots saved to {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Could not create plots: {e}")

    def get_optimization_summary(self) -> Dict:
        """Return optimization summary for reporting."""
        if not self.best_params or not self.convergence_data:
            return {}
            
        return {
            'best_fitness': self.best_fitness,
            'best_bn_weights': self.best_params['bn_weights'].tolist(),
            'best_threshold_percentile': self.best_params['threshold_percentile'],
            'n_generations': len(self.convergence_data),
            'final_avg_fitness': self.convergence_data[-1]['avg_fitness'] if self.convergence_data else 0,
            'convergence_achieved': self.best_fitness > 70  # Good fitness threshold
        }
