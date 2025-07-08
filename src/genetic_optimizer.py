import pandas as pd
import numpy as np
from deap import creator, base, tools, algorithms
from typing import Dict, List, Tuple, Any
import random
import warnings

class GeneticOptimizer:
    """
    Optimizes anomaly detection parameters using genetic algorithms.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the genetic optimizer.
        
        Args:
            config (Dict): Configuration for GA
        """
        self.config = config
        self.toolbox = None
        self.best_individual = None
        self.best_fitness = None
        self.fitness_history = []
        
        # Initialize DEAP components
        self._setup_deap()
    
    def _setup_deap(self):
        """Setup DEAP genetic algorithm components."""
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize fitness
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Initialize toolbox
        self.toolbox = base.Toolbox()
        
        # Define parameter bounds and types
        self._define_parameters()
        
        # Register genetic operators
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self._evaluate_fitness)
    
    def _define_parameters(self):
        """Define the parameters to optimize."""
        self.param_bounds = {
            'threshold_percentile': (1.0, 10.0),      # Anomaly percentage
            'aggregation_method_idx': (0, 4),          # Index for aggregation method
            'use_zscore': (0, 1),                      # Boolean for z-score transform
            'threshold_method_idx': (0, 3)             # Index for threshold method
        }
        
        self.aggregation_methods = ['mean', 'min', 'median', 'weighted', 'sum']
        self.threshold_methods = ['percentile', 'std', 'iqr', 'adaptive']
    
    def _create_individual(self):
        """Create a random individual (parameter set)."""
        individual = []
        for param, (min_val, max_val) in self.param_bounds.items():
            if 'idx' in param or param == 'use_zscore':
                # Integer parameters
                individual.append(random.randint(int(min_val), int(max_val)))
            else:
                # Float parameters
                individual.append(random.uniform(min_val, max_val))
        return individual
    
    def _decode_individual(self, individual: List) -> Dict:
        """
        Decode individual to parameter dictionary.
        
        Args:
            individual (List): Encoded individual
            
        Returns:
            Dict: Decoded parameters
        """
        params = {
            'threshold_percentile': individual[0],
            'aggregation_method': self.aggregation_methods[int(individual[1])],
            'use_zscore_transformation': bool(individual[2]),
            'threshold_method': self.threshold_methods[int(individual[3])]
        }
        return params
    
    def _crossover(self, ind1: List, ind2: List) -> Tuple[List, List]:
        """
        Crossover operation for two individuals.
        
        Args:
            ind1 (List): First parent
            ind2 (List): Second parent
            
        Returns:
            Tuple[List, List]: Two offspring
        """
        # Blend crossover for continuous parameters
        alpha = 0.5
        
        for i in range(len(ind1)):
            if random.random() < self.config.get('crossover_rate', 0.8):
                if 'idx' not in list(self.param_bounds.keys())[i] and i != 2:  # Float parameters
                    # Blend crossover
                    gamma = (1 + 2 * alpha) * random.random() - alpha
                    ind1[i] = (1 - gamma) * ind1[i] + gamma * ind2[i]
                    gamma = (1 + 2 * alpha) * random.random() - alpha
                    ind2[i] = (1 - gamma) * ind2[i] + gamma * ind1[i]
                else:
                    # Discrete crossover
                    if random.random() < 0.5:
                        ind1[i], ind2[i] = ind2[i], ind1[i]
        
        # Ensure bounds
        self._enforce_bounds(ind1)
        self._enforce_bounds(ind2)
        
        return ind1, ind2
    
    def _mutate(self, individual: List) -> Tuple[List]:
        """
        Mutation operation for an individual.
        
        Args:
            individual (List): Individual to mutate
            
        Returns:
            Tuple[List]: Mutated individual
        """
        mutation_rate = self.config.get('mutation_rate', 0.1)
        
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                param_name = list(self.param_bounds.keys())[i]
                min_val, max_val = self.param_bounds[param_name]
                
                if 'idx' in param_name or param_name == 'use_zscore':
                    # Integer mutation
                    individual[i] = random.randint(int(min_val), int(max_val))
                else:
                    # Gaussian mutation for float parameters
                    sigma = (max_val - min_val) * 0.1  # 10% of range
                    individual[i] += random.gauss(0, sigma)
        
        # Ensure bounds
        self._enforce_bounds(individual)
        
        return individual,
    
    def _enforce_bounds(self, individual: List):
        """
        Enforce parameter bounds on an individual.
        
        Args:
            individual (List): Individual to bound
        """
        for i, (param, (min_val, max_val)) in enumerate(self.param_bounds.items()):
            individual[i] = max(min_val, min(max_val, individual[i]))
            
            # Round integer parameters
            if 'idx' in param or param == 'use_zscore':
                individual[i] = int(round(individual[i]))
    
    def optimize(self, likelihood_scores: pd.DataFrame, data: pd.DataFrame) -> Dict:
        """
        Optimize anomaly detection parameters.
        
        Args:
            likelihood_scores (pd.DataFrame): Likelihood scores from BN learning
            data (pd.DataFrame): Original data for validation
            
        Returns:
            Dict: Optimized parameters
        """
        print("     Starting genetic algorithm optimization...")
        
        # Store data for fitness evaluation
        self.likelihood_scores = likelihood_scores
        self.original_data = data
        
        # Create initial population
        population_size = self.config.get('population_size', 50)
        population = self.toolbox.population(n=population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution parameters
        generations = self.config.get('generations', 100)
        crossover_prob = self.config.get('crossover_rate', 0.8)
        mutation_prob = self.config.get('mutation_rate', 0.1)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Hall of fame to keep track of best individuals
        hof = tools.HallOfFame(1)
        
        # Run evolution
        try:
            population, logbook = algorithms.eaSimple(
                population, self.toolbox, 
                cxpb=crossover_prob, mutpb=mutation_prob, 
                ngen=generations, stats=stats, halloffame=hof, verbose=False
            )
            
            # Store results
            self.best_individual = hof[0]
            self.best_fitness = hof[0].fitness.values[0]
            self.fitness_history = [gen['max'] for gen in logbook]
            
            # Decode best parameters
            best_params = self._decode_individual(self.best_individual)
            
            print(f"     ✅ GA optimization completed")
            print(f"     Best fitness: {self.best_fitness:.4f}")
            print(f"     Best parameters: {best_params}")
            
            return best_params
            
        except Exception as e:
            print(f"     ❌ GA optimization failed: {str(e)}")
            # Return default parameters
            return {
                'threshold_percentile': 5.0,
                'aggregation_method': 'mean',
                'use_zscore_transformation': True,
                'threshold_method': 'percentile'
            }
    
    def _evaluate_fitness(self, individual: List) -> Tuple[float]:
        """
        Evaluate fitness of an individual.
        
        Args:
            individual (List): Individual to evaluate
            
        Returns:
            Tuple[float]: Fitness value
        """
        try:
            # Decode parameters
            params = self._decode_individual(individual)
            
            # Create temporary anomaly detector with these parameters
            from .anomaly_detector import AnomalyDetector
            
            temp_config = {
                'aggregation_method': params['aggregation_method'],
                'threshold_method': params['threshold_method'],
                'threshold_percentile': params['threshold_percentile'],
                'use_zscore_transformation': params['use_zscore_transformation']
            }
            
            temp_detector = AnomalyDetector(temp_config)
            
            # Detect anomalies with these parameters (silent mode for GA)
            anomaly_scores, anomaly_indices = temp_detector.detect_anomalies(self.likelihood_scores, verbose=False)
            
            # Calculate fitness based on multiple criteria
            fitness = self._calculate_fitness(anomaly_scores, anomaly_indices, params)
            
            return fitness,
            
        except Exception as e:
            # Return very low fitness for invalid parameters
            return -1000.0,
    
    def _calculate_fitness(self, anomaly_scores: np.ndarray, 
                          anomaly_indices: np.ndarray, params: Dict) -> float:
        """
        Calculate fitness based on anomaly detection results.
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            anomaly_indices (np.ndarray): Anomaly indices
            params (Dict): Parameters used
            
        Returns:
            float: Fitness value
        """
        try:
            # Fitness components
            fitness_components = []
            
            # 1. Score separation: difference between anomaly and normal scores
            if len(anomaly_indices) > 0 and len(anomaly_indices) < len(anomaly_scores):
                normal_indices = np.setdiff1d(np.arange(len(anomaly_scores)), anomaly_indices)
                anomaly_score_mean = np.mean(anomaly_scores[anomaly_indices])
                normal_score_mean = np.mean(anomaly_scores[normal_indices])
                separation = anomaly_score_mean - normal_score_mean
                fitness_components.append(separation * 10)  # Weight: 10
            else:
                fitness_components.append(-100)  # Penalty for all or no anomalies
            
            # 2. Score distribution quality: prefer well-separated distributions
            score_std = np.std(anomaly_scores)
            if score_std > 0:
                fitness_components.append(score_std * 5)  # Weight: 5
            
            # 3. Anomaly rate penalty: prefer reasonable anomaly rates (1-10%)
            anomaly_rate = len(anomaly_indices) / len(anomaly_scores) * 100
            rate_penalty = abs(anomaly_rate - 5)  # Target 5% anomaly rate
            fitness_components.append(-rate_penalty)  # Penalty for deviation
            
            # 4. Threshold quality: prefer thresholds that create clear separation
            if len(anomaly_indices) > 0:
                threshold_score = np.min(anomaly_scores[anomaly_indices])
                normal_max = np.max(anomaly_scores[np.setdiff1d(np.arange(len(anomaly_scores)), anomaly_indices)])
                threshold_separation = threshold_score - normal_max
                fitness_components.append(threshold_separation * 15)  # Weight: 15
            
            # 5. Stability bonus: prefer methods that are more stable
            method_stability = self._get_method_stability_bonus(params)
            fitness_components.append(method_stability)
            
            # Combine fitness components
            total_fitness = sum(fitness_components)
            
            # Add noise reduction for repeated evaluations
            total_fitness += random.gauss(0, 0.1)
            
            return total_fitness
            
        except Exception:
            return -1000.0
    
    def _get_method_stability_bonus(self, params: Dict) -> float:
        """
        Get stability bonus for certain parameter combinations.
        
        Args:
            params (Dict): Parameters
            
        Returns:
            float: Stability bonus
        """
        bonus = 0.0
        
        # Prefer mean aggregation (more stable)
        if params['aggregation_method'] == 'mean':
            bonus += 5.0
        elif params['aggregation_method'] == 'median':
            bonus += 3.0
        
        # Prefer percentile threshold (more interpretable)
        if params['threshold_method'] == 'percentile':
            bonus += 3.0
        elif params['threshold_method'] == 'adaptive':
            bonus += 2.0
        
        # Prefer z-score transformation
        if params['use_zscore_transformation']:
            bonus += 2.0
        
        return bonus
    
    def get_optimization_summary(self) -> Dict:
        """
        Get summary of optimization results.
        
        Returns:
            Dict: Optimization summary
        """
        if self.best_individual is None:
            return {'status': 'Not optimized'}
        
        best_params = self._decode_individual(self.best_individual)
        
        summary = {
            'status': 'Optimized',
            'best_fitness': self.best_fitness,
            'best_parameters': best_params,
            'generations_run': len(self.fitness_history),
            'fitness_improvement': self.fitness_history[-1] - self.fitness_history[0] if len(self.fitness_history) > 1 else 0,
            'convergence_generation': self._find_convergence_generation()
        }
        
        return summary
    
    def _find_convergence_generation(self) -> int:
        """
        Find the generation where the algorithm converged.
        
        Returns:
            int: Convergence generation
        """
        if len(self.fitness_history) < 10:
            return len(self.fitness_history)
        
        # Look for plateau in fitness (convergence)
        for i in range(10, len(self.fitness_history)):
            recent_improvement = max(self.fitness_history[i-10:i]) - min(self.fitness_history[i-10:i])
            if recent_improvement < 0.01:  # Small improvement threshold
                return i - 10
        
        return len(self.fitness_history)
    
    def plot_fitness_evolution(self):
        """Plot the evolution of fitness over generations."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.fitness_history:
                print("No fitness history available")
                return
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.fitness_history, 'b-', linewidth=2, label='Best Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Genetic Algorithm Fitness Evolution')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add convergence point
            conv_gen = self._find_convergence_generation()
            if conv_gen < len(self.fitness_history):
                plt.axvline(x=conv_gen, color='r', linestyle='--', 
                           label=f'Convergence (Gen {conv_gen})')
                plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Plotting failed: {str(e)}")
    
    def validate_optimized_parameters(self, likelihood_scores: pd.DataFrame) -> Dict:
        """
        Validate the optimized parameters on the data.
        
        Args:
            likelihood_scores (pd.DataFrame): Likelihood scores
            
        Returns:
            Dict: Validation results
        """
        if self.best_individual is None:
            return {'status': 'No optimized parameters available'}
        
        best_params = self._decode_individual(self.best_individual)
        
        # Test the parameters
        from .anomaly_detector import AnomalyDetector
        
        detector = AnomalyDetector(best_params)
        anomaly_scores, anomaly_indices = detector.detect_anomalies(likelihood_scores)
        
        # Calculate validation metrics
        validation_results = {
            'parameters': best_params,
            'anomaly_count': len(anomaly_indices),
            'anomaly_rate': len(anomaly_indices) / len(anomaly_scores) * 100,
            'score_statistics': {
                'mean': np.mean(anomaly_scores),
                'std': np.std(anomaly_scores),
                'min': np.min(anomaly_scores),
                'max': np.max(anomaly_scores)
            },
            'threshold': detector.threshold,
            'fitness': self.best_fitness
        }
        
        return validation_results
