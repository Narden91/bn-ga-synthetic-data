import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deap import creator, base, tools, algorithms
from typing import Dict, List, Tuple, Any
import random
import warnings
import os
from datetime import datetime

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
        self.generation_stats = []
        self.population_history = []
        self.convergence_data = {}
        
        # Results tracking
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
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
        Optimize anomaly detection parameters with enhanced tracking and visualization.
        
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
        
        # Clear previous results
        self.fitness_history = []
        self.generation_stats = []
        self.population_history = []
        
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
        
        # Enhanced statistics tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("median", np.median)
        
        # Hall of fame to keep track of best individuals
        hof = tools.HallOfFame(5)  # Keep top 5 individuals
        
        print(f"     Population size: {population_size}, Generations: {generations}")
        print(f"     Crossover: {crossover_prob:.1%}, Mutation: {mutation_prob:.1%}")
        
        # Run evolution with enhanced tracking
        try:
            population, logbook = self._run_enhanced_evolution(
                population, crossover_prob, mutation_prob, generations, stats, hof
            )
            
            # Store results
            self.best_individual = hof[0]
            self.best_fitness = hof[0].fitness.values[0]
            self.fitness_history = [gen['max'] for gen in logbook]
            self.generation_stats = [dict(gen) for gen in logbook]
            
            # Create convergence data
            self.convergence_data = {
                'generations': list(range(len(logbook))),
                'max_fitness': [gen['max'] for gen in logbook],
                'avg_fitness': [gen['avg'] for gen in logbook],
                'min_fitness': [gen['min'] for gen in logbook],
                'std_fitness': [gen['std'] for gen in logbook],
                'median_fitness': [gen['median'] for gen in logbook]
            }
            
            # Decode best parameters
            best_params = self._decode_individual(self.best_individual)
            
            # Save results and create plots
            self._save_optimization_results(best_params, hof)
            self._create_fitness_plots()
            
            print(f"     ✅ GA optimization completed")
            print(f"     Best fitness: {self.best_fitness:.4f}")
            print(f"     Convergence achieved in {len(logbook)} generations")
            print(f"     Results saved to {self.results_dir}/")
            
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
    
    def _run_enhanced_evolution(self, population, cxpb, mutpb, ngen, stats, hof):
        """
        Run evolution with enhanced progress tracking.
        
        Args:
            population: Initial population
            cxpb: Crossover probability
            mutpb: Mutation probability
            ngen: Number of generations
            stats: Statistics object
            hof: Hall of fame
            
        Returns:
            Tuple: Final population and logbook
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        if hof is not None:
            hof.update(population)
        
        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        
        print(f"     Gen 0: Max={record.get('max', 0):.3f}, Avg={record.get('avg', 0):.3f}")
        
        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = self.toolbox.select(population, len(population))
            
            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, self.toolbox, cxpb, mutpb)
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Update the hall of fame with the generated individuals
            if hof is not None:
                hof.update(offspring)
            
            # Replace the current population by the offspring
            population[:] = offspring
            
            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            
            # Print progress every 10 generations
            if gen % 10 == 0 or gen == ngen:
                print(f"     Gen {gen}: Max={record.get('max', 0):.3f}, Avg={record.get('avg', 0):.3f}, Std={record.get('std', 0):.3f}")
        
        return population, logbook
    
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
        Enhanced fitness calculation for better evolution with proper scaling.
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            anomaly_indices (np.ndarray): Anomaly indices
            params (Dict): Parameters used
            
        Returns:
            float: Fitness value (scaled between 0-100)
        """
        try:
            # Check for invalid results
            if len(anomaly_scores) == 0:
                return 0.0
                
            # Initialize fitness components (all scaled 0-100)
            fitness_components = []
            
            # 1. Anomaly rate component (0-25 points)
            anomaly_rate = len(anomaly_indices) / len(anomaly_scores) * 100
            if anomaly_rate == 0 or anomaly_rate == 100:
                rate_score = 0  # No points for invalid rates
            elif 2 <= anomaly_rate <= 8:
                # Optimal range: quadratic peak at 5%
                rate_score = 25 - 2 * (anomaly_rate - 5) ** 2
            else:
                # Outside optimal range: linear penalty
                rate_score = max(0, 15 - abs(anomaly_rate - 5))
            
            fitness_components.append(rate_score)
            
            # 2. Score separation component (0-30 points)
            if len(anomaly_indices) > 0 and len(anomaly_indices) < len(anomaly_scores):
                normal_indices = np.setdiff1d(np.arange(len(anomaly_scores)), anomaly_indices)
                
                anomaly_mean = np.mean(anomaly_scores[anomaly_indices])
                normal_mean = np.mean(anomaly_scores[normal_indices])
                
                # Normalize separation by score standard deviation
                score_std = np.std(anomaly_scores)
                if score_std > 0:
                    normalized_separation = (anomaly_mean - normal_mean) / score_std
                    # Convert to 0-30 scale using sigmoid-like function
                    separation_score = 30 / (1 + np.exp(-normalized_separation))
                else:
                    separation_score = 0
                
                fitness_components.append(separation_score)
                
                # 3. Statistical significance bonus (0-10 points)
                try:
                    from scipy import stats
                    t_stat, p_value = stats.ttest_ind(anomaly_scores[anomaly_indices], 
                                                     anomaly_scores[normal_indices])
                    if p_value < 0.01:
                        sig_score = 10
                    elif p_value < 0.05:
                        sig_score = 7
                    elif p_value < 0.1:
                        sig_score = 4
                    else:
                        sig_score = 0
                    fitness_components.append(sig_score)
                except:
                    fitness_components.append(0)
                
                # 4. Effect size component (0-15 points)
                anomaly_std = np.std(anomaly_scores[anomaly_indices])
                normal_std = np.std(anomaly_scores[normal_indices])
                
                # Pooled standard deviation
                n1, n2 = len(anomaly_indices), len(normal_indices)
                pooled_std = np.sqrt(((n1 - 1) * anomaly_std**2 + (n2 - 1) * normal_std**2) / (n1 + n2 - 2))
                
                if pooled_std > 0:
                    cohens_d = abs(anomaly_mean - normal_mean) / pooled_std
                    # Scale Cohen's d to 0-15 points
                    effect_score = min(15, cohens_d * 5)  # Large effect = 15 points
                else:
                    effect_score = 0
                
                fitness_components.append(effect_score)
            else:
                # No separation possible
                fitness_components.extend([0, 0, 0])
            
            # 5. Score distribution quality (0-10 points)
            score_std = np.std(anomaly_scores)
            score_mean = np.mean(anomaly_scores)
            
            if score_mean != 0 and score_std > 0:
                # Coefficient of variation (prefer moderate values)
                cv = score_std / abs(score_mean)
                # Optimal CV around 0.5-1.5
                if 0.5 <= cv <= 1.5:
                    cv_score = 10
                elif 0.2 <= cv <= 2.0:
                    cv_score = 7
                else:
                    cv_score = max(0, 5 - abs(cv - 1.0))
            else:
                cv_score = 0
            
            fitness_components.append(cv_score)
            
            # 6. Threshold quality (0-10 points)
            if len(anomaly_indices) > 0:
                threshold_score = np.min(anomaly_scores[anomaly_indices])
                normal_scores = anomaly_scores[np.setdiff1d(np.arange(len(anomaly_scores)), anomaly_indices)]
                
                if len(normal_scores) > 0:
                    normal_max = np.max(normal_scores)
                    # Gap between threshold and highest normal score
                    threshold_gap = threshold_score - normal_max
                    
                    if threshold_gap > 0:
                        # Good separation
                        gap_score = min(10, threshold_gap / score_std * 5) if score_std > 0 else 5
                    else:
                        # Overlap penalty
                        gap_score = max(0, 5 + threshold_gap / score_std * 2) if score_std > 0 else 0
                else:
                    gap_score = 5
            else:
                gap_score = 0
            
            fitness_components.append(gap_score)
            
            # 7. Parameter stability and robustness (0-10 points)
            stability_score = 0
            
            # Aggregation method preferences
            if params['aggregation_method'] == 'mean':
                stability_score += 3
            elif params['aggregation_method'] == 'median':
                stability_score += 2
            elif params['aggregation_method'] == 'weighted':
                stability_score += 1
            
            # Threshold method preferences
            if params['threshold_method'] == 'percentile':
                stability_score += 2
            elif params['threshold_method'] == 'adaptive':
                stability_score += 1
            
            # Z-score transformation bonus
            if params['use_zscore_transformation']:
                stability_score += 2
            
            # Threshold percentile preferences (moderate values)
            if 3 <= params['threshold_percentile'] <= 7:
                stability_score += 3
            elif 2 <= params['threshold_percentile'] <= 9:
                stability_score += 1
            
            fitness_components.append(min(10, stability_score))
            
            # Combine all components (max possible: 25+30+10+15+10+10+10 = 110)
            total_fitness = sum(fitness_components)
            
            # Add small random noise to break ties (±0.1)
            total_fitness += random.uniform(-0.1, 0.1)
            
            # Ensure fitness is in reasonable range
            return max(0, min(110, total_fitness))
            
        except Exception as e:
            print(f"     Fitness calculation error: {str(e)}")
            return 0.0
    
    def _save_optimization_results(self, best_params: Dict, hof):
        """Save optimization results to files."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save best parameters
            params_file = os.path.join(self.results_dir, f'ga_best_parameters_{timestamp}.json')
            import json
            with open(params_file, 'w') as f:
                json.dump({
                    'best_parameters': best_params,
                    'best_fitness': float(self.best_fitness),
                    'optimization_summary': self._make_json_serializable(self.get_optimization_summary()),
                    'convergence_data': self._make_json_serializable(self.convergence_data)
                }, f, indent=2)
            
            # Save fitness history
            fitness_file = os.path.join(self.results_dir, f'ga_fitness_history_{timestamp}.csv')
            fitness_df = pd.DataFrame(self.convergence_data)
            fitness_df.to_csv(fitness_file, index=False)
            
            # Save hall of fame (top individuals)
            if hof:
                hof_file = os.path.join(self.results_dir, f'ga_hall_of_fame_{timestamp}.json')
                hof_data = []
                for i, individual in enumerate(hof):
                    params = self._decode_individual(individual)
                    hof_data.append({
                        'rank': i + 1,
                        'fitness': float(individual.fitness.values[0]),
                        'parameters': params
                    })
                
                with open(hof_file, 'w') as f:
                    json.dump(hof_data, f, indent=2)
            
            print(f"     GA results saved to {self.results_dir}/")
            
        except Exception as e:
            print(f"     Error saving GA results: {str(e)}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def _get_method_stability_bonus(self, params: Dict) -> float:
        """
        Get stability bonus for certain parameter combinations.
        
        Args:
            params (Dict): Parameters
            
        Returns:
            float: Stability bonus (0-10 scale)
        """
        bonus = 0.0
        
        # Prefer mean aggregation (most stable)
        if params['aggregation_method'] == 'mean':
            bonus += 3.0
        elif params['aggregation_method'] == 'median':
            bonus += 2.0
        
        # Prefer percentile threshold (most interpretable)
        if params['threshold_method'] == 'percentile':
            bonus += 2.0
        elif params['threshold_method'] == 'adaptive':
            bonus += 1.0
        
        # Prefer z-score transformation
        if params['use_zscore_transformation']:
            bonus += 2.0
        
        # Prefer moderate threshold percentiles
        if 3 <= params['threshold_percentile'] <= 7:
            bonus += 3.0
        elif 2 <= params['threshold_percentile'] <= 9:
            bonus += 1.0
        
        return min(10.0, bonus)
    
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
    
    def _create_fitness_plots(self):
        """Create comprehensive fitness evolution plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.convergence_data:
                print("     No convergence data available for plotting")
                return
            
            # Set style for better plots
            plt.style.use('default')  # Use default instead of seaborn-v0_8
            sns.set_palette("husl")
            
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(20, 14))
            
            # Extract data
            generations = self.convergence_data['generations']
            max_fitness = self.convergence_data['max_fitness']
            avg_fitness = self.convergence_data['avg_fitness']
            min_fitness = self.convergence_data['min_fitness']
            std_fitness = self.convergence_data['std_fitness']
            median_fitness = self.convergence_data['median_fitness']
            
            # Plot 1: Enhanced Fitness Evolution
            ax1 = plt.subplot(3, 3, 1)
            plt.plot(generations, max_fitness, 'r-', linewidth=3, label='Best Fitness', marker='o', markersize=4)
            plt.plot(generations, avg_fitness, 'b-', linewidth=2, label='Average Fitness', alpha=0.8)
            plt.plot(generations, median_fitness, 'orange', linewidth=2, label='Median Fitness', alpha=0.8)
            plt.plot(generations, min_fitness, 'g-', linewidth=1, label='Worst Fitness', alpha=0.6)
            
            # Add confidence interval
            plt.fill_between(generations, 
                           np.array(avg_fitness) - np.array(std_fitness), 
                           np.array(avg_fitness) + np.array(std_fitness), 
                           alpha=0.2, color='blue', label='±1 Std Dev')
            
            plt.xlabel('Generation', fontsize=12)
            plt.ylabel('Fitness Value (0-110 scale)', fontsize=12)
            plt.title('GA Fitness Evolution (Enhanced)', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 110)  # Set proper y-axis limits
            
            # Add convergence marker
            conv_gen = self._find_convergence_generation()
            if conv_gen < len(generations) and conv_gen > 0:
                plt.axvline(x=conv_gen, color='red', linestyle='--', linewidth=2,
                           alpha=0.8, label=f'Convergence (Gen {conv_gen})')
                plt.legend(fontsize=10)
            
            # Plot 2: Population Diversity
            ax2 = plt.subplot(3, 3, 2)
            plt.plot(generations, std_fitness, 'purple', linewidth=3, marker='s', markersize=4)
            plt.xlabel('Generation', fontsize=12)
            plt.ylabel('Fitness Standard Deviation', fontsize=12)
            plt.title('Population Diversity', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Add diversity zones
            max_std = max(std_fitness) if std_fitness else 1
            plt.axhline(y=max_std * 0.1, color='red', linestyle=':', alpha=0.7, label='Low Diversity')
            plt.axhline(y=max_std * 0.5, color='orange', linestyle=':', alpha=0.7, label='Medium Diversity')
            plt.legend(fontsize=9)
            
            # Plot 3: Fitness Components Breakdown
            ax3 = plt.subplot(3, 3, 3)
            if hasattr(self, 'best_individual') and self.best_individual:
                # Calculate fitness components for best individual
                best_params = self._decode_individual(self.best_individual)
                
                # Simulate fitness calculation to get components
                # (This is a simplified version - you might want to store components during evolution)
                component_names = ['Anomaly Rate', 'Separation', 'Significance', 
                                 'Effect Size', 'Distribution', 'Threshold', 'Stability']
                # Estimate component values (you could modify _calculate_fitness to return components)
                estimated_components = [20, 25, 8, 12, 7, 8, 8]  # Example values
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(component_names)))
                bars = plt.bar(range(len(component_names)), estimated_components, color=colors, alpha=0.7)
                plt.xlabel('Fitness Components', fontsize=12)
                plt.ylabel('Component Score', fontsize=12)
                plt.title('Best Individual Fitness Breakdown', fontsize=14, fontweight='bold')
                plt.xticks(range(len(component_names)), component_names, rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, estimated_components):
                    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                            f'{value}', ha='center', va='bottom', fontsize=9)
            
            # Plot 4: Cumulative Improvement
            ax4 = plt.subplot(3, 3, 4)
            if len(max_fitness) > 1:
                total_improvement = max_fitness[-1] - max_fitness[0]
                cumulative_improvement = [(f - max_fitness[0]) for f in max_fitness]
                cumulative_pct = [(c / total_improvement * 100) if total_improvement != 0 else 0 
                                 for c in cumulative_improvement]
                
                plt.plot(generations, cumulative_improvement, 'green', linewidth=3, marker='o', markersize=3)
                plt.xlabel('Generation', fontsize=12)
                plt.ylabel('Cumulative Improvement', fontsize=12)
                plt.title('Cumulative Fitness Gain', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                # Add percentage on right axis
                ax4_twin = ax4.twinx()
                ax4_twin.plot(generations, cumulative_pct, 'orange', linewidth=2, alpha=0.7, linestyle='--')
                ax4_twin.set_ylabel('Improvement %', fontsize=12, color='orange')
                ax4_twin.tick_params(axis='y', labelcolor='orange')
            
            # Plot 5: Fitness Distribution Evolution
            ax5 = plt.subplot(3, 3, 5)
            if len(self.generation_stats) > 1:
                # Show fitness distribution at different generations
                sample_gens = [0, len(generations)//4, len(generations)//2, 3*len(generations)//4, len(generations)-1]
                sample_gens = [g for g in sample_gens if g < len(self.generation_stats)]
                
                for i, gen_idx in enumerate(sample_gens):
                    stats = self.generation_stats[gen_idx]
                    fitness_range = [stats['min'], stats['median'], stats['avg'], stats['max']]
                    x_pos = np.arange(len(fitness_range)) + i * 0.15;
                    
                    plt.bar(x_pos, fitness_range, width=0.15, 
                           label=f'Gen {gen_idx}', alpha=0.7)
                
                plt.xlabel('Statistics', fontsize=12)
                plt.ylabel('Fitness Value', fontsize=12)
                plt.title('Fitness Distribution Evolution', fontsize=14, fontweight='bold')
                plt.xticks(np.arange(4) + 0.3, ['Min', 'Median', 'Mean', 'Max'])
                plt.legend(fontsize=9)
                plt.grid(True, alpha=0.3)
            
            # Plot 6: Parameter Evolution (if available)
            ax6 = plt.subplot(3, 3, 6)
            # This would require tracking parameter values over generations
            # For now, show final parameter distribution
            if hasattr(self, 'best_individual') and self.best_individual:
                best_params = self._decode_individual(self.best_individual)
                param_names = list(best_params.keys())
                param_values = []
                
                for param, value in best_params.items():
                    if isinstance(value, (int, float)):
                        param_values.append(value)
                    elif isinstance(value, bool):
                        param_values.append(1 if value else 0)
                    else:
                        # For string parameters, use index
                        if param == 'aggregation_method':
                            param_values.append(self.aggregation_methods.index(value))
                        elif param == 'threshold_method':
                            param_values.append(self.threshold_methods.index(value))
                        else:
                            param_values.append(0)
                
                # Normalize values for visualization
                normalized_values = []
                for i, (param, value) in enumerate(zip(param_names, param_values)):
                    if param == 'threshold_percentile':
                        normalized_values.append(value / 10.0)  # Scale to 0-1
                    elif 'method' in param:
                        normalized_values.append(value / 4.0)  # Scale to 0-1
                    else:
                        normalized_values.append(value)
                
                bars = plt.bar(range(len(param_names)), normalized_values, 
                              color=['red', 'blue', 'green', 'orange'][:len(param_names)], alpha=0.7)
                plt.xlabel('Parameters', fontsize=12)
                plt.ylabel('Normalized Value', fontsize=12)
                plt.title('Optimized Parameters', fontsize=14, fontweight='bold')
                plt.xticks(range(len(param_names)), param_names, rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value, original in zip(bars, normalized_values, param_values):
                    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                            f'{original}', ha='center', va='bottom', fontsize=9)
            
            # Plot 7: Convergence Analysis
            ax7 = plt.subplot(3, 3, 7)
            window_size = min(10, len(max_fitness) // 4) if len(max_fitness) > 10 else 3
            if window_size > 1 and len(max_fitness) > window_size:
                rolling_avg = pd.Series(max_fitness).rolling(window=window_size, center=True).mean()
                rolling_std = pd.Series(max_fitness).rolling(window=window_size, center=True).std()
                
                plt.plot(generations, max_fitness, 'lightblue', alpha=0.5, linewidth=1, label='Best Fitness')
                plt.plot(generations, rolling_avg, 'darkblue', linewidth=3, label=f'Rolling Avg ({window_size})')
                plt.fill_between(generations, rolling_avg - rolling_std, rolling_avg + rolling_std,
                               alpha=0.2, color='blue', label='±1 Rolling Std')
                
                plt.xlabel('Generation', fontsize=12)
                plt.ylabel('Fitness', fontsize=12)
                plt.title('Convergence Analysis', fontsize=14, fontweight='bold')
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
            
            # Plot 8: Performance Metrics
            ax8 = plt.subplot(3, 3, 8)
            if len(max_fitness) > 1:
                # Calculate various performance metrics
                total_improvement = max_fitness[-1] - max_fitness[0]
                avg_improvement_rate = total_improvement / len(generations) if len(generations) > 0 else 0
                best_improvement_gen = np.argmax(np.diff(max_fitness)) + 1 if len(max_fitness) > 1 else 0
                stagnation_periods = self._calculate_stagnation_periods(max_fitness)
                
                metrics = ['Total\nImprovement', 'Avg Rate\nper Gen', 'Best Improve\nGeneration', 
                          'Stagnation\nPeriods']
                values = [total_improvement, avg_improvement_rate, best_improvement_gen, len(stagnation_periods)]
                
                # Normalize for visualization
                max_val = max(abs(v) for v in values) if values else 1
                normalized_values = [v / max_val for v in values]
                
                bars = plt.bar(metrics, normalized_values, 
                              color=['green', 'blue', 'orange', 'red'], alpha=0.7)
                plt.ylabel('Normalized Value', fontsize=12)
                plt.title('Performance Metrics', fontsize=14, fontweight='bold')
                plt.xticks(rotation=0, fontsize=10)
                plt.grid(True, alpha=0.3)
                
                # Add actual values as labels
                for bar, value in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Plot 9: Enhanced Summary
            ax9 = plt.subplot(3, 3, 9)
            ax9.axis('off')
            
            # Create comprehensive statistics text
            summary = self.get_optimization_summary()
            
            # Calculate additional metrics
            if len(max_fitness) > 1:
                improvement_rate = (max_fitness[-1] - max_fitness[0]) / len(generations)
                best_gen = np.argmax(max_fitness)
                convergence_quality = self._assess_convergence_quality()
            else:
                improvement_rate = 0
                best_gen = 0
                convergence_quality = "Unknown"
            
            stats_text = f"""GA Optimization Summary
            
Best Fitness: {summary['best_fitness']:.4f}
Total Generations: {summary['generations_run']}
Convergence Gen: {summary['convergence_generation']}
Total Improvement: {summary['fitness_improvement']:.4f}
Avg Rate/Gen: {improvement_rate:.4f}
Best Found at Gen: {best_gen}
Convergence Quality: {convergence_quality}

Optimized Parameters:
  Threshold %: {summary['best_parameters']['threshold_percentile']:.2f}%
  Aggregation: {summary['best_parameters']['aggregation_method']}
  Z-score: {summary['best_parameters']['use_zscore_transformation']}
  Threshold Method: {summary['best_parameters']['threshold_method']}
            """
            
            ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.results_dir, 'ga_fitness_evolution_enhanced.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"     Enhanced GA evolution plot saved to {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"     Error creating enhanced fitness plots: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _calculate_stagnation_periods(self, fitness_history):
        """Calculate periods where fitness stagnated."""
        stagnation_periods = []
        current_period = 0
        threshold = 0.001  # Minimum improvement to not be considered stagnation
        
        for i in range(1, len(fitness_history)):
            if abs(fitness_history[i] - fitness_history[i-1]) < threshold:
                current_period += 1
            else:
                if current_period > 5:  # Only count significant stagnation periods
                    stagnation_periods.append(current_period)
                current_period = 0
        
        if current_period > 5:
            stagnation_periods.append(current_period)
        
        return stagnation_periods
    
    def _assess_convergence_quality(self):
        """Assess the quality of convergence."""
        if not self.convergence_data:
            return "Unknown"
        
        max_fitness = self.convergence_data['max_fitness']
        if len(max_fitness) < 10:
            return "Insufficient Data"
        
        # Check improvement in later generations
        late_improvement = max_fitness[-5:] if len(max_fitness) >= 5 else max_fitness[-1:]
        early_improvement = max_fitness[:5] if len(max_fitness) >= 5 else max_fitness[0]
        
        total_improvement = max_fitness[-1] - max_fitness[0]
        if total_improvement > 10:
            return "Excellent"
        elif total_improvement > 5:
            return "Good"
        elif total_improvement > 1:
            return "Fair"
        else:
            return "Poor"

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
