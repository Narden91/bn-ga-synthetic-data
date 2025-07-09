import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deap import creator, base, tools, algorithms
from typing import Dict, List, Tuple
import random
import warnings
import os
import json
import traceback
from scipy import stats

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class GeneticOptimizer:
    """
    Optimizes anomaly detection parameters using a genetic algorithm.
    
    Uses DEAP-based genetic algorithm to find optimal hyperparameters for
    the AnomalyDetector with multi-objective fitness function.
    """

    def __init__(self, config: Dict):
        """
        Initializes the genetic optimizer and sets up DEAP components.
        
        Args:
            config (Dict): Configuration for the genetic algorithm.
        """
        self.config = config
        self.toolbox = None
        self.best_individual = None
        self.convergence_data = {}
        self.base_results_dir = "results"
        self.execution_results_dir = None
        os.makedirs(self.base_results_dir, exist_ok=True)
        self._setup_deap()
    
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

    def _setup_deap(self):
        """Sets up DEAP genetic algorithm components."""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax, components=None)

        self.toolbox = base.Toolbox()
        self._define_parameters()

        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self._evaluate_fitness)

    def _define_parameters(self):
        """Defines the parameter space for optimization."""
        self.param_bounds = {
            'threshold_percentile': (1.0, 10.0),
            'aggregation_method_idx': (0, 4),
            'use_zscore': (0, 1),
            'threshold_method_idx': (0, 3)
        }
        self.aggregation_methods = ['mean', 'min', 'median', 'weighted', 'sum']
        self.threshold_methods = ['percentile', 'std', 'iqr', 'adaptive']

    def _create_individual(self) -> List:
        """Creates a random individual (a set of parameters)."""
        individual = []
        for param, (min_val, max_val) in self.param_bounds.items():
            if 'idx' in param or param == 'use_zscore':
                individual.append(random.randint(int(min_val), int(max_val)))
            else:
                individual.append(random.uniform(min_val, max_val))
        return individual

    def _decode_individual(self, individual: List) -> Dict:
        """Converts an individual's gene list into a human-readable parameter dictionary."""
        return {
            'threshold_percentile': individual[0],
            'aggregation_method': self.aggregation_methods[int(individual[1])],
            'use_zscore_transformation': bool(individual[2]),
            'threshold_method': self.threshold_methods[int(individual[3])]
        }

    def _crossover(self, ind1: List, ind2: List) -> Tuple[List, List]:
        """Performs crossover on two parent individuals."""
        alpha = 0.5
        for i in range(len(ind1)):
            if random.random() < self.config.get('crossover_rate', 0.8):
                if 'idx' not in list(self.param_bounds.keys())[i] and i != 2:
                    gamma = (1 + 2 * alpha) * random.random() - alpha
                    ind1[i], ind2[i] = (1 - gamma) * ind1[i] + gamma * ind2[i], (1 - gamma) * ind2[i] + gamma * ind1[i]
                else:
                    if random.random() < 0.5:
                        ind1[i], ind2[i] = ind2[i], ind1[i]
        self._enforce_bounds(ind1)
        self._enforce_bounds(ind2)
        return ind1, ind2

    def _mutate(self, individual: List) -> Tuple[List]:
        """Performs mutation on an individual."""
        mutation_rate = self.config.get('mutation_rate', 0.1)
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                param_name = list(self.param_bounds.keys())[i]
                min_val, max_val = self.param_bounds[param_name]
                if 'idx' in param_name or param_name == 'use_zscore':
                    individual[i] = random.randint(int(min_val), int(max_val))
                else:
                    sigma = (max_val - min_val) * 0.1
                    individual[i] += random.gauss(0, sigma)
        self._enforce_bounds(individual)
        return individual,

    def _enforce_bounds(self, individual: List):
        """Ensures that all parameters within an individual are within their defined bounds."""
        for i, (param, (min_val, max_val)) in enumerate(self.param_bounds.items()):
            individual[i] = max(min_val, min(max_val, individual[i]))
            if 'idx' in param or param == 'use_zscore':
                individual[i] = int(round(individual[i]))

    def optimize(self, likelihood_scores: pd.DataFrame, data: pd.DataFrame) -> Dict:
        """
        Runs the genetic algorithm optimization process.
        
        Args:
            likelihood_scores: Likelihood scores from the Bayesian Network.
            data: Original data for validation.
            
        Returns:
            A dictionary containing the best found parameters.
        """
        print("     Starting genetic algorithm optimization...")
        self.likelihood_scores = likelihood_scores
        self.original_data = data
        
        population_size = self.config.get('population_size', 50)
        population = self.toolbox.population(n=population_size)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("median", np.median)
        
        hof = tools.HallOfFame(1)

        try:
            population, logbook = self._run_evolution(
                population,
                self.config.get('crossover_rate', 0.8),
                self.config.get('mutation_rate', 0.1),
                self.config.get('generations', 50),
                stats,
                hof
            )
            
            self.best_individual = hof[0]
            self.convergence_data = {
                'generations': logbook.select('gen'),
                'max_fitness': logbook.select('max'),
                'avg_fitness': logbook.select('avg'),
                'min_fitness': logbook.select('min'),
                'std_fitness': logbook.select('std'),
                'median_fitness': logbook.select('median')
            }
            
            best_params = self._decode_individual(self.best_individual)
            self._save_optimization_results(best_params, hof)
            self._create_fitness_plots()

            print(f"     ✅ GA optimization completed. Best fitness: {self.best_individual.fitness.values[0]:.4f}")
            print(f"     Results saved to '{self.results_dir}/'")
            return best_params
            
        except Exception as e:
            print(f"     ❌ GA optimization failed: {e}")
            return self._decode_individual(self._create_individual()) # Return random default

    def _run_evolution(self, population, cxpb, mutpb, ngen, stats, hof):
        """Custom evolution loop with progress logging."""
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Initial evaluation
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0]
            ind.components = fit[1]

        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        print(f"     Gen 0: Max={record.get('max', 0):.3f}, Avg={record.get('avg', 0):.3f}")

        # Evolution loop
        for gen in range(1, ngen + 1):
            offspring = self.toolbox.select(population, len(population))
            offspring = algorithms.varAnd(offspring, self.toolbox, cxpb, mutpb)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit[0]
                ind.components = fit[1]

            hof.update(offspring)
            population[:] = offspring
            
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            if gen % 10 == 0 or gen == ngen:
                print(f"     Gen {gen}: Max={record.get('max', 0):.3f}, Avg={record.get('avg', 0):.3f}, Std={record.get('std', 0):.3f}")
        
        return population, logbook

    def _evaluate_fitness(self, individual: List) -> Tuple[Tuple[float], Dict]:
        """
        Evaluates the fitness of an individual, returning the score and its components.
        """
        try:
            params = self._decode_individual(individual)
            from .anomaly_detector import AnomalyDetector # Local import to avoid circular dependency
            
            temp_detector = AnomalyDetector(params)
            anomaly_scores, anomaly_indices = temp_detector.detect_anomalies(self.likelihood_scores, verbose=False)
            
            fitness, components = self._calculate_fitness(anomaly_scores, anomaly_indices, params)
            return (fitness,), components
            
        except Exception:
            return (-1000.0,), {} # Return very low fitness for invalid individuals

    def _calculate_fitness(self, anomaly_scores: np.ndarray, 
                           anomaly_indices: np.ndarray, params: Dict) -> Tuple[float, Dict]:
        """
        Refactored fitness calculation to promote balanced, robust solutions.
        
        The fitness score (0-100) is a weighted sum of four key components:
        1. Anomaly Rate Quality (30 pts): Rewards an anomaly rate near a target using a Gaussian curve.
        2. Score Separation Quality (40 pts): Measures separation using Cohen's d, rewarding meaningful
           separation with diminishing returns via a sigmoid function.
        3. Distribution Quality (20 pts): Rewards solutions where non-anomalous scores form a coherent,
           statistically "normal-like" group, assessed via the Shapiro-Wilk test p-value.
        4. Parameter Simplicity Bonus (10 pts): A small nudge favoring stable and interpretable parameters.
        """
        components = {
            'rate_score': 0.0, 'separation_score': 0.0, 
            'distribution_score': 0.0, 'simplicity_bonus': 0.0
        }
        num_samples = len(anomaly_scores)
        num_anomalies = len(anomaly_indices)

        if num_anomalies == 0 or num_anomalies >= num_samples / 2:
            return 0.0, components

        # Component 1: Anomaly Rate Quality (Max 30 points)
        anomaly_rate = (num_anomalies / num_samples) * 100
        target_rate, rate_spread = 4.0, 2.0
        components['rate_score'] = 30 * np.exp(-0.5 * ((anomaly_rate - target_rate) / rate_spread) ** 2)

        normal_indices = np.setdiff1d(np.arange(num_samples), anomaly_indices)
        anomaly_values = anomaly_scores[anomaly_indices]
        normal_values = anomaly_scores[normal_indices]

        # Component 2: Score Separation Quality (Max 40 points)
        if len(normal_values) > 1 and len(anomaly_values) > 1:
            mean_anomaly, std_anomaly = np.mean(anomaly_values), np.std(anomaly_values)
            mean_normal, std_normal = np.mean(normal_values), np.std(normal_values)
            
            pooled_std = np.sqrt(((num_anomalies - 1) * std_anomaly**2 + (len(normal_values) - 1) * std_normal**2) / (num_samples - 2))
            if pooled_std > 1e-6:
                cohens_d = (mean_anomaly - mean_normal) / pooled_std
                k, offset = 4.0, 0.5
                components['separation_score'] = 40 / (1 + np.exp(-k * (cohens_d - offset)))

        # Component 3: Distribution Quality (Max 20 points)
        if len(normal_values) > 3:
            _, p_value = stats.shapiro(normal_values)
            components['distribution_score'] = p_value * 20

        # Component 4: Parameter Simplicity Bonus (Max 10 points)
        bonus = 0
        if params['aggregation_method'] in ['mean', 'median']: bonus += 3
        if params['threshold_method'] == 'percentile': bonus += 2
        if params['use_zscore_transformation']: bonus += 2
        if 2 <= params['threshold_percentile'] <= 8: bonus += 3
        components['simplicity_bonus'] = min(10, bonus)

        total_fitness = sum(components.values()) + random.uniform(-0.01, 0.01)
        return max(0, min(100, total_fitness)), components
    
    def _find_convergence_generation(self) -> int:
        """Finds the generation where the best fitness score plateaus."""
        if len(self.convergence_data['max_fitness']) < 10:
            return len(self.convergence_data['max_fitness'])
        
        max_fitness = self.convergence_data['max_fitness']
        window = 10
        for i in range(window, len(max_fitness)):
            recent_window = max_fitness[i-window:i]
            if (np.max(recent_window) - np.min(recent_window)) < 0.1:
                return i - window
        return len(max_fitness)

    def _assess_convergence_quality(self) -> str:
        """Provides a qualitative assessment of the convergence."""
        if not self.convergence_data: return "Unknown"
        
        max_fitness = self.convergence_data['max_fitness']
        if len(max_fitness) < 20: return "Insufficient Data"

        total_improvement = max_fitness[-1] - max_fitness[0]
        last_quarter_idx = int(len(max_fitness) * 0.75)
        late_improvement = max_fitness[-1] - max_fitness[last_quarter_idx]

        if total_improvement <= 0.1: return "Stalled"
        if late_improvement / total_improvement < 0.05: return "Excellent"
        if late_improvement / total_improvement < 0.15: return "Good"
        return "Fair"

    def _create_fitness_plots(self):
        """Creates and displays the 3 most informative GA optimization plots."""
        if not self.convergence_data:
            print("     No convergence data available for plotting.")
            return

        try:
            plt.style.use('default')
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            gens = self.convergence_data['generations']
            max_f = self.convergence_data['max_fitness']
            avg_f = self.convergence_data['avg_fitness']
            std_f = self.convergence_data['std_fitness']
            
            # Plot 1: Fitness Evolution (Most Important)
            ax1 = axes[0]
            ax1.plot(gens, max_f, 'r-', linewidth=3, label='Best Fitness', marker='o', markersize=4)
            ax1.plot(gens, avg_f, 'b-', linewidth=2, label='Average Fitness', alpha=0.8)
            ax1.fill_between(gens, np.array(avg_f) - np.array(std_f), np.array(avg_f) + np.array(std_f), 
                           alpha=0.2, color='blue', label='±1 Std Dev')
            conv_gen = self._find_convergence_generation()
            ax1.axvline(x=conv_gen, color='red', linestyle='--', linewidth=2, label=f'Convergence (Gen {conv_gen})')
            ax1.set_title('GA Fitness Evolution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness Value (0-100)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 105)

            # Plot 2: Best Individual Fitness Breakdown
            ax2 = axes[1]
            if self.best_individual and hasattr(self.best_individual, 'components'):
                components = self.best_individual.components
                names = ['Rate', 'Separation', 'Distribution', 'Simplicity']
                values = [components.get(k, 0) for k in ['rate_score', 'separation_score', 'distribution_score', 'simplicity_bonus']]
                colors = ['#440154', '#31688e', '#35b779', '#fde725'][:len(names)]
                bars = ax2.bar(names, values, color=colors, alpha=0.8)
                ax2.bar_label(bars, fmt='%.1f')
                ax2.set_title('Best Individual Fitness Breakdown', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Fitness Components')
                ax2.set_ylabel('Component Score')
                ax2.grid(True, axis='y', alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No fitness breakdown available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Best Individual Fitness Breakdown', fontsize=14, fontweight='bold')

            # Plot 3: Optimized Parameters
            ax3 = axes[2]
            best_params = self._decode_individual(self.best_individual)
            param_names = ['threshold_percentile', 'aggregation_method', 'use_zscore_transformation', 'threshold_method']
            param_labels = [p.replace('_', '\n').title() for p in param_names]
            values = [
                best_params['threshold_percentile'],
                self.aggregation_methods.index(best_params['aggregation_method']),
                int(best_params['use_zscore_transformation']),
                self.threshold_methods.index(best_params['threshold_method'])
            ]
            text_values = [
                f"{values[0]:.2f}",
                best_params['aggregation_method'],
                str(bool(values[2])),
                best_params['threshold_method']
            ]
            bars = ax3.bar(param_labels, values, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'], alpha=0.8)
            for bar, text_val in zip(bars, text_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        text_val, ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax3.set_title('Optimized Parameters', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Parameter Value/Index')
            ax3.grid(True, axis='y', alpha=0.3)

            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.results_dir, 'ga_fitness_evolution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"     GA plots saved to {plot_path}")
            
            # Display the plot
            plt.show()

        except Exception as e:
            print(f"     Error creating fitness plots: {e}")
            traceback.print_exc()
    
    def _calculate_stagnation_periods(self, fitness_history, window=5, threshold=0.01):
        """Calculates periods where fitness stagnated."""
        periods = []
        in_stagnation = False
        for i in range(window, len(fitness_history)):
            if np.max(fitness_history[i-window:i]) - np.min(fitness_history[i-window:i]) < threshold:
                if not in_stagnation:
                    periods.append(1)
                    in_stagnation = True
                else:
                    periods[-1] += 1
            else:
                in_stagnation = False
        return periods

    def get_optimization_summary(self) -> Dict:
        """Returns a dictionary summarizing the optimization results."""
        if not self.best_individual:
            return {'status': 'Not optimized'}
        
        return {
            'status': 'Optimized',
            'best_fitness': self.best_individual.fitness.values[0],
            'best_parameters': self._decode_individual(self.best_individual),
            'generations_run': len(self.convergence_data.get('generations', [])),
            'convergence_generation': self._find_convergence_generation()
        }

    def _save_optimization_results(self, best_params: Dict, hof):
        """Saves optimization artifacts like parameters and history to files."""
        try:
            # Save comprehensive JSON summary
            summary_data = self.get_optimization_summary()
            summary_data['convergence_data'] = self._make_json_serializable(self.convergence_data)
            summary_path = os.path.join(self.results_dir, 'ga_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)

            history_path = os.path.join(self.results_dir, 'ga_fitness_history.csv')
            pd.DataFrame(self.convergence_data).to_csv(history_path, index=False)
            
            print(f"     GA results saved to: {summary_path}")
            print(f"     GA fitness history saved to: {history_path}")
            
        except Exception as e:
            print(f"     Error saving GA results: {e}")

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