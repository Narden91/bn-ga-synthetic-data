"""
Genetic Algorithm for optimizing Conditional Probability Tables (CPTs) in Bayesian Networks.
"""

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random
from copy import deepcopy
from pgmpy.factors.discrete import TabularCPD
from bn_model.bn_sampler import sample_bn_data
from utils.evaluation import compute_fitness_score


# Global variables for GA
model_global = None
real_data_global = None


def flatten_cpts(model):
    """
    Flatten all CPTs in the model into a single parameter vector.
    
    Args:
        model (BayesianNetwork): BN model
        
    Returns:
        tuple: (flattened_params, cpt_info)
    """
    flattened_params = []
    cpt_info = []
    
    for cpd in model.get_cpds():
        # Store CPD information for reconstruction
        cpt_data = {
            'variable': cpd.variable,
            'variable_card': cpd.cardinality[0],
            'evidence': cpd.variables[1:] if len(cpd.variables) > 1 else [],
            'evidence_card': cpd.cardinality[1:] if len(cpd.cardinality) > 1 else [],
            'shape': cpd.values.shape,
            'start_idx': len(flattened_params),
            'n_params': cpd.values.size
        }
        
        # Flatten CPD values
        flattened_params.extend(cpd.values.flatten())
        cpt_info.append(cpt_data)
    
    return np.array(flattened_params), cpt_info


def reconstruct_cpts(flattened_params, cpt_info, model):
    """
    Reconstruct CPTs from flattened parameters and update model.
    
    Args:
        flattened_params (np.array): Flattened parameter vector
        cpt_info (list): CPT reconstruction information
        model (BayesianNetwork): BN model to update
        
    Returns:
        BayesianNetwork: Updated model
    """
    new_model = deepcopy(model)
    new_model.remove_cpds(*new_model.get_cpds())
    
    for cpt_data in cpt_info:
        # Extract parameters for this CPT
        start_idx = cpt_data['start_idx']
        end_idx = start_idx + cpt_data['n_params']
        params = flattened_params[start_idx:end_idx]
        
        # Reshape to original CPT shape
        values = params.reshape(cpt_data['shape'])
        
        # Normalize to ensure probabilities sum to 1
        values = normalize_cpt_values(values)
        
        # Create new CPD
        if cpt_data['evidence']:
            cpd = TabularCPD(
                variable=cpt_data['variable'],
                variable_card=cpt_data['variable_card'],
                values=values,
                evidence=cpt_data['evidence'],
                evidence_card=cpt_data['evidence_card']
            )
        else:
            cpd = TabularCPD(
                variable=cpt_data['variable'],
                variable_card=cpt_data['variable_card'],
                values=values.reshape(-1, 1)
            )
        
        new_model.add_cpds(cpd)
    
    return new_model


def normalize_cpt_values(values):
    """
    Normalize CPT values to ensure they sum to 1 along the first axis.
    
    Args:
        values (np.array): CPT values
        
    Returns:
        np.array: Normalized values
    """
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    values = np.maximum(values, epsilon)
    
    # Sum along first axis (variable states)
    sums = np.sum(values, axis=0, keepdims=True)
    sums = np.maximum(sums, epsilon)
    
    return values / sums


def fitness_function(individual):
    """
    Fitness function for genetic algorithm.
    Evaluates how well the synthetic data matches the real data.
    
    Args:
        individual (list): Flattened CPT parameters
        
    Returns:
        tuple: Fitness score (negative loss for maximization)
    """
    global model_global, real_data_global
    
    try:
        # Reconstruct model with new parameters
        flattened_params = np.array(individual)
        cpt_info = getattr(fitness_function, 'cpt_info', None)
        
        if cpt_info is None:
            return (float('-inf'),)  # Invalid fitness
        
        updated_model = reconstruct_cpts(flattened_params, cpt_info, model_global)
        
        # Generate synthetic data
        n_samples = min(len(real_data_global), 1000)  # Limit samples for speed
        synthetic_data = sample_bn_data(updated_model, n_samples)
        
        # Compute fitness score
        fitness_score = compute_fitness_score(real_data_global, synthetic_data)
        
        return (fitness_score,)
        
    except Exception as e:
        # Return very poor fitness for invalid individuals
        return (float('-inf'),)


def create_individual(cpt_info):
    """
    Create a random individual (parameter vector).
    
    Args:
        cpt_info (list): CPT information
        
    Returns:
        list: Random parameter vector
    """
    individual = []
    
    for cpt_data in cpt_info:
        # Generate random parameters for this CPT
        n_params = cpt_data['n_params']
        random_params = np.random.random(n_params)
        individual.extend(random_params)
    
    return individual


def mutate_individual(individual, indpb=0.1, sigma=0.1):
    """
    Mutate an individual by adding Gaussian noise.
    
    Args:
        individual (list): Individual to mutate
        indpb (float): Probability of mutating each parameter
        sigma (float): Standard deviation of mutation
        
    Returns:
        tuple: Mutated individual
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] += random.gauss(0, sigma)
            individual[i] = max(0.001, individual[i])  # Ensure positive
    
    return (individual,)


def crossover_individuals(ind1, ind2, alpha=0.5):
    """
    Crossover two individuals using blend crossover.
    
    Args:
        ind1, ind2 (list): Parent individuals
        alpha (float): Crossover parameter
        
    Returns:
        tuple: Two offspring individuals
    """
    for i in range(len(ind1)):
        if random.random() < 0.5:
            # Blend crossover
            gamma = (1 + 2 * alpha) * random.random() - alpha
            ind1[i], ind2[i] = ind1[i] + gamma * (ind2[i] - ind1[i]), ind2[i] + gamma * (ind1[i] - ind2[i])
            
            # Ensure positive values
            ind1[i] = max(0.001, ind1[i])
            ind2[i] = max(0.001, ind2[i])
    
    return ind1, ind2


def ga_optimize(bn_model, real_data, n_gen=50, pop_size=30, cx_prob=0.7, mut_prob=0.3, verbose=True):
    """
    Optimize CPT parameters using Genetic Algorithm.
    
    Args:
        bn_model (BayesianNetwork): Initial BN model
        real_data (pd.DataFrame): Real training data
        n_gen (int): Number of generations
        pop_size (int): Population size
        cx_prob (float): Crossover probability
        mut_prob (float): Mutation probability
        verbose (bool): Print progress
        
    Returns:
        BayesianNetwork: Optimized BN model
    """
    global model_global, real_data_global
    
    print(f"Starting GA optimization with {pop_size} individuals for {n_gen} generations")
    
    # Set global variables for fitness function
    model_global = deepcopy(bn_model)
    real_data_global = real_data.copy()
    
    # Flatten initial CPTs
    initial_params, cpt_info = flatten_cpts(bn_model)
    fitness_function.cpt_info = cpt_info  # Store for fitness function
    
    # Setup DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Register functions
    toolbox.register("individual", lambda: creator.Individual(create_individual(cpt_info)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual, indpb=0.1, sigma=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    population = toolbox.population(n=pop_size)
    
    # Add the original model as one individual
    population[0] = creator.Individual(initial_params.tolist())
    
    # Track statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Hall of fame to keep best individuals
    hof = tools.HallOfFame(1)
    
    # Run the algorithm
    try:
        population, logbook = algorithms.eaSimple(
            population, toolbox, cx_prob, mut_prob, n_gen,
            stats=stats, halloffame=hof, verbose=verbose
        )
        
        # Get best individual
        best_individual = hof[0]
        best_fitness = best_individual.fitness.values[0]
        
        print(f"GA optimization completed. Best fitness: {best_fitness:.4f}")
        
        # Reconstruct best model
        best_model = reconstruct_cpts(np.array(best_individual), cpt_info, bn_model)
        
        return best_model, logbook
        
    except Exception as e:
        print(f"Error during GA optimization: {e}")
        print("Returning original model")
        return bn_model, None


def evaluate_ga_progress(logbook):
    """
    Evaluate and display GA optimization progress.
    
    Args:
        logbook: DEAP logbook from GA run
        
    Returns:
        dict: Progress statistics
    """
    if logbook is None:
        return None
    
    generations = [record['gen'] for record in logbook]
    avg_fitness = [record['avg'] for record in logbook]
    max_fitness = [record['max'] for record in logbook]
    
    progress_stats = {
        'generations': generations,
        'avg_fitness': avg_fitness,
        'max_fitness': max_fitness,
        'final_improvement': max_fitness[-1] - max_fitness[0] if len(max_fitness) > 1 else 0,
        'convergence_gen': None
    }
    
    # Find convergence point (when improvement becomes small)
    if len(max_fitness) > 5:
        for i in range(5, len(max_fitness)):
            recent_improvement = max_fitness[i] - max_fitness[i-5]
            if recent_improvement < 0.001:  # Small improvement threshold
                progress_stats['convergence_gen'] = i
                break
    
    print("GA Optimization Progress:")
    print(f"  Initial fitness: {max_fitness[0]:.4f}")
    print(f"  Final fitness: {max_fitness[-1]:.4f}")
    print(f"  Total improvement: {progress_stats['final_improvement']:.4f}")
    if progress_stats['convergence_gen']:
        print(f"  Converged at generation: {progress_stats['convergence_gen']}")
    
    return progress_stats
