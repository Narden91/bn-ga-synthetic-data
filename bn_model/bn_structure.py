"""
Bayesian Network structure learning and parameter estimation.
"""

import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import warnings


def learn_bn_structure(df, scoring_method='bic', max_indegree=3, timeout_seconds=300):
    """
    Learn Bayesian Network structure from data using score-based approach.
    
    Args:
        df (pd.DataFrame): Discretized input data
        scoring_method (str): Scoring method ('bic' or 'k2')
        max_indegree (int): Maximum number of parents per node
        timeout_seconds (int): Maximum time to spend on structure learning
        
    Returns:
        pgmpy.models.BayesianNetwork: Learned BN structure
    """
    print(f"Learning BN structure from data with shape {df.shape}")
    print(f"Using {scoring_method.upper()} scoring method with timeout {timeout_seconds}s")
    
    # For large datasets, use simpler approach
    if df.shape[1] > 20 or df.shape[0] * df.shape[1] > 50000:
        print("Large dataset detected, using simplified structure learning")
        return learn_simple_bn_structure(df)
    
    # Choose scoring method and initialize Hill Climb Search with scoring
    try:
        if scoring_method.lower() == 'bic':
            hc = HillClimbSearch(df, scoring_method=BicScore(df))
        elif scoring_method.lower() == 'k2':
            hc = HillClimbSearch(df, scoring_method=K2Score(df))
        else:
            raise ValueError(f"Unsupported scoring method: {scoring_method}")
    except TypeError:
        # Fallback for different pgmpy API versions
        hc = HillClimbSearch(df)
    
    # Learn structure with constraints
    try:
        import time
        start_time = time.time()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Try with max_indegree parameter
            try:
                best_model = hc.estimate(max_indegree=max_indegree)
                
                # Check if we exceeded timeout (simple check)
                if time.time() - start_time > timeout_seconds:
                    print(f"Structure learning took too long, using simple structure")
                    best_model = learn_simple_bn_structure(df)
                    
            except TypeError:
                # Fallback without max_indegree if not supported
                best_model = hc.estimate()
                
    except Exception as e:
        print(f"Structure learning failed: {e}")
        print("Falling back to simple structure")
        best_model = learn_simple_bn_structure(df)
    
    # Convert DAG to BayesianNetwork if needed
    if hasattr(best_model, 'edges') and not hasattr(best_model, 'add_cpds'):
        # Convert DAG to BayesianNetwork
        bn_model = BayesianNetwork()
        bn_model.add_nodes_from(best_model.nodes())
        bn_model.add_edges_from(best_model.edges())
        best_model = bn_model
    
    print(f"Learned network with {len(best_model.nodes())} nodes and {len(best_model.edges())} edges")
    print(f"Edges: {list(best_model.edges())}")
    
    return best_model


def learn_simple_bn_structure(df):
    """
    Learn a simple Bayesian Network structure for large datasets.
    Creates a naive structure with limited connections.
    
    Args:
        df (pd.DataFrame): Discretized input data
        
    Returns:
        pgmpy.models.BayesianNetwork: Simple BN structure
    """
    print("Creating simple BN structure for large dataset")
    
    columns = list(df.columns)
    bn_model = BayesianNetwork()
    bn_model.add_nodes_from(columns)
    
    # Add some edges based on correlation (but limit to avoid complexity)
    max_edges = min(len(columns), 10)  # Limit number of edges
    
    # Calculate correlations and add strongest ones
    corr_matrix = df.corr().abs()
    
    # Get top correlations (excluding diagonal)
    correlations = []
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            corr_val = corr_matrix.iloc[i, j]
            if not np.isnan(corr_val):
                correlations.append((corr_val, columns[i], columns[j]))
    
    # Sort by correlation strength and add top edges
    correlations.sort(reverse=True)
    
    edges_added = 0
    for corr_val, node1, node2 in correlations[:max_edges]:
        if edges_added >= max_edges:
            break
        try:
            bn_model.add_edge(node1, node2)
            edges_added += 1
        except:
            # Skip if edge creates cycle
            continue
    
    print(f"Created simple structure with {edges_added} edges")
    return bn_model


def estimate_parameters(model, data, estimator_type='mle', prior_type='BDeu'):
    """
    Estimate parameters (CPTs) for the Bayesian Network.
    
    Args:
        model (BayesianNetwork): BN structure
        data (pd.DataFrame): Training data
        estimator_type (str): 'mle' or 'bayes'
        prior_type (str): Prior type for Bayesian estimation
        
    Returns:
        BayesianNetwork: BN with estimated parameters
    """
    print(f"Estimating parameters using {estimator_type.upper()} estimation")
    
    try:
        if estimator_type.lower() == 'mle':
            model.fit(data, estimator=MaximumLikelihoodEstimator)
        elif estimator_type.lower() == 'bayes':
            model.fit(data, estimator=BayesianEstimator, prior_type=prior_type)
        else:
            raise ValueError(f"Unsupported estimator type: {estimator_type}")
            
        print("Parameter estimation completed successfully")
        
    except Exception as e:
        print(f"Error in parameter estimation: {e}")
        print("Creating uniform CPTs as fallback")
        model = create_uniform_cpts(model, data)
    
    return model


def create_uniform_cpts(model, data):
    """
    Create uniform CPTs for a Bayesian Network as fallback.
    
    Args:
        model (BayesianNetwork): BN structure
        data (pd.DataFrame): Training data for cardinality information
        
    Returns:
        BayesianNetwork: BN with uniform CPTs
    """
    print("Creating uniform CPTs...")
    
    for node in model.nodes():
        parents = list(model.predecessors(node))
        
        # Get cardinalities
        node_cardinality = data[node].nunique()
        parent_cardinalities = [data[parent].nunique() for parent in parents]
        
        if not parents:
            # No parents - create marginal CPT
            prob_values = np.ones(node_cardinality) / node_cardinality
            cpd = TabularCPD(
                variable=node,
                variable_card=node_cardinality,
                values=prob_values.reshape(-1, 1)
            )
        else:
            # Has parents - create conditional CPT
            evidence_card = np.prod(parent_cardinalities)
            prob_values = np.ones((node_cardinality, evidence_card)) / node_cardinality
            
            cpd = TabularCPD(
                variable=node,
                variable_card=node_cardinality,
                values=prob_values,
                evidence=parents,
                evidence_card=parent_cardinalities
            )
        
        model.add_cpds(cpd)
    
    # Validate the model
    if model.check_model():
        print("Uniform CPTs created successfully")
    else:
        print("Warning: Model validation failed")
    
    return model


def get_model_info(model):
    """
    Get information about the learned Bayesian Network.
    
    Args:
        model (BayesianNetwork): Trained BN model
        
    Returns:
        dict: Model information
    """
    info = {
        'nodes': list(model.nodes()),
        'edges': list(model.edges()),
        'n_nodes': len(model.nodes()),
        'n_edges': len(model.edges()),
        'cpds': {}
    }
    
    # Get CPD information
    for cpd in model.get_cpds():
        node = cpd.variable
        parents = cpd.variables[1:] if len(cpd.variables) > 1 else []
        info['cpds'][node] = {
            'parents': parents,
            'cardinality': cpd.cardinality[0],
            'n_parameters': cpd.values.size
        }
    
    return info


def validate_bn_model(model, data):
    """
    Validate the Bayesian Network model.
    
    Args:
        model (BayesianNetwork): BN model to validate
        data (pd.DataFrame): Training data
        
    Returns:
        bool: True if model is valid
    """
    try:
        # Check model consistency
        if not model.check_model():
            print("Error: Model consistency check failed")
            return False
        
        # Check if all data columns are in the model
        missing_nodes = set(data.columns) - set(model.nodes())
        if missing_nodes:
            print(f"Error: Missing nodes in model: {missing_nodes}")
            return False
        
        # Check CPDs
        if not model.get_cpds():
            print("Error: No CPDs found in model")
            return False
        
        print("Model validation passed")
        return True
        
    except Exception as e:
        print(f"Error during model validation: {e}")
        return False
