"""
Bayesian Network GA-based Synthetic Data Generator
Main entry point for the synthetic data generation pipeline.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Import project modules
from data.loader import load_data, validate_data
from preprocessing.preprocessing import prepare_data_for_bn
from bn_model.bn_structure import learn_bn_structure, estimate_parameters, validate_bn_model, get_model_info
from bn_model.bn_sampler import sample_bn_data, validate_synthetic_data, save_synthetic_data
from ga_optimizer.ga_cpt_optimizer import ga_optimize, evaluate_ga_progress
from utils.evaluation import compute_comprehensive_evaluation, print_evaluation_summary, save_evaluation_results


def main():
    """
    Main pipeline for generating synthetic data using Bayesian Networks and Genetic Algorithm.
    """
    print("="*80)
    print("BAYESIAN NETWORK GA-BASED SYNTHETIC DATA GENERATOR")
    print("="*80)
    
    # Configuration
    data_file = "data/Dati_wallbox_aggregati.csv"
    n_bins = 3  # Number of bins for discretization
    n_samples_synthetic = None  # Will use same as real data
    max_variables = 50  # Maximum number of variables to use (for performance)
    
    # GA parameters
    ga_params = {
        'n_gen': 10,  # Reduced for large datasets
        'pop_size': 10,  # Reduced for large datasets
        'cx_prob': 0.7,  # Crossover probability
        'mut_prob': 0.3,  # Mutation probability
        'verbose': True
    }
    
    try:
        # Step 1: Load and validate data
        print(f"\n1. Loading data from: {data_file}")
        if not os.path.exists(data_file):
            print(f"Error: Data file not found: {data_file}")
            print("Please ensure the CSV file is in the data/ directory")
            return
        
        df_original = load_data(data_file)
        
        if not validate_data(df_original):
            print("Data validation failed. Please check your data.")
            return
        
        # Step 2: Preprocess data
        print(f"\n2. Preprocessing data (discretization with {n_bins} bins)")
        df_processed, preprocessing_info = prepare_data_for_bn(df_original, n_bins=n_bins)
        
        print(f"Preprocessed data shape: {df_processed.shape}")
        
        # Step 2.5: Feature selection for large datasets
        if df_processed.shape[1] > max_variables:
            print(f"\n2.5. Feature selection (reducing from {df_processed.shape[1]} to {max_variables} variables)")
            
            # Remove constant variables (those with only 1 unique value)
            constant_cols = []
            for col in df_processed.columns:
                if df_processed[col].nunique() <= 1:
                    constant_cols.append(col)
            
            if constant_cols:
                print(f"Removing {len(constant_cols)} constant variables")
                df_processed = df_processed.drop(columns=constant_cols)
            
            # If still too many variables, select most informative ones
            if df_processed.shape[1] > max_variables:
                from sklearn.feature_selection import mutual_info_classif
                from sklearn.preprocessing import LabelEncoder
                
                print(f"Selecting top {max_variables} most informative variables")
                
                # Use the first variable as target for feature selection
                target_col = df_processed.columns[0]
                feature_cols = df_processed.columns[1:]
                
                # Calculate mutual information
                mi_scores = mutual_info_classif(df_processed[feature_cols], df_processed[target_col])
                
                # Select top features
                feature_importance = list(zip(feature_cols, mi_scores))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                selected_features = [target_col] + [f[0] for f in feature_importance[:max_variables-1]]
                df_processed = df_processed[selected_features]
                
                print(f"Selected features: {selected_features[:10]}..." if len(selected_features) > 10 else f"Selected features: {selected_features}")
        
        print(f"Final data shape: {df_processed.shape}")
        print(f"Variables: {list(df_processed.columns)}")
        
        # Step 3: Learn Bayesian Network structure
        print(f"\n3. Learning Bayesian Network structure")
        bn_model = learn_bn_structure(df_processed, scoring_method='bic')
        
        # Get model information
        model_info = get_model_info(bn_model)
        print(f"Learned BN structure:")
        print(f"  Nodes: {model_info['n_nodes']}")
        print(f"  Edges: {model_info['n_edges']}")
        print(f"  Edges: {model_info['edges']}")
        
        # Step 4: Estimate initial parameters
        print(f"\n4. Estimating initial CPT parameters")
        bn_model = estimate_parameters(bn_model, df_processed, estimator_type='mle')
        
        # Validate the model
        if not validate_bn_model(bn_model, df_processed):
            print("Model validation failed. Exiting.")
            return
        
        # Step 5: Generate baseline synthetic data
        print(f"\n5. Generating baseline synthetic data")
        n_samples_synthetic = len(df_processed) if n_samples_synthetic is None else n_samples_synthetic
        
        baseline_synthetic = sample_bn_data(bn_model, n_samples_synthetic)
        validate_synthetic_data(baseline_synthetic, df_processed)
        
        # Step 6: Evaluate baseline performance
        print(f"\n6. Evaluating baseline model performance")
        baseline_evaluation = compute_comprehensive_evaluation(df_processed, baseline_synthetic)
        
        print("Baseline Model Performance:")
        print_evaluation_summary(baseline_evaluation)
        
        # Step 7: Optimize CPTs using Genetic Algorithm
        print(f"\n7. Optimizing CPT parameters using Genetic Algorithm")
        print(f"GA Parameters: {ga_params}")
        
        optimized_model, ga_logbook = ga_optimize(bn_model, df_processed, **ga_params)
        
        # Evaluate GA progress
        if ga_logbook:
            ga_progress = evaluate_ga_progress(ga_logbook)
        
        # Step 8: Generate optimized synthetic data
        print(f"\n8. Generating optimized synthetic data")
        optimized_synthetic = sample_bn_data(optimized_model, n_samples_synthetic)
        validate_synthetic_data(optimized_synthetic, df_processed)
        
        # Step 9: Evaluate optimized performance
        print(f"\n9. Evaluating optimized model performance")
        optimized_evaluation = compute_comprehensive_evaluation(df_processed, optimized_synthetic)
        
        print("Optimized Model Performance:")
        print_evaluation_summary(optimized_evaluation)
        
        # Step 10: Compare baseline vs optimized
        print(f"\n10. Comparing baseline vs optimized models")
        
        baseline_fitness = baseline_evaluation['summary_metrics']
        optimized_fitness = optimized_evaluation['summary_metrics']
        
        print("Performance Comparison:")
        print(f"  Average KL Divergence: {baseline_fitness['average_kl_divergence']:.4f} → {optimized_fitness['average_kl_divergence']:.4f}")
        print(f"  Average JS Divergence: {baseline_fitness['average_js_divergence']:.4f} → {optimized_fitness['average_js_divergence']:.4f}")
        print(f"  MI Correlation: {baseline_fitness['mutual_info_correlation']:.4f} → {optimized_fitness['mutual_info_correlation']:.4f}")
        print(f"  Similar Distributions: {baseline_fitness['similar_distributions_ratio']:.1%} → {optimized_fitness['similar_distributions_ratio']:.1%}")
        
        # Step 11: Save results
        print(f"\n11. Saving results")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save synthetic data
        baseline_file = f"synthetic_data_baseline_{timestamp}.csv"
        optimized_file = f"synthetic_data_optimized_{timestamp}.csv"
        
        save_synthetic_data(baseline_synthetic, baseline_file, include_timestamp=False)
        save_synthetic_data(optimized_synthetic, optimized_file, include_timestamp=False)
        
        # Save evaluation results
        evaluation_file = f"evaluation_results_{timestamp}.json"
        combined_evaluation = {
            'baseline': baseline_evaluation,
            'optimized': optimized_evaluation,
            'preprocessing_info': preprocessing_info,
            'model_info': model_info,
            'ga_params': ga_params
        }
        
        save_evaluation_results(combined_evaluation, evaluation_file)
        
        print(f"\nResults saved:")
        print(f"  Baseline synthetic data: {baseline_file}")
        print(f"  Optimized synthetic data: {optimized_file}")
        print(f"  Evaluation results: {evaluation_file}")
        
        print(f"\n{'='*80}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\nError in main pipeline: {e}")
        import traceback
        traceback.print_exc()


def run_quick_test():
    """
    Run a quick test with minimal parameters for debugging.
    """
    print("Running quick test...")
    
    # Generate small synthetic dataset for testing
    np.random.seed(42)
    test_data = pd.DataFrame({
        'A': np.random.randint(0, 3, 100),
        'B': np.random.randint(0, 2, 100),
        'C': np.random.randint(0, 4, 100)
    })
    
    # Make B somewhat dependent on A
    test_data.loc[test_data['A'] == 0, 'B'] = np.random.choice([0, 1], 
                                                               size=sum(test_data['A'] == 0), 
                                                               p=[0.8, 0.2])
    
    print(f"Test data shape: {test_data.shape}")
    
    # Quick BN learning
    bn_model = learn_bn_structure(test_data)
    bn_model = estimate_parameters(bn_model, test_data)
    
    # Generate synthetic data
    synthetic = sample_bn_data(bn_model, 100)
    
    # Quick evaluation
    evaluation = compute_comprehensive_evaluation(test_data, synthetic)
    print_evaluation_summary(evaluation)
    
    print("Quick test completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_quick_test()
    else:
        main()