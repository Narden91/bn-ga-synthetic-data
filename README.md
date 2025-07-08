# Bayesian Network GA-based Synthetic Data Generator

A Python pipeline that learns Bayesian Network structures from unlabeled data and uses a Genetic Algorithm to optimize Conditional Probability Tables (CPTs) for generating high-quality synthetic data.

## Project Structure

```
bn-ga-synthetic-data/
│
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/
│   ├── __init__.py
│   ├── loader.py              # Data loading utilities
│   └── Dati_wallbox_aggregati.csv  # Your data file
│
├── preprocessing/
│   ├── __init__.py
│   └── preprocessing.py       # Data discretization and encoding
│
├── bn_model/
│   ├── __init__.py
│   ├── bn_structure.py        # BN structure learning
│   └── bn_sampler.py          # Synthetic data sampling
│
├── ga_optimizer/
│   ├── __init__.py
│   └── ga_cpt_optimizer.py    # Genetic Algorithm CPT optimization
│
├── utils/
│   ├── __init__.py
│   └── evaluation.py          # Evaluation metrics
│
└── results/                   # Output directory for results
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your data file is in the `data/` directory.

## Usage

### Run the full pipeline:
```bash
python main.py
```

### Run a quick test:
```bash
python main.py --test
```

## Pipeline Overview

1. **Data Loading**: Load CSV data and handle missing values
2. **Preprocessing**: Discretize continuous variables and encode categorical ones
3. **Structure Learning**: Learn Bayesian Network structure using Hill Climb Search with BIC scoring
4. **Parameter Estimation**: Estimate initial CPTs using Maximum Likelihood Estimation
5. **Baseline Evaluation**: Generate and evaluate baseline synthetic data
6. **GA Optimization**: Use Genetic Algorithm to optimize CPT parameters
7. **Final Evaluation**: Generate and evaluate optimized synthetic data
8. **Results Saving**: Save synthetic data and evaluation results

## Key Features

- **Robust Data Preprocessing**: Handles both continuous and categorical variables
- **Structure Learning**: Uses pgmpy's Hill Climb Search with BIC scoring
- **Genetic Algorithm Optimization**: Evolves CPT parameters to improve synthetic data quality
- **Comprehensive Evaluation**: Multiple metrics including KL divergence, mutual information, and statistical tests
- **Fallback Mechanisms**: Handles edge cases and provides fallback options

## Evaluation Metrics

- **Marginal Distributions**: KL divergence and Jensen-Shannon divergence
- **Mutual Information**: Correlation between real and synthetic MI matrices
- **Statistical Tests**: Kolmogorov-Smirnov tests for distribution similarity
- **Fitness Score**: Combined metric for GA optimization

## Configuration

Key parameters can be modified in `main.py`:

- `n_bins`: Number of bins for discretization (default: 3)
- `ga_params`: Genetic Algorithm parameters
  - `n_gen`: Number of generations (default: 30)
  - `pop_size`: Population size (default: 20)
  - `cx_prob`: Crossover probability (default: 0.7)
  - `mut_prob`: Mutation probability (default: 0.3)

## Output Files

The pipeline generates:
- `synthetic_data_baseline_TIMESTAMP.csv`: Baseline synthetic data
- `synthetic_data_optimized_TIMESTAMP.csv`: GA-optimized synthetic data
- `evaluation_results_TIMESTAMP.json`: Detailed evaluation results

## Dependencies

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `pgmpy`: Bayesian Network learning and inference
- `deap`: Genetic Algorithm framework
- `scikit-learn`: Machine learning utilities and metrics

## Troubleshooting

1. **Data Loading Issues**: Ensure your CSV file is properly formatted and in the `data/` directory
2. **Memory Issues**: Reduce `n_samples_synthetic` or GA population size for large datasets
3. **Convergence Issues**: Increase `n_gen` or adjust GA parameters
4. **Structure Learning Failures**: The pipeline includes fallback mechanisms for edge cases

## Advanced Usage

### Custom Data Files
Modify the `data_file` variable in `main.py` to point to your data file.

### Custom Evaluation
Extend the evaluation metrics in `utils/evaluation.py` for domain-specific requirements.

### Different Structure Learning
Modify `bn_model/bn_structure.py` to use different structure learning algorithms (e.g., constraint-based methods).

## Example Output

```
BAYESIAN NETWORK GA-BASED SYNTHETIC DATA GENERATOR
================================================================================

1. Loading data from: data/Dati_wallbox_aggregati.csv
Original data shape: (1000, 5)
Columns: ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']

2. Preprocessing data (discretization with 3 bins)
Discretizing column 'feature1' with 156 unique values into 3 bins
Discretizing column 'feature2' with 89 unique values into 3 bins
...

3. Learning Bayesian Network structure
Learned network with 5 nodes and 4 edges
Edges: [('feature1', 'feature2'), ('feature2', 'feature3'), ...]

...

10. Comparing baseline vs optimized models
Performance Comparison:
  Average KL Divergence: 0.1234 → 0.0987
  Average JS Divergence: 0.0876 → 0.0654
  MI Correlation: 0.7654 → 0.8432
  Similar Distributions: 60.0% → 80.0%

PIPELINE COMPLETED SUCCESSFULLY!
```

## License

This project is open source and available under the MIT License.