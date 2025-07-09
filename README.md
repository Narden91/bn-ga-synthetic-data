# Bayesian Network Anomaly Detection System

A sophisticated scalable anomaly detection system that combines Bayesian Networks with evolutionary optimization algorithms (Genetic Algorithm and CMA-ES) to detect anomalies in high-dimensional time series data, specifically designed for electrical vehicle charging station (wallbox) data analysis.

## 🚀 Project Overview

This system addresses the **"curse of dimensionality"** problem in Bayesian Network learning by implementing a **feature grouping strategy** that divides large feature sets into manageable groups, learns separate Bayesian Networks for each group, and then aggregates the likelihood scores for final anomaly detection.

### Key Innovation
The core technical innovation lies in the scalable approach to Bayesian Network learning through intelligent feature grouping, making it computationally feasible to apply probabilistic anomaly detection to high-dimensional datasets.

## 📊 Input Data Specifications

- **Input File**: `Dati_wallbox_aggregati.csv`
- **Original Dimensions**: 2,400 samples × 230 features
- **Domain**: Electrical vehicle charging station operational data
- **Data Type**: Mixed numerical/categorical time series data
- **Post-Processing**: 2,400 samples × 136 features (after feature selection and encoding)

## 🏗️ System Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   DataLoader    │───▶│ DataPreprocessor │───▶│ FeatureGrouper  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ResultVisualizer │◀───│ AnomalyDetector  │◀───│BayesianNetworkLearner│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                    ┌─────────────────────┐
                    │EvolutionaryOptimizer│
                    │  (GA & CMA-ES)      │
                    └─────────────────────┘
```

## 🔧 Technical Pipeline

### 1. Data Loading (`DataLoader`)

**Purpose**: Robust data ingestion with comprehensive validation

**Key Features**:
- File existence validation with error handling
- Data integrity checks (≥10 rows, ≥2 columns)
- Memory usage monitoring and reporting
- Missing value statistics computation
- Data type profiling (numeric vs categorical)

**Technical Rationale**: Ensures data quality before expensive Bayesian Network computations.

### 2. Data Preprocessing (`DataPreprocessor`)

**Purpose**: Transform raw data into Bayesian Network-compatible format

#### Stage 1: Missing Value Handling
- **Column Dropping**: Features with >50% missing values (configurable)
- **Numeric Imputation**: Median strategy (robust to outliers)
- **Categorical Imputation**: Mode strategy
- **Rationale**: Median imputation is robust for skewed distributions common in operational data

#### Stage 2: Categorical Encoding
- **Label Encoding** for categorical variables
- **Automatic type detection** and conversion
- **Error handling** with problematic column removal
- **Rationale**: Preserves ordinality while being compatible with discretization

#### Stage 3: Feature Scaling
- **StandardScaler** (z-score normalization)
- **Rationale**: Ensures equal contribution to correlation-based grouping

### 3. Feature Grouping (`FeatureGrouper`)

**Purpose**: Divide features into computationally manageable groups for scalable BN learning

#### Primary Strategy: Correlation-Based Grouping
Multi-level fallback approach:
1. **Hierarchical Clustering** on correlation distance matrix
2. **K-Means Clustering** on correlation features
3. **Graph-based clustering** using correlation thresholds
4. **Greedy correlation grouping** as final fallback

**Technical Implementation**:
- **Distance Metric**: `1 - |correlation|` (correlation distance)
- **Linkage Method**: Complete linkage for compact clusters
- **Cluster Cutting**: Dynamic threshold to achieve target group sizes

#### Alternative Strategies:
- **Random Grouping**: Baseline comparison
- **Variance-based Grouping**: Groups by feature variance
- **Domain-based Grouping**: Manual feature categorization

**Configuration**: Group size = 10-15 features (optimal for BN learning complexity)

**Technical Rationale**: Correlated features likely share similar probabilistic dependencies, making them suitable for joint Bayesian Network modeling.

### 4. Bayesian Network Learning (`BayesianNetworkLearner`)

**Purpose**: Learn probabilistic models for each feature group
**Framework**: Uses pgmpy library for Bayesian Network operations

#### Data Discretization
- **Method**: K-Bins discretization with uniform strategy
- **Bins**: 3-5 bins (configurable)
- **Rationale**: BNs require discrete variables; fewer bins ensure sufficient samples per bin for reliable probability estimation

#### Structure Learning Algorithms

**Primary**: **Independence Model (Naive Bayes)**
- **Assumption**: Features within groups are conditionally independent
- **Rationale**: Computationally efficient, robust with limited data, provides baseline probabilistic model

**Alternative Algorithms**:
- **Hill Climbing**: Structure optimization with BIC scoring
- **PC Algorithm**: Constraint-based structure learning
- **Rationale**: More complex structures when computational resources allow

#### Parameter Learning
- **Maximum Likelihood Estimation** for parameter fitting
- **Laplace Smoothing** to handle zero probabilities
- **Rationale**: MLE provides unbiased parameter estimates with sufficient data

### 5. Likelihood Computation

**Purpose**: Compute log-likelihood scores for each sample under each Bayesian Network

**Process**:
1. Data discretization using fitted discretizers
2. Log-likelihood computation for each sample under each BN
3. Missing value handling during inference
4. Matrix assembly: Samples × Groups likelihood matrix

**Technical Rationale**: Log-likelihoods prevent numerical underflow and enable additive combination.

### 6. Anomaly Detection (`AnomalyDetector`)

**Purpose**: Aggregate likelihood scores and identify anomalies

#### Score Aggregation Methods
- **Mean**: Average likelihood across groups
- **Min**: Most pessimistic (lowest) likelihood
- **Weighted**: Importance-weighted combination
- **Median**: Robust central tendency

#### Anomaly Score Computation
**Transformation**: Negative log-likelihood → Anomaly score
**Normalization**:
- **Z-score Transformation**: `(score - mean) / std`
- **Rank Transformation**: Percentile-based scoring

#### Threshold Determination
- **Percentile-based**: Top X% as anomalies (default: 5%)
- **Standard Deviation**: Mean + k×std threshold
- **IQR-based**: Interquartile range outlier detection

### 7. Evolutionary Optimization

#### Genetic Algorithm (`GeneticOptimizer`)
**Framework**: DEAP (Distributed Evolutionary Algorithms in Python)

**Parameter Space**:
```python
{
    'threshold_percentile': (1.0, 10.0),
    'aggregation_method': ['mean', 'min', 'median', 'weighted', 'sum'],
    'use_zscore_transformation': [True, False],
    'threshold_method': ['percentile', 'std', 'iqr', 'adaptive']
}
```

**Genetic Operators**:
- **Selection**: Tournament selection (size=3)
- **Crossover**: Simulated binary crossover for continuous variables
- **Mutation**: Gaussian mutation with adaptive sigma
- **Population**: 50-100 individuals
- **Generations**: 100-150

**Fitness Function**: Multi-objective components:
1. Anomaly separation: Distance between normal and anomaly score distributions
2. Statistical significance: T-test p-value between groups
3. Score spread: Variance in anomaly scores
4. Threshold stability: Robustness of threshold choice

#### CMA-ES Optimizer (`CMAESOptimizer`)
**Framework**: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

**Technical Advantages**:
- **Self-adaptive**: Automatically adjusts step sizes and search directions
- **Continuous optimization**: Better for real-valued parameters
- **Robust convergence**: Less prone to local optima

**Configuration**:
- **Initial Sigma**: 0.8 (high exploration)
- **Population Size**: Auto-determined by algorithm
- **Generations**: 150

### 8. Results Visualization (`ResultVisualizer`)

**Purpose**: Comprehensive result analysis and presentation

**Visualization Components**:
1. **Anomaly Score Distribution**: Histograms and box plots comparing normal vs anomaly distributions
2. **Likelihood Heatmap**: Feature group contribution visualization
3. **Anomaly Timeline**: Temporal pattern analysis
4. **Feature Group Contributions**: Relative importance analysis

**Output Format**: High-resolution PNG files (300 DPI) with publication-quality formatting

## 📈 Performance Results

Based on execution results:
- **Original Data**: 2,400 samples × 230 features
- **After Preprocessing**: 136 features retained
- **Feature Groups**: 14 groups created
- **Anomalies Detected**: 93 samples (3.875% anomaly rate)
- **Optimization Algorithm**: CMA-ES achieved superior performance
- **Best Parameters**: 
  - Threshold percentile: 3.85%
  - Aggregation: Weighted
  - Z-score: Disabled
  - Fitness: 81.36

## 🔍 Technical Strengths

### Scalability Solutions
- **Feature Grouping**: Reduces BN complexity from O(2^n) to O(k×2^(n/k))
- **Parallel BN Learning**: Independent group processing
- **Memory-Efficient**: Streaming likelihood computation

### Robustness Features
- **Multi-level Fallbacks**: Each component has multiple implementation strategies
- **Parameter Validation**: Extensive bounds checking and error handling
- **Missing Data Handling**: Comprehensive imputation strategies

### Configurability
- **Modular Design**: Each component independently configurable
- **Multiple Algorithms**: Choice between GA and CMA-ES optimization
- **Flexible Aggregation**: Multiple score combination methods

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
pgmpy>=0.1.15
deap>=1.3.1
cma>=3.1.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🚀 Usage

### Basic Usage
```python
from bayesian_anomaly_detector import BayesianAnomalyDetector

# Initialize detector
detector = BayesianAnomalyDetector()

# Load and process data
data = detector.load_data('Dati_wallbox_aggregati.csv')
processed_data = detector.preprocess_data(data)

# Create feature groups
feature_groups = detector.create_feature_groups(processed_data)

# Learn Bayesian Networks
networks = detector.learn_bayesian_networks(processed_data, feature_groups)

# Detect anomalies
anomaly_scores, anomaly_labels = detector.detect_anomalies(processed_data, networks, feature_groups)

# Visualize results
detector.visualize_results(anomaly_scores, anomaly_labels)
```

### Advanced Usage with Optimization
```python
from bayesian_anomaly_detector import BayesianAnomalyDetector
from evolutionary_optimizer import CMAESOptimizer

# Initialize components
detector = BayesianAnomalyDetector()
optimizer = CMAESOptimizer()

# Load and process data
data = detector.load_data('Dati_wallbox_aggregati.csv')
processed_data = detector.preprocess_data(data)

# Optimize parameters
best_params = optimizer.optimize(processed_data, generations=150)

# Run detection with optimized parameters
anomaly_scores, anomaly_labels = detector.detect_anomalies(
    processed_data, networks, feature_groups, params=best_params
)
```

## 📊 Output Files

### Quantitative Results
- **Anomaly Scores**: Continuous anomaly likelihood for each sample
- **Binary Classification**: Normal/anomaly labels
- **Confidence Metrics**: Statistical significance measures
- **Optimization Convergence**: Algorithm performance metrics

### Generated Artifacts
- **CSV Results**: Detailed anomaly scores and classifications
- **Visualizations**: Publication-ready plots and charts
- **Configuration**: Complete parameter settings for reproducibility
- **Summary Reports**: Human-readable analysis summaries
- **Optimization Logs**: Algorithm convergence and performance data

## 🔬 Theoretical Foundation

This system is grounded in several key theoretical principles:

1. **Probabilistic Anomaly Detection**: Uses Bayesian principles to model normal data distribution and detect deviations
2. **Divide-and-Conquer**: Addresses computational complexity through intelligent feature partitioning
3. **Evolutionary Optimization**: Applies metaheuristic optimization to find optimal detection parameters
4. **Multi-objective Optimization**: Balances multiple anomaly detection quality metrics simultaneously

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models: Principles and Techniques
- Scutari, M. (2010). Learning Bayesian Networks with the bnlearn R Package
- Hansen, N. (2006). The CMA Evolution Strategy: A Comparing Review
- Fortin, F. A., et al. (2012). DEAP: Evolutionary algorithms made easy

## 🙋‍♂️ Support

For questions, issues, or contributions, please open an issue on the GitHub repository or contact the development team.

---

*This system represents a sophisticated approach to scalable anomaly detection that maintains theoretical rigor while providing practical applicability to real-world high-dimensional datasets.*