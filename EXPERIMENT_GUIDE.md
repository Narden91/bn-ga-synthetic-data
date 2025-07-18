# Bayesian Anomaly Detection System - Experiment Guide

This document provides a comprehensive guide to all available experiments in the Bayesian Anomaly Detection System.

## How to Use

To run an experiment, edit the `experiment_to_run` variable in `main.py` and set it to the desired experiment number (1-9), then run:

```bash
python main.py
```

## Available Experiments

### 1. Default Configuration
- **File**: Built-in configuration in main.py
- **Description**: Built-in optimized settings with balanced parameters
- **Optimizer**: Genetic Algorithm
- **Best for**: General purpose, first-time users
- **Features**: Balanced detection with good performance

### 2. High-Performance GA
- **File**: `experiments/experiment_1.yaml`
- **Description**: Thorough exploration with enhanced genetic algorithm
- **Optimizer**: Genetic Algorithm
- **Best for**: When you need comprehensive parameter exploration
- **Features**: Enhanced GA with thorough exploration parameters

### 3. CMA-ES Optimization
- **File**: `experiments/experiment_2_cmaes.yaml`
- **Description**: Advanced parameter tuning with CMA-ES algorithm
- **Optimizer**: CMA-ES
- **Best for**: Continuous parameter optimization, research purposes
- **Features**: Advanced CMA-ES optimization algorithm

### 4. Balanced Detection
- **File**: `experiments/experiment_weighted_balanced.yaml`
- **Description**: Equal focus on detection quality and robustness
- **Optimizer**: CMA-ES
- **Best for**: Production environments, balanced approach
- **Features**: Well-balanced fitness weights for quality and robustness

### 5. Conservative Detection
- **File**: `experiments/experiment_weighted_conservative.yaml`
- **Description**: Minimize false positives, high separation quality
- **Optimizer**: CMA-ES
- **Best for**: When false positives are costly
- **Features**: Lower detection rate, higher precision, strict quality control

### 6. Aggressive Detection
- **File**: `experiments/experiment_weighted_aggressive.yaml`
- **Description**: Catch more anomalies with parameter exploration
- **Optimizer**: Genetic Algorithm
- **Best for**: When missing anomalies is costly
- **Features**: Higher detection rate, more exploration, catch edge cases

### 7. Custom Weights
- **File**: `experiments/experiment_custom_weights.yaml`
- **Description**: Experimental fitness weight configurations
- **Optimizer**: Genetic Algorithm
- **Best for**: Experimental purposes, custom scenarios
- **Features**: Experimental weight combinations for research

### 8. Standard Aggressive
- **File**: `experiments/experiment_aggressive.yaml`
- **Description**: Basic aggressive detection settings
- **Optimizer**: Genetic Algorithm
- **Best for**: Simple aggressive detection
- **Features**: Simple aggressive parameters without complex weights

### 9. Standard Conservative
- **File**: `experiments/experiment_conservative.yaml`
- **Description**: Basic conservative detection settings
- **Optimizer**: CMA-ES
- **Best for**: Simple conservative detection
- **Features**: Simple conservative parameters without complex weights

## Experiment Selection Guide

### For New Users
- **Start with**: Experiment 1 (Default Configuration)
- **Then try**: Experiment 4 (Balanced Detection)

### For Research/Development
- **Try**: Experiment 2 (High-Performance GA)
- **Or**: Experiment 3 (CMA-ES Optimization)
- **Advanced**: Experiment 7 (Custom Weights)

### For Production Use
- **Conservative**: Experiment 5 (Conservative Detection)
- **Balanced**: Experiment 4 (Balanced Detection)
- **Aggressive**: Experiment 6 (Aggressive Detection)

### For Quick Testing
- **Aggressive**: Experiment 8 (Standard Aggressive)
- **Conservative**: Experiment 9 (Standard Conservative)

## Key Differences

| Experiment | Optimizer | Detection Rate | Quality Focus | Exploration | Use Case |
|------------|-----------|----------------|---------------|-------------|----------|
| 1 | GA | Medium | High | Medium | General |
| 2 | GA | Medium | High | High | Research |
| 3 | CMA-ES | Medium | High | High | Research |
| 4 | CMA-ES | Medium | High | Medium | Production |
| 5 | CMA-ES | Low | Very High | Low | High Precision |
| 6 | GA | High | Medium | High | High Recall |
| 7 | GA | High | Low | Very High | Experimental |
| 8 | GA | High | Medium | Medium | Quick Aggressive |
| 9 | CMA-ES | Low | High | Low | Quick Conservative |

## Configuration Files

All experiment configurations are stored in the `experiments/` folder and follow the YAML format. You can:

1. Modify existing experiment files
2. Create new experiment files
3. Add new experiments to the main.py experiment list

## Results

Each experiment saves results to a timestamped folder in `results/` including:
- CSV files with anomaly scores
- PNG plots and visualizations
- JSON configuration and summary files
- Text reports with detailed analysis
- Optimization results (GA or CMA-ES specific)

## Customization

To create your own experiment:
1. Copy an existing experiment file
2. Modify the parameters as needed
3. Add it to the experiment list in main.py
4. Update the descriptions accordingly

