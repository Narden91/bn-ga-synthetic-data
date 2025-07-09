import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings
import os

class ResultVisualizer:
    """
    Creates visualizations for anomaly detection results.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.figures = []
        self.base_results_dir = "results"
        self.execution_results_dir: Optional[str] = None
        os.makedirs(self.base_results_dir, exist_ok=True)
        # Create execution folder immediately upon initialization
        self.create_execution_folder()
        
    def create_visualizations(self, anomaly_scores: np.ndarray, 
                            anomaly_indices: np.ndarray,
                            likelihood_scores: pd.DataFrame) -> None:
        """
        Create comprehensive visualizations.
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            anomaly_indices (np.ndarray): Anomaly indices
            likelihood_scores (pd.DataFrame): Original likelihood scores
        """
        print("     Creating visualizations...")
        
        # Ensure execution folder exists
        self.get_results_dir()
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from datetime import datetime
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Clear any existing figures
            plt.close('all')
            
            # Create multiple visualizations
            self._plot_anomaly_score_distribution(anomaly_scores, anomaly_indices)
            self._plot_likelihood_heatmap(likelihood_scores, anomaly_indices)
            self._plot_anomaly_timeline(anomaly_scores, anomaly_indices)
            self._plot_feature_group_contributions(likelihood_scores, anomaly_indices)
            
            # Auto-save all figures
            self._auto_save_figures()
            
            print("     ✅ Visualizations created successfully")
            
        except ImportError:
            print("     ⚠️  Matplotlib/Seaborn not available, skipping visualizations")
        except Exception as e:
            print(f"     ❌ Visualization failed: {str(e)}")
    
    def _auto_save_figures(self):
        """Automatically save all open figures."""
        try:
            import matplotlib.pyplot as plt
            
            figure_names = [
                'anomaly_score_distribution',
                'likelihood_heatmap',
                'anomaly_timeline',
                'feature_group_contributions'
            ]
            
            # Get all open figures
            fig_nums = plt.get_fignums()
            
            for i, fig_num in enumerate(fig_nums):
                fig = plt.figure(fig_num)
                if i < len(figure_names):
                    filename = f"{figure_names[i]}.png"
                else:
                    filename = f"plot_{i}.png"
                
                filepath = os.path.join(self.get_results_dir(), filename)
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"     Saved: {filename}")
            
        except Exception as e:
            print(f"     Error auto-saving figures: {str(e)}")
    
    def _plot_anomaly_score_distribution(self, anomaly_scores: np.ndarray, 
                                       anomaly_indices: np.ndarray) -> None:
        """
        Plot distribution of anomaly scores.
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            anomaly_indices (np.ndarray): Anomaly indices
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Histogram of all scores
            ax1.hist(anomaly_scores, bins=50, alpha=0.7, color='skyblue', 
                    edgecolor='black', density=True, label='All samples')
            
            if len(anomaly_indices) > 0:
                ax1.hist(anomaly_scores[anomaly_indices], bins=30, alpha=0.8, 
                        color='red', edgecolor='darkred', density=True, label='Anomalies')
            
            ax1.set_xlabel('Anomaly Score')
            ax1.set_ylabel('Density')
            ax1.set_title('Distribution of Anomaly Scores')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Box plot comparison
            normal_indices = np.setdiff1d(np.arange(len(anomaly_scores)), anomaly_indices)
            
            data_to_plot = []
            labels = []
            
            if len(normal_indices) > 0:
                data_to_plot.append(anomaly_scores[normal_indices])
                labels.append('Normal')
            
            if len(anomaly_indices) > 0:
                data_to_plot.append(anomaly_scores[anomaly_indices])
                labels.append('Anomaly')
            
            if data_to_plot:
                box_plot = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
                colors = ['lightblue', 'lightcoral']
                for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
                    patch.set_facecolor(color)
            
            ax2.set_ylabel('Anomaly Score')
            ax2.set_title('Score Comparison: Normal vs Anomaly')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"     Score distribution plot failed: {str(e)}")
    
    def _plot_likelihood_heatmap(self, likelihood_scores: pd.DataFrame, 
                               anomaly_indices: np.ndarray) -> None:
        """
        Plot heatmap of likelihood scores across feature groups.
        
        Args:
            likelihood_scores (pd.DataFrame): Likelihood scores
            anomaly_indices (np.ndarray): Anomaly indices
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Limit to reasonable number of samples for visualization
            max_samples = 100
            sample_indices = np.arange(len(likelihood_scores))
            
            if len(sample_indices) > max_samples:
                # Sample normal and anomaly points
                normal_indices = np.setdiff1d(sample_indices, anomaly_indices)
                
                # Sample from normal points
                normal_sample = np.random.choice(
                    normal_indices, 
                    size=min(max_samples - len(anomaly_indices), len(normal_indices)), 
                    replace=False
                )
                
                # Combine with all anomaly points (limited)
                anomaly_sample = anomaly_indices[:min(20, len(anomaly_indices))]
                selected_indices = np.concatenate([normal_sample, anomaly_sample])
            else:
                selected_indices = sample_indices
            
            # Create heatmap data
            heatmap_data = likelihood_scores.iloc[selected_indices].T
            
            # Create annotation for anomalies
            anomaly_mask = np.isin(selected_indices, anomaly_indices)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create heatmap
            sns.heatmap(heatmap_data, 
                       cmap='RdYlBu_r', 
                       center=0,
                       cbar_kws={'label': 'Log-Likelihood'},
                       ax=ax)
            
            # Highlight anomaly columns
            for i, is_anomaly in enumerate(anomaly_mask):
                if is_anomaly:
                    ax.axvline(x=i+0.5, color='red', linewidth=2, alpha=0.7)
                    ax.axvline(x=i-0.5, color='red', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Feature Group')
            ax.set_title('Likelihood Scores Heatmap\n(Red lines indicate anomalies)')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"     Likelihood heatmap failed: {str(e)}")
    
    def _plot_anomaly_timeline(self, anomaly_scores: np.ndarray, 
                             anomaly_indices: np.ndarray) -> None:
        """
        Plot anomaly scores over sample indices (timeline).
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            anomaly_indices (np.ndarray): Anomaly indices
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot all scores
            sample_indices = np.arange(len(anomaly_scores))
            ax.plot(sample_indices, anomaly_scores, 'b-', alpha=0.6, linewidth=1, label='Anomaly Scores')
            
            # Highlight anomalies
            if len(anomaly_indices) > 0:
                ax.scatter(anomaly_indices, anomaly_scores[anomaly_indices], 
                          color='red', s=50, alpha=0.8, zorder=5, label='Detected Anomalies')
            
            # Add threshold line if available
            if len(anomaly_indices) > 0:
                threshold = np.min(anomaly_scores[anomaly_indices])
                ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, 
                          alpha=0.8, label=f'Threshold: {threshold:.3f}')
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Anomaly Score')
            ax.set_title('Anomaly Scores Timeline')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text box
            stats_text = f'Total Samples: {len(anomaly_scores)}\n'
            stats_text += f'Anomalies: {len(anomaly_indices)}\n'
            stats_text += f'Anomaly Rate: {len(anomaly_indices)/len(anomaly_scores)*100:.1f}%'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"     Timeline plot failed: {str(e)}")
    
    def _plot_feature_group_contributions(self, likelihood_scores: pd.DataFrame, 
                                        anomaly_indices: np.ndarray) -> None:
        """
        Plot feature group contributions to anomaly detection.
        
        Args:
            likelihood_scores (pd.DataFrame): Likelihood scores
            anomaly_indices (np.ndarray): Anomaly indices
        """
        try:
            import matplotlib.pyplot as plt
            
            if len(anomaly_indices) == 0:
                print("     No anomalies to analyze group contributions")
                return
            
            # Calculate mean likelihood scores for normal and anomaly samples
            normal_indices = np.setdiff1d(np.arange(len(likelihood_scores)), anomaly_indices)
            
            if len(normal_indices) == 0:
                print("     No normal samples to compare")
                return
            
            anomaly_means = likelihood_scores.iloc[anomaly_indices].mean()
            normal_means = likelihood_scores.iloc[normal_indices].mean()
            
            # Calculate contribution (difference between normal and anomaly means)
            contributions = normal_means - anomaly_means  # Higher = more discriminative
            contributions = contributions.sort_values(ascending=False)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: Group contributions
            x_pos = np.arange(len(contributions))
            bars = ax1.bar(x_pos, contributions.values, 
                          color=['red' if x > 0 else 'blue' for x in contributions.values],
                          alpha=0.7, edgecolor='black')
            
            ax1.set_xlabel('Feature Group')
            ax1.set_ylabel('Contribution to Anomaly Detection')
            ax1.set_title('Feature Group Contributions\n(Higher = More Discriminative)')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(contributions.index, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, contributions.values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Plot 2: Mean likelihood comparison
            group_names = likelihood_scores.columns[:min(10, len(likelihood_scores.columns))]  # Limit to 10 groups
            
            normal_subset = normal_means[group_names]
            anomaly_subset = anomaly_means[group_names]
            
            x_pos2 = np.arange(len(group_names))
            width = 0.35
            
            ax2.bar(x_pos2 - width/2, normal_subset.values, width, 
                   label='Normal', alpha=0.7, color='blue', edgecolor='black')
            ax2.bar(x_pos2 + width/2, anomaly_subset.values, width,
                   label='Anomaly', alpha=0.7, color='red', edgecolor='black')
            
            ax2.set_xlabel('Feature Group')
            ax2.set_ylabel('Mean Log-Likelihood')
            ax2.set_title('Mean Likelihood Comparison: Normal vs Anomaly')
            ax2.set_xticks(x_pos2)
            ax2.set_xticklabels(group_names, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"     Feature group contribution plot failed: {str(e)}")
    
    def create_summary_report(self, anomaly_scores: np.ndarray, 
                            anomaly_indices: np.ndarray,
                            likelihood_scores: pd.DataFrame,
                            feature_groups: List[List[str]]) -> str:
        """
        Create a text summary report.
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            anomaly_indices (np.ndarray): Anomaly indices
            likelihood_scores (pd.DataFrame): Likelihood scores
            feature_groups (List[List[str]]): Feature groups
            
        Returns:
            str: Summary report
        """
        report = []
        report.append("=" * 60)
        report.append("BAYESIAN ANOMALY DETECTION SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic statistics
        report.append("BASIC STATISTICS:")
        report.append(f"  Total samples: {len(anomaly_scores)}")
        report.append(f"  Anomalies detected: {len(anomaly_indices)}")
        report.append(f"  Anomaly rate: {len(anomaly_indices)/len(anomaly_scores)*100:.2f}%")
        report.append("")
        
        # Score statistics
        report.append("ANOMALY SCORE STATISTICS:")
        report.append(f"  Mean score: {np.mean(anomaly_scores):.4f}")
        report.append(f"  Standard deviation: {np.std(anomaly_scores):.4f}")
        report.append(f"  Min score: {np.min(anomaly_scores):.4f}")
        report.append(f"  Max score: {np.max(anomaly_scores):.4f}")
        report.append(f"  Median score: {np.median(anomaly_scores):.4f}")
        report.append("")
        
        # Threshold information
        if len(anomaly_indices) > 0:
            threshold = np.min(anomaly_scores[anomaly_indices])
            report.append("THRESHOLD INFORMATION:")
            report.append(f"  Threshold value: {threshold:.4f}")
            report.append(f"  Samples above threshold: {len(anomaly_indices)}")
            report.append("")
        
        # Feature group analysis
        report.append("FEATURE GROUP ANALYSIS:")
        report.append(f"  Total feature groups: {len(feature_groups)}")
        
        if len(anomaly_indices) > 0:
            normal_indices = np.setdiff1d(np.arange(len(likelihood_scores)), anomaly_indices)
            if len(normal_indices) > 0:
                anomaly_means = likelihood_scores.iloc[anomaly_indices].mean()
                normal_means = likelihood_scores.iloc[normal_indices].mean()
                contributions = normal_means - anomaly_means
                top_groups = contributions.nlargest(3)
                
                report.append("  Top 3 discriminative groups:")
                for i, (group, contrib) in enumerate(top_groups.items(), 1):
                    report.append(f"    {i}. {group}: {contrib:.4f}")
        report.append("")
        
        # Top anomalies
        if len(anomaly_indices) > 0:
            report.append("TOP ANOMALIES:")
            top_anomaly_indices = anomaly_indices[np.argsort(anomaly_scores[anomaly_indices])[-5:]][::-1]
            for i, idx in enumerate(top_anomaly_indices, 1):
                report.append(f"  {i}. Sample {idx}: score = {anomaly_scores[idx]:.4f}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_visualizations(self, output_dir: str = "visualizations") -> None:
        """
        Save all created visualizations to files.
        
        Args:
            output_dir (str): Output directory for visualizations
        """
        try:
            import os
            import matplotlib.pyplot as plt
            from datetime import datetime
            
            # Use results directory by default
            if output_dir == "visualizations":
                output_dir = self.get_results_dir()
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save all figures without timestamp (folder is timestamped)
            for i, fig in enumerate(plt.get_fignums()):
                plt.figure(fig)
                filepath = os.path.join(output_dir, f"anomaly_detection_plot_{i+1}.png")
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
            print(f"     Visualizations saved to {output_dir}/")
            
        except Exception as e:
            print(f"     Error saving visualizations: {str(e)}")
    
    def create_interactive_plot(self, anomaly_scores: np.ndarray, 
                              anomaly_indices: np.ndarray) -> None:
        """
        Create interactive plot (if plotly is available).
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            anomaly_indices (np.ndarray): Anomaly indices
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Anomaly Scores Timeline', 'Score Distribution'),
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )
            
            # Timeline plot
            sample_indices = np.arange(len(anomaly_scores))
            normal_indices = np.setdiff1d(sample_indices, anomaly_indices)
            
            # Normal points
            fig.add_trace(
                go.Scatter(x=normal_indices, y=anomaly_scores[normal_indices],
                          mode='markers', name='Normal', 
                          marker=dict(color='blue', size=4, opacity=0.6)),
                row=1, col=1
            )
            
            # Anomaly points
            if len(anomaly_indices) > 0:
                fig.add_trace(
                    go.Scatter(x=anomaly_indices, y=anomaly_scores[anomaly_indices],
                              mode='markers', name='Anomaly',
                              marker=dict(color='red', size=8, symbol='diamond')),
                    row=1, col=1
                )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=anomaly_scores, name='All Scores', opacity=0.7),
                row=2, col=1
            )
            
            if len(anomaly_indices) > 0:
                fig.add_trace(
                    go.Histogram(x=anomaly_scores[anomaly_indices], name='Anomaly Scores', 
                               opacity=0.8),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title_text="Interactive Anomaly Detection Results",
                height=800,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Sample Index", row=1, col=1)
            fig.update_yaxes(title_text="Anomaly Score", row=1, col=1)
            fig.update_xaxes(title_text="Anomaly Score", row=2, col=1)
            fig.update_yaxes(title_text="Count", row=2, col=1)
            
            fig.show()
            
        except ImportError:
            print("     Plotly not available for interactive plots")
        except Exception as e:
            print(f"     Interactive plot failed: {str(e)}")
    
    def export_results(self, anomaly_scores: np.ndarray, 
                      anomaly_indices: np.ndarray,
                      likelihood_scores: pd.DataFrame,
                      output_file: Optional[str] = None) -> None:
        """
        Export results to CSV file.
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            anomaly_indices (np.ndarray): Anomaly indices
            likelihood_scores (pd.DataFrame): Likelihood scores
            output_file (str): Output file path
        """
        try:
            # Default filename without timestamp (since folder is timestamped)
            if output_file is None:
                output_file = os.path.join(self.get_results_dir(), "anomaly_results.csv")
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'sample_index': np.arange(len(anomaly_scores)),
                'anomaly_score': anomaly_scores,
                'is_anomaly': np.isin(np.arange(len(anomaly_scores)), anomaly_indices)
            })
            
            # Add likelihood scores
            for col in likelihood_scores.columns:
                results_df[f'likelihood_{col}'] = likelihood_scores[col].values
            
            # Save to CSV
            results_df.to_csv(output_file, index=False)
            print(f"     Results exported to {output_file}")
            
        except Exception as e:
            print(f"     Error exporting results: {str(e)}")
    
    def create_execution_folder(self):
        """
        Create a timestamped folder for this execution's results.
        
        Returns:
            str: Path to the created execution folder
        """
        from datetime import datetime
        
        # Use microseconds to avoid conflicts in rapid executions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove last 3 digits of microseconds
        self.execution_results_dir = os.path.join(self.base_results_dir, f"execution_{timestamp}")
        os.makedirs(self.execution_results_dir, exist_ok=True)
        
        print(f"     Created results folder: {self.execution_results_dir}")
        return self.execution_results_dir
    
    def get_results_dir(self) -> str:
        """
        Get the current execution results directory.
        
        Returns:
            str: Path to the execution results directory
        """
        if self.execution_results_dir is None:
            self.create_execution_folder()
        assert self.execution_results_dir is not None  # for type checker
        return self.execution_results_dir
    
    def get_execution_folder_path(self) -> str:
        """
        Public method to get the execution folder path for use by other components.
        
        Returns:
            str: Path to the execution results directory
        """
        return self.get_results_dir()
