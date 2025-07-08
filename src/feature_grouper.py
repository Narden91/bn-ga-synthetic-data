import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from typing import List, Dict, Tuple
import random

class FeatureGrouper:
    """
    Groups features into smaller sets for scalable Bayesian Network learning.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the feature grouper.
        
        Args:
            config (Dict): Configuration for feature grouping
        """
        self.config = config
        self.groups = []
        self.group_info = {}
        
    def create_groups(self, data: pd.DataFrame) -> List[List[str]]:
        """
        Create feature groups based on the specified strategy.
        
        Args:
            data (pd.DataFrame): Preprocessed data
            
        Returns:
            List[List[str]]: List of feature groups (each group is a list of column names)
        """
        print("     Creating feature groups...")
        
        features = list(data.columns)
        group_size = self.config.get('group_size', 15)
        strategy = self.config.get('strategy', 'correlation')
        
        if strategy == 'random':
            self.groups = self._random_grouping(features, group_size)
        elif strategy == 'correlation':
            self.groups = self._correlation_based_grouping(data, group_size)
        elif strategy == 'domain':
            self.groups = self._domain_based_grouping(features, group_size)
        elif strategy == 'variance':
            self.groups = self._variance_based_grouping(data, group_size)
        else:
            print(f"     Unknown strategy '{strategy}', using random grouping")
            self.groups = self._random_grouping(features, group_size)
        
        # Store group information
        self._store_group_info(data)
        
        print(f"     Created {len(self.groups)} groups with strategy: {strategy}")
        self._print_group_stats()
        
        return self.groups
    
    def _random_grouping(self, features: List[str], group_size: int) -> List[List[str]]:
        """
        Create random feature groups.
        
        Args:
            features (List[str]): List of feature names
            group_size (int): Target size for each group
            
        Returns:
            List[List[str]]: Random feature groups
        """
        print("     Using random grouping strategy")
        
        # Shuffle features
        shuffled_features = features.copy()
        random.shuffle(shuffled_features)
        
        # Create groups
        groups = []
        for i in range(0, len(shuffled_features), group_size):
            group = shuffled_features[i:i + group_size]
            groups.append(group)
        
        return groups
    
    def _correlation_based_grouping(self, data: pd.DataFrame, group_size: int) -> List[List[str]]:
        """
        Create feature groups based on correlation clustering.
        
        Args:
            data (pd.DataFrame): Input data
            group_size (int): Target size for each group
            
        Returns:
            List[List[str]]: Correlation-based feature groups
        """
        print("     Using correlation-based grouping strategy")
        
        # Calculate correlation matrix
        correlation_matrix = data.corr().abs()
        
        # Handle NaN values in correlation matrix
        correlation_matrix = correlation_matrix.fillna(0)
        
        # Convert correlation to distance (1 - correlation)
        distance_matrix = 1 - correlation_matrix
        
        # Perform hierarchical clustering
        try:
            # Convert distance matrix to condensed form
            condensed_distances = pdist(distance_matrix, metric='precomputed')
            
            # Perform linkage
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Determine number of clusters
            n_features = len(data.columns)
            n_clusters = max(1, n_features // group_size)
            
            # Get cluster assignments
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Create groups from clusters
            groups = [[] for _ in range(n_clusters)]
            for i, label in enumerate(cluster_labels):
                groups[label - 1].append(data.columns[i])
            
            # Remove empty groups and balance group sizes
            groups = [group for group in groups if group]
            groups = self._balance_group_sizes(groups, group_size)
            
        except Exception as e:
            print(f"     Correlation clustering failed: {str(e)}, falling back to random")
            return self._random_grouping(list(data.columns), group_size)
        
        return groups
    
    def _domain_based_grouping(self, features: List[str], group_size: int) -> List[List[str]]:
        """
        Create feature groups based on domain knowledge (feature name patterns).
        
        Args:
            features (List[str]): List of feature names
            group_size (int): Target size for each group
            
        Returns:
            List[List[str]]: Domain-based feature groups
        """
        print("     Using domain-based grouping strategy")
        
        # Define common patterns in feature names
        patterns = {
            'voltage': ['V_', 'Gr_V_', 'phase_Gr_V_'],
            'current': ['I_', 'Gr_I_', 'phase_Gr_I_'],
            'power': ['P_', 'P_50', 'P_H', 'P_DC'],
            'phase': ['phi_', 'phase_'],
            'harmonic': ['_H', 'THD_'],
            'other': []  # Catch-all for remaining features
        }
        
        # Categorize features by patterns
        categorized_features = {category: [] for category in patterns.keys()}
        
        for feature in features:
            assigned = False
            for category, pattern_list in patterns.items():
                if category == 'other':
                    continue
                for pattern in pattern_list:
                    if pattern in feature:
                        categorized_features[category].append(feature)
                        assigned = True
                        break
                if assigned:
                    break
            
            if not assigned:
                categorized_features['other'].append(feature)
        
        # Create groups within each category
        groups = []
        for category, category_features in categorized_features.items():
            if not category_features:
                continue
            
            # Split category into groups of specified size
            for i in range(0, len(category_features), group_size):
                group = category_features[i:i + group_size]
                groups.append(group)
        
        # If we have very small groups, merge them
        groups = self._merge_small_groups(groups, min_size=3)
        
        return groups
    
    def _variance_based_grouping(self, data: pd.DataFrame, group_size: int) -> List[List[str]]:
        """
        Create feature groups based on variance levels.
        
        Args:
            data (pd.DataFrame): Input data
            group_size (int): Target size for each group
            
        Returns:
            List[List[str]]: Variance-based feature groups
        """
        print("     Using variance-based grouping strategy")
        
        # Calculate variance for each feature
        variances = data.var()
        
        # Sort features by variance
        sorted_features = variances.sort_values(ascending=False).index.tolist()
        
        # Create groups by interleaving high and low variance features
        groups = []
        n_groups = max(1, len(sorted_features) // group_size)
        
        for i in range(n_groups):
            group = []
            for j in range(group_size):
                feature_idx = (i + j * n_groups) % len(sorted_features)
                if feature_idx < len(sorted_features):
                    group.append(sorted_features[feature_idx])
            if group:
                groups.append(group)
        
        # Add remaining features to the last group
        remaining_features = sorted_features[n_groups * group_size:]
        if remaining_features:
            if groups:
                groups[-1].extend(remaining_features)
            else:
                groups.append(remaining_features)
        
        return groups
    
    def _balance_group_sizes(self, groups: List[List[str]], target_size: int) -> List[List[str]]:
        """
        Balance group sizes by redistributing features.
        
        Args:
            groups (List[List[str]]): Input groups
            target_size (int): Target group size
            
        Returns:
            List[List[str]]: Balanced groups
        """
        # Flatten all features
        all_features = [feature for group in groups for feature in group]
        
        # Create new balanced groups
        balanced_groups = []
        for i in range(0, len(all_features), target_size):
            group = all_features[i:i + target_size]
            balanced_groups.append(group)
        
        return balanced_groups
    
    def _merge_small_groups(self, groups: List[List[str]], min_size: int = 3) -> List[List[str]]:
        """
        Merge groups that are too small.
        
        Args:
            groups (List[List[str]]): Input groups
            min_size (int): Minimum group size
            
        Returns:
            List[List[str]]: Groups with small ones merged
        """
        merged_groups = []
        small_group_features = []
        
        for group in groups:
            if len(group) < min_size:
                small_group_features.extend(group)
            else:
                merged_groups.append(group)
        
        # Add small features to existing groups or create new group
        if small_group_features:
            if merged_groups:
                # Distribute small features among existing groups
                for i, feature in enumerate(small_group_features):
                    group_idx = i % len(merged_groups)
                    merged_groups[group_idx].append(feature)
            else:
                # Create new group from small features
                merged_groups.append(small_group_features)
        
        return merged_groups
    
    def _store_group_info(self, data: pd.DataFrame) -> None:
        """
        Store information about created groups.
        
        Args:
            data (pd.DataFrame): Input data
        """
        self.group_info = {
            'n_groups': len(self.groups),
            'group_sizes': [len(group) for group in self.groups],
            'total_features': sum(len(group) for group in self.groups),
            'strategy': self.config.get('strategy', 'unknown'),
            'target_group_size': self.config.get('group_size', 15)
        }
        
        # Calculate group statistics
        for i, group in enumerate(self.groups):
            if len(group) > 1:
                group_data = data[group]
                self.group_info[f'group_{i}_stats'] = {
                    'size': len(group),
                    'features': group,
                    'mean_variance': group_data.var().mean(),
                    'mean_correlation': group_data.corr().abs().mean().mean()
                }
    
    def _print_group_stats(self) -> None:
        """Print statistics about the created groups."""
        sizes = self.group_info['group_sizes']
        print(f"     Group sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
        print(f"     Total features distributed: {self.group_info['total_features']}")
    
    def get_group_info(self) -> Dict:
        """
        Get detailed information about the groups.
        
        Returns:
            Dict: Group information
        """
        return self.group_info
    
    def visualize_groups(self, data: pd.DataFrame) -> None:
        """
        Create visualization of feature groups (optional).
        
        Args:
            data (pd.DataFrame): Input data
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Plot group size distribution
            plt.figure(figsize=(10, 6))
            
            plt.subplot(1, 2, 1)
            sizes = [len(group) for group in self.groups]
            plt.hist(sizes, bins=10, alpha=0.7, edgecolor='black')
            plt.xlabel('Group Size')
            plt.ylabel('Frequency')
            plt.title('Distribution of Group Sizes')
            plt.grid(True, alpha=0.3)
            
            # Plot correlation heatmap for first few groups
            plt.subplot(1, 2, 2)
            if len(self.groups) > 0 and len(self.groups[0]) > 1:
                first_group = self.groups[0][:10]  # Limit to first 10 features for visibility
                corr_matrix = data[first_group].corr()
                sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
                plt.title('Correlation in First Group')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("     Matplotlib not available for visualization")
        except Exception as e:
            print(f"     Visualization failed: {str(e)}")
    
    def save_groups(self, file_path: str) -> None:
        """
        Save feature groups to file.
        
        Args:
            file_path (str): Path to save groups
        """
        try:
            import json
            
            group_data = {
                'groups': self.groups,
                'info': self.group_info,
                'config': self.config
            }
            
            with open(file_path, 'w') as f:
                json.dump(group_data, f, indent=2)
            
            print(f"     Groups saved to: {file_path}")
            
        except Exception as e:
            print(f"     Error saving groups: {str(e)}")
    
    def load_groups(self, file_path: str) -> List[List[str]]:
        """
        Load feature groups from file.
        
        Args:
            file_path (str): Path to load groups from
            
        Returns:
            List[List[str]]: Loaded feature groups
        """
        try:
            import json
            
            with open(file_path, 'r') as f:
                group_data = json.load(f)
            
            self.groups = group_data['groups']
            self.group_info = group_data['info']
            self.config.update(group_data['config'])
            
            print(f"     Groups loaded from: {file_path}")
            return self.groups
            
        except Exception as e:
            print(f"     Error loading groups: {str(e)}")
            return []
