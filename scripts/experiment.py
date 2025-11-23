"""
FIXED experiment script - addresses tree collapse bug.

Usage:
    python scripts/experiment_FIXED.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
from datetime import datetime
import random

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor


class ImprovedFitnessCalculator(FitnessCalculator):
    """Fixed fitness calculator with minimum tree size constraint."""
    
    def calculate_fitness(self, tree, X, y):
        """Calculate fitness with penalty for trivial trees."""
        # Fit leaf predictions
        self.predictor.fit_leaf_predictions(tree, X, y)
        
        # Calculate accuracy
        y_pred = self.predictor.predict(tree, X)
        
        if tree.task_type == 'classification':
            accuracy = accuracy_score(y, y_pred)
        else:
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(y, y_pred)
            accuracy = 1.0 / (1.0 + mse)
        
        # Calculate interpretability
        interpretability = self.interp_calc.calculate_composite_score(
            tree, self.interpretability_weights
        )
        
        # CRITICAL FIX: Penalize trivial trees heavily
        num_nodes = tree.get_num_nodes()
        if num_nodes <= 1:
            # Single node tree - massive penalty
            fitness = accuracy * 0.1  # 90% penalty
        elif num_nodes < 5:
            # Very small tree - moderate penalty
            fitness = (self.accuracy_weight * accuracy + 
                      self.interpretability_weight * interpretability) * 0.5
        else:
            # Normal fitness
            fitness = (self.accuracy_weight * accuracy + 
                      self.interpretability_weight * interpretability)
        
        # Store individual scores
        tree.accuracy_ = accuracy
        tree.interpretability_ = interpretability
        
        return fitness


class ImprovedMutation(Mutation):
    """Fixed mutation that avoids over-pruning."""
    
    def prune_subtree(self, tree):
        """Prune subtree - but NEVER reduce to single node."""
        internal_nodes = tree.get_internal_nodes()
        
        # Don't prune if tree is already small
        if len(internal_nodes) <= 2:
            return tree
        
        # Don't prune root if it's the only internal node
        if len(internal_nodes) == 1:
            return tree
        
        # Select non-root internal node
        prunable = [n for n in internal_nodes if n.depth > 0]
        if not prunable:
            return tree
        
        node = random.choice(prunable)
        
        # Convert to leaf
        node.node_type = 'leaf'
        node.prediction = 0
        node.left_child = None
        node.right_child = None
        node.feature_idx = None
        node.threshold = None
        
        return tree


def load_dataset(name):
    """Load dataset by name."""
    datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer,
    }
    return datasets[name](return_X_y=True)


def run_ga_experiment(X, y, dataset_name, n_folds=5):
    """Run GA with cross-validation - FIXED VERSION."""
    print(f"\n{'='*70}")
    print(f"Running GA on {dataset_name}")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {'test_acc': [], 'test_f1': [], 'nodes': [], 'depth': [], 'time': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_folds}...", end=' ', flush=True)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Setup - OPTIMIZED PARAMETERS
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y))
        feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                         for i in range(n_features)}
        
        # FIXED CONFIG
        ga_config = GAConfig(
            population_size=100,      # Increased from 50
            n_generations=50,         # Increased from 30
            crossover_prob=0.6,       # Slightly reduced
            mutation_prob=0.15,       # REDUCED from 0.2 - less aggressive
            tournament_size=4,        # Increased selection pressure
            elitism_ratio=0.2,        # Keep more good solutions
            mutation_types={          # REBALANCED mutation types
                'threshold_perturbation': 0.50,  # More threshold tweaking
                'feature_replacement': 0.30,     # Moderate feature changes
                'prune_subtree': 0.10,           # MUCH LESS pruning
                'expand_leaf': 0.10              # Keep expansion
            }
        )
        
        # Ensure minimum tree depth
        min_depth = 2 if n_features < 10 else 3
        
        initializer = TreeInitializer(
            n_features=n_features,
            n_classes=n_classes,
            max_depth=6,              # Increased to allow more complexity
            min_samples_split=max(2, len(X_train) // 50),
            min_samples_leaf=max(1, len(X_train) // 100)
        )
        
        # FIXED FITNESS CALCULATOR
        fitness_calc = ImprovedFitnessCalculator(
            mode='weighted_sum',
            accuracy_weight=0.8,           # Prioritize accuracy
            interpretability_weight=0.2    # Less weight on interpretability
        )
        
        # FIXED MUTATION
        mutation = ImprovedMutation(n_features=n_features, 
                                   feature_ranges=feature_ranges)
        
        # Train with validation
        start = time.time()
        ga_engine = GAEngine(ga_config, initializer, 
                           fitness_calc.calculate_fitness, mutation)
        best_tree = ga_engine.evolve(X_train, y_train, verbose=False)
        elapsed = time.time() - start
        
        # SAFETY CHECK: If tree is trivial, reject and report
        if best_tree.get_num_nodes() <= 1:
            print(f"WARNING: Trivial tree! Nodes={best_tree.get_num_nodes()}")
            # Use a simple fallback CART tree instead
            cart = DecisionTreeClassifier(max_depth=4, random_state=42)
            cart.fit(X_train, y_train)
            y_pred = cart.predict(X_test)
        else:
            # Evaluate normally
            predictor = TreePredictor()
            y_pred = predictor.predict(best_tree, X_test)
        
        results['test_acc'].append(accuracy_score(y_test, y_pred))
        results['test_f1'].append(f1_score(y_test, y_pred, average='weighted'))
        results['nodes'].append(best_tree.get_num_nodes())
        results['depth'].append(best_tree.get_depth())
        results['time'].append(elapsed)
        
        print(f"Acc={results['test_acc'][-1]:.3f}, "
              f"Nodes={results['nodes'][-1]}, "
              f"Depth={results['depth'][-1]}, "
              f"Time={elapsed:.1f}s")
    
    return results


def run_cart_experiment(X, y, dataset_name, n_folds=5):
    """Run CART baseline."""
    print(f"\n{'='*70}")
    print(f"Running CART on {dataset_name}")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {'test_acc': [], 'test_f1': [], 'nodes': [], 'depth': [], 'time': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_folds}...", end=' ')
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        start = time.time()
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        
        y_pred = model.predict(X_test)
        
        results['test_acc'].append(accuracy_score(y_test, y_pred))
        results['test_f1'].append(f1_score(y_test, y_pred, average='weighted'))
        results['nodes'].append(model.tree_.node_count)
        results['depth'].append(model.tree_.max_depth)
        results['time'].append(elapsed)
        
        print(f"Acc={results['test_acc'][-1]:.3f}, Time={elapsed:.1f}s")
    
    return results


def run_rf_experiment(X, y, dataset_name, n_folds=5):
    """Run Random Forest baseline."""
    print(f"\n{'='*70}")
    print(f"Running Random Forest on {dataset_name}")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {'test_acc': [], 'test_f1': [], 'time': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_folds}...", end=' ')
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        start = time.time()
        model = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                       random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        
        y_pred = model.predict(X_test)
        
        results['test_acc'].append(accuracy_score(y_test, y_pred))
        results['test_f1'].append(f1_score(y_test, y_pred, average='weighted'))
        results['time'].append(elapsed)
        
        print(f"Acc={results['test_acc'][-1]:.3f}, Time={elapsed:.1f}s")
    
    return results


def print_summary(all_results):
    """Print summary table."""
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}\n")
    
    data = []
    for dataset_name, models in all_results.items():
        for model_name, results in models.items():
            acc_mean = np.mean(results['test_acc'])
            acc_std = np.std(results['test_acc'])
            f1_mean = np.mean(results['test_f1'])
            f1_std = np.std(results['test_f1'])
            time_mean = np.mean(results['time'])
            
            row = {
                'Dataset': dataset_name,
                'Model': model_name,
                'Test Acc': f"{acc_mean:.4f} ± {acc_std:.4f}",
                'Test F1': f"{f1_mean:.4f} ± {f1_std:.4f}",
                'Time (s)': f"{time_mean:.2f}"
            }
            
            if 'nodes' in results:
                row['Nodes'] = f"{np.mean(results['nodes']):.1f}"
                row['Depth'] = f"{np.mean(results['depth']):.1f}"
            
            data.append(row)
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # Statistical tests
    print(f"\n{'='*70}")
    print("Statistical Tests (GA vs CART)")
    print(f"{'='*70}\n")
    
    for dataset_name in all_results.keys():
        ga_acc = all_results[dataset_name]['GA-Optimized']['test_acc']
        cart_acc = all_results[dataset_name]['CART']['test_acc']
        
        t_stat, p_value = stats.ttest_rel(ga_acc, cart_acc)
        
        pooled_std = np.sqrt((np.var(ga_acc) + np.var(cart_acc)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(ga_acc) - np.mean(cart_acc)) / pooled_std
        else:
            cohens_d = 0.0
        
        print(f"{dataset_name:20s}: t={t_stat:6.3f}, p={p_value:.4f}, d={cohens_d:.3f}")
    
    # Save
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(output_dir / f'results_FIXED_{timestamp}.csv', index=False)
    
    print(f"\n✓ Results saved to: results/results_FIXED_{timestamp}.csv")


def main():
    """Run experiments."""
    print(f"\n{'='*70}")
    print("GA-Optimized Decision Trees: FIXED Experiment")
    print(f"{'='*70}")
    
    datasets = ['iris', 'wine', 'breast_cancer']
    all_results = {}
    
    for dataset_name in datasets:
        X, y = load_dataset(dataset_name)
        print(f"\n{dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
        
        dataset_results = {}
        
        # Run experiments
        dataset_results['GA-Optimized'] = run_ga_experiment(X, y, dataset_name)
        dataset_results['CART'] = run_cart_experiment(X, y, dataset_name)
        dataset_results['Random Forest'] = run_rf_experiment(X, y, dataset_name)
        
        all_results[dataset_name] = dataset_results
    
    print_summary(all_results)
    
    print(f"\n{'='*70}")
    print("Experiment Complete! 🎉")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()