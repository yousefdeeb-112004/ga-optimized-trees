"""Run Pareto optimization and visualize results - FIXED VERSION."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor


def run_multiple_configs(X_train, y_train, X_test, y_test, n_configs=10):
    """
    Run GA with different accuracy/interpretability weight combinations
    to approximate Pareto front.
    """
    print("="*70)
    print("PARETO FRONT APPROXIMATION")
    print("Running GA with different objective weights...")
    print("="*70)
    
    # Try different weight combinations
    weight_configs = []
    for acc_weight in np.linspace(0.3, 0.95, n_configs):
        weight_configs.append({
            'accuracy': acc_weight,
            'interpretability': 1.0 - acc_weight
        })
    
    results = []
    
    for i, weights in enumerate(weight_configs, 1):
        print(f"\nConfig {i}/{n_configs}: Accuracy={weights['accuracy']:.2f}, "
              f"Interpretability={weights['interpretability']:.2f}")
        
        # Setup
        n_features = X_train.shape[1]
        feature_ranges = {j: (X_train[:, j].min(), X_train[:, j].max()) 
                         for j in range(n_features)}
        
        ga_config = GAConfig(
            population_size=50,
            n_generations=30,
            crossover_prob=0.7,
            mutation_prob=0.2,
            tournament_size=3,
            elitism_ratio=0.15
        )
        
        # FIXED: Added missing arguments
        initializer = TreeInitializer(
            n_features=n_features,
            n_classes=2,
            max_depth=5,
            min_samples_split=10,  # ADDED
            min_samples_leaf=5     # ADDED
        )
        
        fitness_calc = FitnessCalculator(
            mode='weighted_sum',
            accuracy_weight=weights['accuracy'],
            interpretability_weight=weights['interpretability']
        )
        
        mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)
        
        # Train
        ga_engine = GAEngine(ga_config, initializer, 
                           fitness_calc.calculate_fitness, mutation)
        best_tree = ga_engine.evolve(X_train, y_train, verbose=False)
        
        # Evaluate
        predictor = TreePredictor()
        y_pred = predictor.predict(best_tree, X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'accuracy_weight': weights['accuracy'],
            'test_accuracy': accuracy,
            'interpretability': best_tree.interpretability_,
            'nodes': best_tree.get_num_nodes(),
            'depth': best_tree.get_depth(),
            'tree': best_tree
        })
        
        print(f"  → Test Accuracy: {accuracy:.4f}, "
              f"Interpretability: {best_tree.interpretability_:.4f}, "
              f"Nodes: {best_tree.get_num_nodes()}")
    
    return results


def plot_pareto_front(results, save_path='results/figures/pareto_front.png'):
    """Visualize approximate Pareto front."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy vs Interpretability
    ax1 = axes[0]
    
    accuracies = [r['test_accuracy'] for r in results]
    interpretabilities = [r['interpretability'] for r in results]
    nodes = [r['nodes'] for r in results]
    
    scatter = ax1.scatter(interpretabilities, accuracies, 
                         s=200, c=nodes, cmap='viridis', 
                         alpha=0.7, edgecolors='black', linewidth=2)
    
    ax1.set_xlabel('Interpretability Score', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title('Pareto Front: Accuracy vs Interpretability', 
                 fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Number of Nodes', fontsize=11)
    
    # Add annotations for extreme points
    min_nodes_idx = np.argmin(nodes)
    max_acc_idx = np.argmax(accuracies)
    
    ax1.annotate('Most Interpretable', 
                xy=(interpretabilities[min_nodes_idx], accuracies[min_nodes_idx]),
                xytext=(20, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    ax1.annotate('Most Accurate', 
                xy=(interpretabilities[max_acc_idx], accuracies[max_acc_idx]),
                xytext=(-20, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    # Plot 2: Accuracy vs Tree Size
    ax2 = axes[1]
    
    scatter2 = ax2.scatter(nodes, accuracies, 
                          s=200, c=interpretabilities, cmap='coolwarm',
                          alpha=0.7, edgecolors='black', linewidth=2)
    
    ax2.set_xlabel('Number of Nodes', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title('Trade-off: Accuracy vs Tree Complexity', 
                 fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Interpretability Score', fontsize=11)
    
    # Add "sweet spot" region
    ax2.axhspan(accuracies[max_acc_idx] - 0.02, max(accuracies), 
               alpha=0.1, color='green', label='High Accuracy Zone')
    ax2.axvspan(min(nodes), min(nodes) + 5, 
               alpha=0.1, color='blue', label='High Interpretability Zone')
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Pareto front saved to: {save_path}")
    
    plt.show()


def print_pareto_summary(results):
    """Print summary of Pareto front."""
    print("\n" + "="*70)
    print("PARETO FRONT SUMMARY")
    print("="*70)
    
    print(f"\n{'Config':<8} {'Acc Weight':<12} {'Test Acc':<12} {'Interp':<12} {'Nodes':<8} {'Depth'}")
    print("-"*70)
    
    for i, r in enumerate(results, 1):
        print(f"{i:<8} {r['accuracy_weight']:<12.2f} "
              f"{r['test_accuracy']:<12.4f} {r['interpretability']:<12.4f} "
              f"{r['nodes']:<8} {r['depth']}")
    
    # Find interesting solutions
    print("\n" + "="*70)
    print("INTERESTING SOLUTIONS")
    print("="*70)
    
    best_acc_idx = np.argmax([r['test_accuracy'] for r in results])
    smallest_tree_idx = np.argmin([r['nodes'] for r in results])
    best_interp_idx = np.argmax([r['interpretability'] for r in results])
    
    # Find balanced solution (closest to 0.7 accuracy weight)
    balanced_idx = np.argmin([abs(r['accuracy_weight'] - 0.7) for r in results])
    
    print(f"\n1. MOST ACCURATE:")
    print(f"   Accuracy: {results[best_acc_idx]['test_accuracy']:.4f}")
    print(f"   Nodes: {results[best_acc_idx]['nodes']}")
    print(f"   Interpretability: {results[best_acc_idx]['interpretability']:.4f}")
    
    print(f"\n2. MOST INTERPRETABLE:")
    print(f"   Accuracy: {results[best_interp_idx]['test_accuracy']:.4f}")
    print(f"   Nodes: {results[best_interp_idx]['nodes']}")
    print(f"   Interpretability: {results[best_interp_idx]['interpretability']:.4f}")
    
    print(f"\n3. SMALLEST TREE:")
    print(f"   Accuracy: {results[smallest_tree_idx]['test_accuracy']:.4f}")
    print(f"   Nodes: {results[smallest_tree_idx]['nodes']}")
    print(f"   Interpretability: {results[smallest_tree_idx]['interpretability']:.4f}")
    
    print(f"\n4. BALANCED (0.7 accuracy weight):")
    print(f"   Accuracy: {results[balanced_idx]['test_accuracy']:.4f}")
    print(f"   Nodes: {results[balanced_idx]['nodes']}")
    print(f"   Interpretability: {results[balanced_idx]['interpretability']:.4f}")


def main():
    """Run Pareto front approximation."""
    print("\n" + "="*70)
    print("PARETO FRONT ANALYSIS FOR BREAST CANCER")
    print("="*70)
    
    # Load data
    print("\nLoading dataset...")
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Run with multiple weight configurations
    results = run_multiple_configs(X_train, y_train, X_test, y_test, n_configs=10)
    
    # Visualize Pareto front
    plot_pareto_front(results)
    
    # Print summary
    print_pareto_summary(results)
    
    print("\n" + "="*70)
    print("Analysis Complete! 🎉")
    print("="*70)
    print("\nKey Insights:")
    print("• The Pareto front shows optimal trade-offs")
    print("• No single solution dominates on both objectives")
    print("• Choose solution based on your priorities:")
    print("  - High accuracy needed? → Pick rightmost point")
    print("  - Interpretability critical? → Pick topmost point")
    print("  - Balanced approach? → Pick middle points")


if __name__ == '__main__':
    main()