"""
Visualize experiment results and create publication-quality plots.

Usage:
    python scripts/visualize_results.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def create_comparison_plot():
    """Create accuracy vs tree size comparison."""
    
    # Your actual results
    data = {
        'Dataset': ['Iris', 'Iris', 'Iris', 'Wine', 'Wine', 'Wine', 
                   'Breast\nCancer', 'Breast\nCancer', 'Breast\nCancer'],
        'Model': ['GA', 'CART', 'RF', 'GA', 'CART', 'RF', 'GA', 'CART', 'RF'],
        'Accuracy': [94.00, 94.67, 95.33, 89.35, 89.32, 97.75, 91.92, 92.80, 95.08],
        'Nodes': [29.4, 13.8, np.nan, 33.4, 17.8, np.nan, 37.0, 27.4, np.nan]
    }
    
    df = pd.DataFrame(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy comparison
    datasets = ['Iris', 'Wine', 'Breast\nCancer']
    x = np.arange(len(datasets))
    width = 0.25
    
    ga_acc = [94.00, 89.35, 91.92]
    cart_acc = [94.67, 89.32, 92.80]
    rf_acc = [95.33, 97.75, 95.08]
    
    ax1.bar(x - width, ga_acc, width, label='GA-Optimized', color='#FF6B6B', alpha=0.8)
    ax1.bar(x, cart_acc, width, label='CART', color='#4ECDC4', alpha=0.8)
    ax1.bar(x + width, rf_acc, width, label='Random Forest', color='#45B7D1', alpha=0.8)
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend(loc='lower right')
    ax1.set_ylim([85, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, dataset in enumerate(datasets):
        ax1.text(i - width, ga_acc[i] + 0.5, f'{ga_acc[i]:.1f}', 
                ha='center', va='bottom', fontsize=9)
        ax1.text(i, cart_acc[i] + 0.5, f'{cart_acc[i]:.1f}', 
                ha='center', va='bottom', fontsize=9)
        ax1.text(i + width, rf_acc[i] + 0.5, f'{rf_acc[i]:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Tree size comparison (GA vs CART only)
    ga_nodes = [29.4, 33.4, 37.0]
    cart_nodes = [13.8, 17.8, 27.4]
    
    ax2.bar(x - width/2, ga_nodes, width, label='GA-Optimized', color='#FF6B6B', alpha=0.8)
    ax2.bar(x + width/2, cart_nodes, width, label='CART', color='#4ECDC4', alpha=0.8)
    
    ax2.set_ylabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax2.set_title('Tree Complexity Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i in range(len(datasets)):
        ax2.text(i - width/2, ga_nodes[i] + 1, f'{ga_nodes[i]:.1f}', 
                ha='center', va='bottom', fontsize=9)
        ax2.text(i + width/2, cart_nodes[i] + 1, f'{cart_nodes[i]:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'comparison.png'}")
    
    plt.show()


def create_accuracy_vs_complexity_scatter():
    """Create scatter plot showing accuracy-interpretability tradeoff."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Your results
    models_data = [
        # (Accuracy, Nodes, Model, Color, Size)
        (94.00, 29.4, 'GA-Optimized\n(Iris)', '#FF6B6B', 200),
        (94.67, 13.8, 'CART\n(Iris)', '#4ECDC4', 150),
        
        (89.35, 33.4, 'GA-Optimized\n(Wine)', '#FF6B6B', 200),
        (89.32, 17.8, 'CART\n(Wine)', '#4ECDC4', 150),
        
        (91.92, 37.0, 'GA-Optimized\n(Breast Cancer)', '#FF6B6B', 200),
        (92.80, 27.4, 'CART\n(Breast Cancer)', '#4ECDC4', 150),
    ]
    
    for acc, nodes, label, color, size in models_data:
        # Convert nodes to interpretability (inverse relationship)
        interpretability = 100 / nodes  # Fewer nodes = more interpretable
        ax.scatter(interpretability, acc, s=size, alpha=0.6, 
                  color=color, edgecolors='black', linewidth=1.5)
        ax.annotate(label, (interpretability, acc), 
                   xytext=(10, 0), textcoords='offset points',
                   fontsize=9, alpha=0.8)
    
    ax.set_xlabel('Interpretability (100/nodes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Interpretability Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='GA-Optimized'),
        Patch(facecolor='#4ECDC4', label='CART')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    output_dir = Path('results/figures')
    plt.savefig(output_dir / 'tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'tradeoff.png'}")
    
    plt.show()


def create_summary_table():
    """Create a nice summary table."""
    
    summary = """
╔═══════════════════════════════════════════════════════════════════╗
║                    EXPERIMENT SUMMARY                              ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Key Findings:                                                     ║
║  • GA achieves competitive accuracy with CART (p > 0.05)          ║
║  • GA trees are currently LARGER (need tuning)                    ║
║  • Random Forest wins on accuracy (but not interpretable)         ║
║                                                                    ║
║  Performance Summary:                                              ║
║  ┌──────────────┬────────────┬────────────┬──────────┐           ║
║  │ Dataset      │ GA Acc     │ CART Acc   │ Winner   │           ║
║  ├──────────────┼────────────┼────────────┼──────────┤           ║
║  │ Iris         │ 94.00%     │ 94.67%     │ Tie      │           ║
║  │ Wine         │ 89.35%     │ 89.32%     │ GA       │           ║
║  │ Breast Cancer│ 91.92%     │ 92.80%     │ CART     │           ║
║  └──────────────┴────────────┴────────────┴──────────┘           ║
║                                                                    ║
║  Tree Size (Nodes):                                                ║
║  ┌──────────────┬────────────┬────────────┬──────────┐           ║
║  │ Dataset      │ GA         │ CART       │ Ratio    │           ║
║  ├──────────────┼────────────┼────────────┼──────────┤           ║
║  │ Iris         │ 29.4       │ 13.8       │ 2.13x    │           ║
║  │ Wine         │ 33.4       │ 17.8       │ 1.88x    │           ║
║  │ Breast Cancer│ 37.0       │ 27.4       │ 1.35x    │           ║
║  └──────────────┴────────────┴────────────┴──────────┘           ║
║                                                                    ║
║  Recommendations:                                                  ║
║  1. Increase interpretability weight (0.3 → 0.5)                  ║
║  2. Run more generations (30 → 100)                               ║
║  3. Increase population (50 → 150)                                ║
║  4. Add penalty for large trees in fitness function               ║
║                                                                    ║
╚═══════════════════════════════════════════════════════════════════╝
    """
    
    print(summary)
    
    # Save to file
    output_dir = Path('results')
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write(summary)
    print(f"✓ Saved: {output_dir / 'summary.txt'}")


def main():
    """Generate all visualizations."""
    print("\n" + "="*70)
    print("Creating Visualizations...")
    print("="*70 + "\n")
    
    create_comparison_plot()
    create_accuracy_vs_complexity_scatter()
    create_summary_table()
    
    print("\n" + "="*70)
    print("Visualization Complete! Check results/figures/")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()