"""
Create publication-quality figures for research paper/thesis.
Highlights the key achievements: 48% size reduction and p=0.64 equivalence.
to see the same results in this visualzier, run python scripts/experiment.py --config configs/target.yaml
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
sns.set_palette("colorblind")

# Your actual results
results = {
    'iris': {'ga_acc': 94.55, 'cart_acc': 92.41, 'ga_nodes': 7.4, 'cart_nodes': 16.4, 'p_val': 0.186},
    'wine': {'ga_acc': 88.19, 'cart_acc': 87.22, 'ga_nodes': 10.7, 'cart_nodes': 20.7, 'p_val': 0.683},
    'breast_cancer': {'ga_acc': 91.05, 'cart_acc': 91.57, 'ga_nodes': 6.5, 'cart_nodes': 35.5, 'p_val': 0.640}
}

# FIGURE 1: The Money Shot - Size Reduction with Target Line
fig, ax = plt.subplots(figsize=(8, 5))

datasets = ['Iris', 'Wine', 'Breast\nCancer']
ga_nodes = [7.4, 10.7, 6.5]
cart_nodes = [16.4, 20.7, 35.5]
reductions = [55, 48, 82]

x = np.arange(len(datasets))
width = 0.35

bars1 = ax.bar(x - width/2, ga_nodes, width, label='GA', color='#2ecc71', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, cart_nodes, width, label='CART', color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)

# Add reduction percentages
for i, (ga, cart, red) in enumerate(zip(ga_nodes, cart_nodes, reductions)):
    y_pos = max(ga, cart) + 2
    color = '#27ae60' if red >= 46 and red <= 49 else '#3498db'
    marker = '★' if red >= 46 and red <= 49 else ''
    ax.text(i, y_pos, f'{red}%↓ {marker}', ha='center', fontsize=11, fontweight='bold', color=color)

# Target zone
ax.axhspan(0, 50, alpha=0.05, color='green', zorder=0)
ax.text(2.5, 25, '', fontsize=9, ha='right', color='darkgreen', style='italic')

ax.set_ylabel('Number of Nodes', fontsize=12, fontweight='bold')
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_title('GA Achieves 46-82% Tree Size Reduction\n', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 40)

plt.tight_layout()
plt.savefig('results/figures/paper_fig1_size_reduction.png', dpi=300, bbox_inches='tight')
plt.savefig('results/figures/paper_fig1_size_reduction.pdf', bbox_inches='tight')
print("✓ Figure 1 saved (Size Reduction)")

# FIGURE 2: Statistical Equivalence - The p-value Plot
fig, ax = plt.subplots(figsize=(8, 5))

datasets = ['Iris', 'Wine', 'Breast Cancer']
p_values = [0.186, 0.683, 0.640]
colors = ['#3498db', '#3498db', '#27ae60']  # Green for target achievement

bars = ax.barh(datasets, p_values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)

# Significance threshold line
ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05 (significance threshold)')

# Target line for breast cancer
ax.axvline(0.55, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Target p-value (0.55)')

# Annotations
for i, (dataset, p) in enumerate(zip(datasets, p_values)):
    label = f'p = {p:.3f}'
    if dataset == 'Wine' and p >= 0.46 and p <= 0.49:
        label += ' ★'
    if dataset == 'Breast Cancer' and p >= 0.55:
        label += ' ✓ Target!'
    ax.text(p + 0.03, i, label, va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('p-value (Paired t-test, 20-fold CV)', fontsize=12, fontweight='bold')
ax.set_title('Statistical Equivalence to CART\n(All p > 0.05 = No Significant Difference)', 
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(0, 0.75)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/figures/paper_fig2_statistical_equiv.png', dpi=300, bbox_inches='tight')
plt.savefig('results/figures/paper_fig2_statistical_equiv.pdf', bbox_inches='tight')
print("✓ Figure 2 saved (Statistical Equivalence)")

# FIGURE 3: Accuracy-Interpretability Trade-off Scatter
fig, ax = plt.subplots(figsize=(8, 6))

for dataset, data in results.items():
    # GA point
    ax.scatter(data['ga_nodes'], data['ga_acc'], s=400, alpha=0.8, 
               color='#2ecc71', marker='o', edgecolors='black', linewidth=2, label=f'GA - {dataset}')
    # CART point
    ax.scatter(data['cart_nodes'], data['cart_acc'], s=300, alpha=0.6,
               color='#e74c3c', marker='s', edgecolors='black', linewidth=2, label=f'CART - {dataset}')
    
    # Connect with arrow
    ax.annotate('', xy=(data['ga_nodes'], data['ga_acc']), 
                xytext=(data['cart_nodes'], data['cart_acc']),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))

# Ideal region
ax.axhspan(90, 95, alpha=0.05, color='green', zorder=0)
ax.axvspan(0, 15, alpha=0.05, color='blue', zorder=0)
ax.text(7.5, 94.5, '← Ideal Region', fontsize=10, color='darkgreen', fontweight='bold')

ax.set_xlabel('Tree Size (Number of Nodes)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('GA Finds Pareto-Optimal Solutions\n(Smaller trees, competitive accuracy)', 
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 40)
ax.set_ylim(85, 96)

# Legend with summary
handles, labels = ax.get_legend_handles_labels()
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', 
               markersize=12, label='GA (Smaller)', markeredgecolor='black', markeredgewidth=1.5),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#e74c3c', 
               markersize=10, label='CART (Baseline)', markeredgecolor='black', markeredgewidth=1.5)
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.95)

plt.tight_layout()
plt.savefig('results/figures/paper_fig3_pareto_tradeoff.png', dpi=300, bbox_inches='tight')
plt.savefig('results/figures/paper_fig3_pareto_tradeoff.pdf', bbox_inches='tight')
print("✓ Figure 3 saved (Pareto Trade-off)")

# FIGURE 4: Comparison Table as Image (for slides)
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

table_data = [
    ['Iris', '94.55 ± 8.07%', '92.41 ± 10.43%', '0.186', '7.4', '16.4', '55%'],
    ['Wine', '88.19 ± 10.39%', '87.22 ± 10.70%', '0.683', '10.7', '20.7', '48%'],
    ['Breast Cancer', '91.05 ± 5.60%', '91.57 ± 3.92%', '0.640 ✓', '6.5', '35.5', '82%']
]

headers = ['Dataset', 'GA Accuracy', 'CART Accuracy', 'p-value', 'GA Nodes', 'CART Nodes', 'Reduction']

table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                colWidths=[0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white')

# Highlight achievements
for i in range(1, 4):
    # p-value column - green if > 0.05
    cell = table[(i, 3)]
    if float(table_data[i-1][3].split()[0]) > 0.05:
        cell.set_facecolor('#d5f4e6')
    
    # Reduction column - green if 46-49%
    cell = table[(i, 6)]
    reduction = int(table_data[i-1][6].replace('%', '').split()[0])
    if 46 <= reduction <= 49:
        cell.set_facecolor('#f9e79f')  # Gold for exact target
    else:
        cell.set_facecolor('#d5f4e6')

ax.set_title('Complete Results Summary (20-fold CV)', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('results/figures/paper_table_summary.png', dpi=300, bbox_inches='tight')
print("✓ Summary table saved")

print("\n" + "="*70)
print("✅ All paper-quality figures generated!")
print("="*70)
print("\nFiles created:")
print("  1. paper_fig1_size_reduction.png/pdf (The 48% achievement)")
print("  2. paper_fig2_statistical_equiv.png/pdf (The p=0.64 proof)")
print("  3. paper_fig3_pareto_tradeoff.png/pdf (Trade-off visualization)")
print("  4. paper_table_summary.png (Results table)")
print("\n💡 Use these for your paper/presentation!")