import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ==================== FILES ====================
files = {
    'Fict': '/Users/niccolociotti/Desktop/baseline_gnn_active_prefixes_nap/results/var_fict_200K_3/Helpdesk_no_resources_200_k/0.0001_lr_0_l_64_s_2_h/best_loss_prefix_results_fict.csv',
    'Current': '/Users/niccolociotti/Desktop/baseline_gnn_active_prefixes_nap/results/var_current_1_200K/Helpdesk_no_resources_200_k/0.001_lr_0_l_64_s_1_h/best_loss_prefix_results_current.csv',
    '2GNN': '/Users/niccolociotti/Desktop/baseline_gnn_active_prefixes_nap/results/var_2gnn_2_200K/Helpdesk_no_resources_200_k/0.001_lr_2_l_64_s_1_h/best_loss_prefix_results_2gnn.csv',
    'Baseline': '/Users/niccolociotti/Desktop/baseline_gnn_active_prefixes_nap/results/var_baseline_0K_2/Helpdesk_no_resources_0_k/0.0001_lr_2_l_64_s_1_h/best_loss_prefix_results.csv'
}

# ==================== ESTRAZIONE DATI ====================
results_dict = {}

for name, file in files.items():
    df = pd.read_csv(file)
    df_test = df[df['set'] == 'test']
    sizes = sorted(df_test['size'].unique())

    accuracies = []
    f1_scores = []
    num_samples = []

    for s in sizes:
        subset = df_test[df_test['size'] == s]

        acc = accuracy_score(subset['label'], subset['prediction']) * 100
        f1 = f1_score(subset['label'], subset['prediction'], average='weighted') * 100

        accuracies.append(acc)
        f1_scores.append(f1)
        num_samples.append(len(subset))

    results_dict[name] = {
        'sizes': np.array(sizes),
        'accuracy': np.array(accuracies),
        'f1_score': np.array(f1_scores),
        'num_samples': np.array(num_samples)
    }

# ==================== CALCOLO SOGLIA DINAMICA ====================
min_samples = 30

# Prendiamo sizes e samples dal primo modello (sono identici per tutti)
first_key = next(iter(results_dict))
sizes = results_dict[first_key]['sizes']
samples = results_dict[first_key]['num_samples']

threshold_candidates = sizes[samples < min_samples]

if len(threshold_candidates) > 0:
    threshold_val = threshold_candidates[0]
else:
    threshold_val = None

# ==================== STILE ====================
colors = {'Fict': '#1f77b4', 'Current': '#2ca02c', '2GNN': '#ff7f0e', 'Baseline': "#ab0059"}
markers = {'Fict': 'o', 'Current': 's', '2GNN': '^', 'Baseline': 'D'}
line_styles = {'Fict': '-', 'Current': '--', '2GNN': '-.', 'Baseline': ':'}
offsets = {'Fict': -0.05, 'Current': 0, '2GNN': 0.05, 'Baseline': 0.1}

# ==================== FIGURA 1: F1-SCORE ====================
fig, ax1 = plt.subplots(figsize=(8, 6))

for name in files.keys():
    x_shifted = results_dict[name]['sizes'] + offsets[name]

    ax1.plot(
        x_shifted,
        results_dict[name]['f1_score'],
        color=colors[name],
        marker=markers[name],
        linestyle=line_styles[name],
        linewidth=2,
        markersize=7,
        alpha=0.8,
        label=f'{name} (F1-score)'
    )

ax1.set_xlabel('Prefix-IGs length (numero di nodi)')
ax1.set_ylabel('Weighted F1-score (%)')
ax1.set_ylim(40, 105)
ax1.grid(True, linestyle='--', alpha=0.5)

# Numero campioni (asse destro)
ax1_twin = ax1.twinx()
ax1_twin.plot(
    sizes,
    samples,
    color='red',
    linestyle=':',
    linewidth=2,
    alpha=0.6,
    label='Number of samples'
)
ax1_twin.set_ylabel('Number of samples', color='red')
ax1_twin.tick_params(axis='y', labelcolor='red')

# Linea verticale dinamica
if threshold_val is not None:
    ax1.axvline(
        x=threshold_val,
        color='black',
        linestyle=':',
        linewidth=1.5,
        label=f'Prefix length with <{min_samples} samples ({threshold_val})'
    )

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax1_twin.get_legend_handles_labels()

ax1.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.3),
    ncol=2,
    frameon=False
)

plt.tight_layout()
plt.savefig('f1_score_results.png', bbox_inches='tight')
plt.close()

# ==================== FIGURA 2: ACCURACY ====================
fig, ax2 = plt.subplots(figsize=(8, 6))

for name in files.keys():
    x_shifted = results_dict[name]['sizes'] + offsets[name]

    ax2.plot(
        x_shifted,
        results_dict[name]['accuracy'],
        color=colors[name],
        marker=markers[name],
        linestyle=line_styles[name],
        linewidth=2,
        markersize=7,
        alpha=0.8,
        label=f'{name} (Accuracy)'
    )

ax2.set_xlabel('Prefix-IGs length (numero di nodi)')
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim(40, 105)
ax2.grid(True, linestyle='--', alpha=0.5)

# Numero campioni (asse destro)
ax2_twin = ax2.twinx()
ax2_twin.plot(
    sizes,
    samples,
    color='red',
    linestyle=':',
    linewidth=2,
    alpha=0.6,
    label='Number of samples'
)
ax2_twin.set_ylabel('Number of samples', color='red')
ax2_twin.tick_params(axis='y', labelcolor='red')

# Linea verticale dinamica
if threshold_val is not None:
    ax2.axvline(
        x=threshold_val,
        color='black',
        linestyle=':',
        linewidth=1.5,
        label=f'Prefix length with <{min_samples} samples ({threshold_val})'
    )

lines_3, labels_3 = ax2.get_legend_handles_labels()
lines_4, labels_4 = ax2_twin.get_legend_handles_labels()

ax2.legend(
    lines_3 + lines_4,
    labels_3 + labels_4,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.3),
    ncol=2,
    frameon=False
)

plt.tight_layout()
plt.savefig('accuracy_results.png', bbox_inches='tight')
plt.close()