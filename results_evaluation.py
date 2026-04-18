import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from os.path import join, isfile, split, isdir
from os import listdir
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from config import get_result_path


def compute_metrics(y_true, y_pred, set):
    def get_flat_dict(d, parent_key='', sep= '_'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(get_flat_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    metrics = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # filtering single class metrics
    filtered_metrics = {k: v for k, v in metrics.items() if not k[0].isdigit()}
    flatten_metrics = get_flat_dict(filtered_metrics)
    flatten_metrics ={f'{set}_{k}': round(v*100, 2) for k, v in flatten_metrics.items() if 'support' not in k}

    return flatten_metrics


def get_comb_filepaths(run_path):
    csv_filepaths = []
    combinations = list(listdir(run_path))
    for combination in combinations:
        # results/ds/comb
        if combination == '.DS_Store':
            continue
        comb_path = join(run_path, combination)
        if not isdir(comb_path):
            continue
        comb_items = list(listdir(comb_path))
        for comb_item in comb_items:
            # results/ds/comb/results_comb.csv
            if comb_item == '.DS_Store':
                continue
            item_path = join(comb_path, comb_item)
            is_result_csv = isfile(item_path) and comb_item.startswith('results') and comb_item.endswith('.csv')

            if is_result_csv:
                csv_filepaths.append(item_path)

    return csv_filepaths


def plot_confusion_matrix(results, classes, path):
    labels = torch.tensor(results['label'].tolist())
    predictions = torch.tensor(results['prediction'].tolist())
    stacked = torch.stack((labels, predictions), dim=1)
    cmt = torch.zeros(len(classes), len(classes), dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cmt[int(tl), int(pl)] = cmt[int(tl), int(pl)] + 1

    cmt_np = cmt.numpy()
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    cmt_percent = cmt_np.astype('float') / cmt_np.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 8))

    sns.heatmap(cmt_percent,
                annot=True,
                fmt='.1f',
                cmap='RdYlBu_r',  # rosso-blu 
                xticklabels=classes,
                yticklabels=classes,
                square=True,
                linewidths=1.5,
                linecolor='white',
                cbar_kws={'shrink': 0.8, 'aspect': 20, 'label': 'Percentage (%)'},
                annot_kws={'size': 12, 'weight': 'bold'})

    plt.title('Confusion Matrix (Percentages)', fontsize=16, fontweight='bold', pad=20)

    plt.xlabel('Predicted', fontsize=12, fontweight='semibold')
    plt.ylabel('True', fontsize=12, fontweight='semibold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='right')
    plt.tight_layout()
    plt.savefig(join(path, 'confusion_matrix_test.png'),
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                pad_inches=0.1)
    plt.close('all')


def plot_metrics_by_prefix_size(results, path):
    def _calculate_accuracy(group):
        return accuracy_score(group['label'], group['prediction'])

    def _calculate_f1(group):
        return f1_score(group['label'], group['prediction'], average='weighted')  # , zero_division=1))

    # metriche dell'epoca al variare della lunghezza del prefisso per il set corrente
    accuracy = results.groupby('active_prefix_size').apply(_calculate_accuracy).reset_index(name='test_accuracy')
    w_f1 = results.groupby('active_prefix_size').apply(_calculate_f1).reset_index(name='test_weighted_f1')
    prefix_counter = results.groupby('active_prefix_size').size().reset_index(name='count').sort_values(by='active_prefix_size')

    result = pd.merge(accuracy, w_f1, on='active_prefix_size')
    result = pd.merge(result, prefix_counter, on='active_prefix_size')

    fig, ax1 = plt.subplots(figsize=(10, 4))
    # plt.suptitle("Metrics on test set varying prefix size", fontsize=16, fontweight='bold', y=0.95)

    ax1.set_xlabel("Active prefix size", fontsize=12, fontweight='semibold')
    ax1.set_ylabel("Metrics", fontsize=12, fontweight='semibold')
    sns.lineplot(data=results, x='active_prefix_size', y='test_weighted_f1', color='blue', marker='o', linewidth=2.5, markersize=8,
                      label='Weighted F1-score', ax=ax1)
    sns.lineplot(data=results, x='active_prefix_size', y='test_accuracy', color='orange', marker='s', linewidth=2.5, markersize=8,
                      label='Accuracy', ax=ax1)

    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    sns.lineplot(data=results, x='active_prefix_size', y='count', color='red', linestyle='--', marker='^',
                      linewidth=2.5, markersize=8, label='#samples', ax=ax2)
    ax2.set_ylabel('Samples', fontsize=12, fontweight='semibold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax1.set_xticks(sorted(results['active_prefix_size'].unique()))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=True, fancybox=True, shadow=True)
    ax2.get_legend().remove() if ax2.get_legend() else None

    plt.tight_layout()
    plt.savefig(join(path, 'prefix_metrics_test.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close('all')


def best_metric_on_set(df, log_name, k, path, metric='loss', set='test'):
    on_metric = f'{set}_{metric}'

    list_dfs = []
    for comb in df['combination'].unique().tolist():
        comb_df = df.loc[df['combination'] == comb]
        if metric == 'loss':
            best = comb_df.loc[(comb_df[on_metric] == comb_df[on_metric].min())]
        else:
            best = comb_df.loc[(comb_df[on_metric] == comb_df[on_metric].max())]
        list_dfs.append(best)

    best_df = pd.concat(list_dfs)
    best_df.sort_values(by=[on_metric], ascending=True if metric == 'loss' else False, inplace=True)
    best_df.to_csv(join(path, f'{log_name}_{k}_k_best_{on_metric}.csv'), header=True, index=False, sep=',')


def eval_results(log_name, variant, k):
    df_results = pd.DataFrame()
    result_path = get_result_path(log_name, variant, k)
    combinations = get_comb_filepaths(result_path)
    for idx, comb_filepath in enumerate(combinations):
        comb_string = split(comb_filepath)[1].split('.csv')[0]
        print(f'** Processing: {comb_string} ({idx+1}/{len(combinations)})')
        comb_results = pd.read_csv(comb_filepath, header=0)

        comb_results[['log_name', 'combination']] = [f'{log_name}_{k}_k', comb_string]
        df_results = pd.concat([df_results, comb_results])

    col = df_results.pop("log_name")
    df_results.insert(0, "log_name", col)
    col = df_results.pop("combination")
    df_results.insert(1, "combination", col)

    best_metric_on_set(df_results, log_name, k, result_path, metric='loss', set='test')
