import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

import argparse

def make_plot_diffs(df, clean_df, all_eps=None, label_prefix='', marker='o', ylabel='Std Err(AT) - Std Err(Std) (%)', color=None):
    if all_eps is None:
        all_eps =[0.0039, 0.0078, 0.0118, 0.0157]
    take_fractions = [0.1, 0.125, 0.2, 0.5, 1.0]
    total = 60000
    num_samples = np.asarray([total * ss for ss in take_fractions])

    for i, eps in enumerate(all_eps):
        res = []
        var_res = []
        clean_res = []
        var_clean_res = []
        for take_fraction in take_fractions:
            take_frac_mask = np.asarray([f'take_frac={take_fraction}' in name for name in list(df['name'])])
            eps_mask = np.asarray([f'eps{eps}' in name for name in list(df['run_name'])])
            curr_df = df[take_frac_mask & eps_mask]
            clean_take_frac_mask = np.asarray([f'take_frac={take_fraction}' in name for name in list(clean_df['name'])])
            curr_clean_df = clean_df[clean_take_frac_mask]

            res.append((100 * curr_df.mean()).to_dict())
            var_res.append(((100**2 * curr_df.var()) / (len(curr_df) - 1)).to_dict())
            clean_res.append((100 * curr_clean_df.mean()).to_dict())
            var_clean_res.append(((100**2 * curr_clean_df.var()) / (len(curr_clean_df) - 1)).to_dict())
        curr_df = pd.DataFrame(res)
        curr_var_df = pd.DataFrame(var_res)
        curr_clean_df = pd.DataFrame(clean_res)
        curr_var_clean_df = pd.DataFrame(var_clean_res)

        diff = -(curr_df['test_accuracy']-curr_clean_df['test_accuracy'])
        std = np.sqrt(curr_var_df['test_accuracy'] + curr_var_clean_df['test_accuracy'])

        label = label_prefix + f'$\epsilon$={i+1}/255'
        if color is None:
            curr_color = f'C{i+2}'
        else:
            curr_color = color
        plt.semilogx(num_samples, diff, label=label, color=curr_color, marker=marker, markersize=10)
        plt.fill_between(
            num_samples,
            diff - std,
            diff + std,
            alpha=0.2, color=curr_color)
    plt.xlabel('Number of labeled samples')
    plt.ylabel(ylabel)
    plt.legend()
    plt.semilogx(num_samples, np.zeros(len(num_samples)), linestyle='dashed', color='gray')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='compile stats for sample sizes plot')
    parser.add_argument('-p','--paths', nargs='+', help='paths to process', required=True)
    args = parser.parse_args()

    dfs = []
    for stats_dir in args.paths:
        df = pd.read_csv(Path(stats_dir) / 'stats.csv')
        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    # filter into three dfs: clean, at, rst
    clean_mask = np.asarray(['clean' in name for name in df['name']])
    clean_df = df[clean_mask]

    at_mask = np.asarray(['unw=0.0-urw=0.0' in name for name in df['name']])
    at_df = df[at_mask]

    rst_mask = np.asarray(['urw=0.5' in name for name in df['name']])
    rst_df = df[rst_mask]

    make_plot_diffs(at_df, clean_df)
    plt.savefig('sample_sizes_at.png', bbox_inches='tight')
    plt.clf()
    make_plot_diffs(at_df, clean_df, label_prefix='AT, ', all_eps =[0.0039, 0.0078], ylabel='Std Err(Aug) - Std Err(Std) (%)', color='C1')
    make_plot_diffs(rst_df, clean_df, label_prefix='RST, ', all_eps=[0.0039, 0.0078], ylabel='Std Err(Aug) - Std Err(Std) (%)', color='mediumaquamarine')
    plt.savefig('sample_sizes_rst.png', bbox_inches='tight')
