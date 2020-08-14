import json
from pathlib import Path
import numpy as np
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='compile stats for sample sizes plot')
    parser.add_argument('-p', '--paths', nargs='+',
                        help='paths to process', required=True)
    args = parser.parse_args()

    nat_accs = []
    rob_accs = []
    for model_dir in args.paths:
        model_dir = Path(model_dir)

        if 'clean_models' in str(model_dir):
            df = pd.read_csv(model_dir / 'stats_eval.csv')
            nat_acc = df.iloc[-1]['test_accuracy'] * 100
            rob_acc = df.iloc[-1]['test_robust_accuracy'] * 100
        else:
            final_attack_stats_json = model_dir / 'stats.json'
            if not final_attack_stats_json.exists():
                for inner_dir in model_dir.iterdir():
                    if str(inner_dir.name).startswith('eps='):
                        break
                final_attack_stats_json = inner_dir / 'stats.json'

            with open(final_attack_stats_json, 'r') as f:
                final_attack_stats = json.load(f)

            nat_acc = final_attack_stats['natural_accuracy'] * 100
            rob_acc = final_attack_stats['robust_accuracy'] * 100

        nat_accs.append(nat_acc)
        rob_accs.append(rob_acc)
    nat_acc = np.mean(nat_accs)
    rob_acc = np.mean(rob_accs)

    with open('stats.json', 'w') as f:
        json.dump({'StandardAcc': nat_acc, 'RobustAcc': rob_acc}, f)
