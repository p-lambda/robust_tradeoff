import argparse
import pandas as pd
from pathlib import Path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='compile stats for sample sizes plot')
    parser.add_argument('-p','--paths', nargs='+', help='paths to process', required=True)
    args = parser.parse_args()

    res = []
    for bundle_path in args.paths:
        bundle_path = Path(bundle_path)
        if 'clean' in str(bundle_path):
            model_dir = bundle_path / 'clean_models'
        else:
            model_dir = bundle_path / 'models'
        for ckp_dir in model_dir.iterdir():
            stats_csv = pd.read_csv(str(ckp_dir / 'stats_eval.csv'))
            res_dict = stats_csv.iloc[-1].to_dict()
            res_dict['name'] = str(ckp_dir)
            res_dict['run_name'] = str(bundle_path)
            res.append(res_dict)
    df = pd.DataFrame(res)
    df.to_csv('stats.csv', index=False)

