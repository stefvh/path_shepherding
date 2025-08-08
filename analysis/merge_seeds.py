import click
import pandas as pd
import glob
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", type=str, default="")
    parser.add_argument("--out_folder", type=str, default="")

    args = parser.parse_args()
    return args


def merge(in_folder="",
          out_folder=""):
    filenames = glob.glob(f'{in_folder}/*.pkl')
    metrics = set()
    for filename in filenames:
        parts = filename.split('_')
        metric = "_".join(parts[:-1])
        metrics.add(metric)

    for metric in metrics:
        suffix = metric.split('/')[-1]

        frames = []
        files = glob.glob(f'{metric}_*')
        for file in files:
            df = pd.read_pickle(file)
            frames.append(df)
        df_concat = pd.concat(frames)
        df_concat = df_concat.sort_values(by=['NR', 'NH', 'seed'])
        df_concat.to_pickle(f'{out_folder}{suffix}_perseed.pkl')

        data_averaged = {
            'Mean': [], 'Std': [],
        }
        for column in df_concat.columns[2:]:
            data_averaged[column] = []
        for name, group in df_concat.groupby(list(df.columns[2:])):
            data_averaged['Mean'].append(group['Measurement'].mean())
            data_averaged['Std'].append(group['Measurement'].std())
            for c_i, column in enumerate(df.columns[2:]):
                data_averaged[column].append(name[c_i])
        df_averaged = pd.DataFrame(data=data_averaged)
        df_averaged.to_pickle(f'{out_folder}{suffix}.pkl')


if __name__ == '__main__':
    args = parse_args()
    merge(in_folder=args.in_folder,
          out_folder=args.out_folder)
