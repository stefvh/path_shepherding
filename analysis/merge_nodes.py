import os

import numpy as np
import pandas as pd
import glob
import re
import click

from utils import sorting



@click.command()
@click.option('--folder', type=str, default="data/")
@click.option('--exp_id', type=str, default="2023_03_28_16_34_42")
def run(nodes="node0,node1,node2",
        folder="../data/",
        exp_id="2023_03_28_16_34_42"):
    folder += exp_id + "/"
    nodes = nodes.split(",")

    files = glob.glob(f'{folder}{nodes[0]}/merged/*_perseed.pkl')
    if len(files) == 0:
        files = glob.glob(f'{folder}{nodes[1]}/merged/*_perseed.pkl')
    if len(files) == 0:
        files = glob.glob(f'{folder}{nodes[2]}/merged/*_perseed.pkl')

    for file in files:
        frames = []
        seed_start = 0
        fs = file.split("/")
        filename = fs[-1]
        b_merge = True
        for node in nodes:
            filepath = f'{folder}{node}/merged/{filename}'
            if os.path.exists(filepath):
                try:
                    df = pd.read_pickle(filepath)
                    # df['seed'] = df['seed'].add(seed_start)
                    # seed_start = df['seed'].max() + 1
                    frames.append(df)
                except:
                    b_merge = False
                    print(f'!! NOT ANALYSING {node}/merged/{filename} !!')
        if b_merge:
            df_concat = pd.concat(frames)
            df_concat = df_concat.sort_values(by=['NR', 'NH', 'seed'])
            prefix = filename.split("perseed.pkl")[0]
            prefix = prefix.split('_')[0]
            # prefix = '_'.join(prefix.split('_')[1:-1])
            df_concat.to_pickle(f'{folder}{prefix}_perseed.pkl')

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
            df_averaged.to_pickle(f'{folder}{prefix}.pkl')


run()
