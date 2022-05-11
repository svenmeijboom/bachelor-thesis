from collections import OrderedDict
import json
import os
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import wandb

from tables import get_wandb_tables


SWEEPS = {
    'bert': '6yv4y9t6',
    't5': '7j0wjjg2',
}

TABLES_ROOT = Path('~/Data/Tables').expanduser()

LABELS = OrderedDict({
    'text': 'text',
    'html-base': 'html',
    'html-id': 'html-i',
    'html-class': 'html-c',
    'html-id-class': 'html-ic',
    'html-id-class-subset': 'html-ic-sub',
    'html-id-expanded': 'html-i-*',
    'html-class-expanded': 'html-c-*',
    'html-id-class-expanded': 'html-ic-*',
    'html-id-class-expanded-subset': 'html-ic-*-sub',
})
LABEL_INDEX = list(LABELS.values())


def download_tables(sweep_id: str):
    tables_dir = TABLES_ROOT / sweep_id

    os.makedirs(tables_dir)

    api = wandb.Api()
    sweep = api.sweep(f'information_extraction/{sweep_id}')

    for run in sweep.runs:
        for file in run.files():
            if file.name.startswith('media/table/instances/test_'):
                print(f'{run.name}: {file.name}')
                with file.download(root=str(tables_dir), replace=True) as _file:
                    table_data = json.load(_file)
                    df = pd.DataFrame(**table_data)

                    df.to_csv(tables_dir / f'{run.name}.csv')

    shutil.rmtree(tables_dir / 'media')


def main():
    dfs = []

    for model, sweep_id in SWEEPS.items():
        model_dfs = get_wandb_tables(TABLES_ROOT, sweep_id)

        for name, df in model_dfs.items():
            df = df.groupby('feature')[['em', 'f1']].agg(
                em=('em', 'mean'),
                f1=('f1', 'mean'),
            )
            df['run'] = LABELS[name]
            df['model'] = model
            dfs.append(df)

    df = pd.concat(dfs).reset_index().sort_values('run', key=lambda col: col.apply(LABEL_INDEX.index), ascending=False)

    features = sorted(df['feature'].unique())
    fig, axes = plt.subplots(len(features), len(SWEEPS), figsize=(10, 15),
                             sharex='col', sharey='all')

    low = df.groupby('model')[['em', 'f1']].min().min(axis=1)
    high = df.groupby('model')[['em', 'f1']].max().max(axis=1)
    offset = 0.1 * (high - low)

    low, high = low - offset, high + offset

    for feature, row_axes in zip(features, axes):
        df_sub = df[df['feature'] == feature]

        for model, ax in zip(SWEEPS, row_axes):
            df_sub[df_sub['model'] == model].plot.barh(x='run', ax=ax, xlim=(max(low[model], 0), min(high[model], 1)))
            ax.set_title(f'{model.upper()}: {feature}')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.legend(loc='lower right')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
