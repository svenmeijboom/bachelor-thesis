from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tables import get_wandb_tables


SWEEP_ID = '69bkix9k'
TABLES_ROOT = Path('~/Data/Tables').expanduser()


def main():
    dfs = get_wandb_tables(TABLES_ROOT, SWEEP_ID)

    for name, df in dfs.items():
        df = df.groupby('feature')[['em', 'f1']].agg(
            em=('em', 'mean'),
            f1=('f1', 'mean'),
        )
        df['run'] = name
        df['model'], df['context_size'], df['representation'] = name.split('-')
        dfs[name] = df

    df = (pd.concat(list(dfs.values()))
            .reset_index()
            .sort_values(['context_size', 'model', 'representation'],
                         ascending=[False, False, True]))

    features = sorted(df['feature'].unique())
    fig, axes = plt.subplots(len(features), figsize=(5, 15),
                             sharex='all', sharey='all')

    low = df[['em', 'f1']].min().min()
    high = df[['em', 'f1']].max().max()
    offset = 0.1 * (high - low)

    low, high = low - offset, high + offset

    for feature, ax in zip(features, axes):
        df[df['feature'] == feature].plot.barh(x='run', ax=ax, ylim=(max(low, 0), min(high, 1)))
        ax.set_title(feature)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend(loc='lower right')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
