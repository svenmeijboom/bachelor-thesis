import json
import os
from pathlib import Path
import shutil
from typing import Dict

import pandas as pd
import wandb


def get_wandb_tables(tables_root: Path, sweep_id: str, table_type: str = 'instances') -> Dict[str, pd.DataFrame]:
    tables_dir = tables_root / sweep_id

    if os.path.exists(tables_dir):
        return {
            filename[:-4]: pd.read_csv(tables_dir / filename, dtype=str).fillna('')
            for filename in os.listdir(tables_dir)
            if filename.endswith('.csv')
        }

    os.makedirs(tables_dir)

    api = wandb.Api()
    sweep = api.sweep(f'information_extraction/{sweep_id}')

    dfs = {}

    for run in sweep.runs:
        for file in run.files():
            if file.name.startswith(f'media/table/{table_type}/test_'):
                print(f'Downloading {run.name}: {file.name}')
                with file.download(root=str(tables_dir), replace=True) as _file:
                    table_data = json.load(_file)

                df = pd.DataFrame(**table_data)
                df.to_csv(tables_dir / f'{run.name}.csv', index=False)
                dfs[run.name] = df

    shutil.rmtree(tables_dir / 'media')

    return dfs
