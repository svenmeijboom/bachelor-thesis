import json
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Union

import pandas as pd
from pandas.io.formats.style import Styler
import wandb

from information_extraction.evaluation import Evaluator
from information_extraction.data import DOMAINS
from information_extraction.config import DATA_DIR, WANDB_PROJECT


DEFAULT_TABLES_ROOT = DATA_DIR / 'Tables'


def get_wandb_tables(sweep_id: str, tables_root: Optional[Union[str, Path]] = None,
                     table_type: str = 'documents') -> Dict[str, pd.DataFrame]:
    if tables_root is None:
        tables_root = DEFAULT_TABLES_ROOT
    else:
        tables_root = Path(tables_root)

    tables_dir = tables_root / sweep_id / table_type

    if os.path.exists(tables_dir):
        return {
            filename[:-4]: pd.read_csv(tables_dir / filename).fillna('')
            for filename in os.listdir(tables_dir)
            if filename.endswith('.csv')
        }

    os.makedirs(tables_dir)

    api = wandb.Api()
    sweep = api.sweep(f'{WANDB_PROJECT}/{sweep_id}')

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


def print_latex_table(table: Union[pd.DataFrame, Styler], caption: str, label: str,
                      highlight_axis: Optional[str] = None, format_float: bool = True):
    if isinstance(table, pd.DataFrame):
        formatted = table.style
    else:
        formatted = table

    if format_float:
        formatted = formatted.format(precision=2)

    if highlight_axis is not None:
        formatted = formatted.highlight_max(axis=highlight_axis, props='textbf:--rwrap;')

    print(formatted.to_latex(multicol_align='c', caption=caption, label=label, position_float='centering',
                             hrules=True, clines='skip-last;data'))


def aggregate_tables(
    evaluator: Evaluator, tables: dict[str, pd.DataFrame], run_name_parts: Optional[List[int]] = None,
    per_vertical: bool = False, per_website: bool = False, per_attribute: bool = False
) -> Union[pd.Series, pd.DataFrame]:

    group_values = []

    if run_name_parts is not None:
        group_values.extend(f'run_name_{i}' for i in run_name_parts)
    if per_vertical:
        group_values.append('vertical')
    if per_website:
        group_values.append('website')
    if per_attribute:
        group_values.append('attribute')

    rows = []
    for run_name, table in tables.items():
        table = table.copy()
        table[['vertical', 'website', 'doc_id']] = table['doc_id'].str.split('/', expand=True)

        row_base = {
            f'run_name_{i}': part
            for i, part in enumerate(run_name.split('-'))
        }

        for _, row in table.iterrows():
            for attribute in DOMAINS[row['vertical']]:
                if f'{attribute}/true' not in row:
                    continue

                new_row = {
                    **row_base,
                    'vertical': row['vertical'],
                    'website': row['website'],
                    'attribute': attribute,
                }

                for field in ['true', 'pred', 'em', 'f1']:
                    new_row[f'result/{field}'] = row[f'{attribute}/{field}']

                rows.append(new_row)

    df = pd.DataFrame(rows)

    if not group_values:
        return evaluator.get_document_summary(df)

    rows = []
    for name, sub_df in df.sort_values(group_values).groupby(group_values):
        summary = evaluator.get_document_summary(sub_df)
        summary.name = name
        rows.append(summary)

    df = pd.DataFrame(rows)

    if len(group_values) > 1:
        df.index = pd.MultiIndex.from_tuples(df.index)

    df.columns = pd.MultiIndex.from_tuples(df.columns)

    return df
