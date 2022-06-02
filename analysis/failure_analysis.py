from collections import defaultdict
from pathlib import Path
from pprint import pprint
import os

import numpy as np
import pandas as pd

from tables import get_wandb_tables
from metrics import compute_f1, compute_exact


DOMAINS = {
    'auto': ['engine', 'fuel_economy', 'model', 'price'],
    'book': ['author', 'isbn_13', 'publication_date', 'publisher', 'title'],
    'camera': ['manufacturer', 'model', 'price'],
    'job': ['company', 'date_posted', 'location', 'title'],
    'movie': ['director', 'genre', 'mpaa_rating', 'title'],
    'nbaplayer': ['height', 'name', 'team', 'weight'],
    'restaurant': ['address', 'cuisine', 'name', 'phone'],
    'university': ['name', 'phone', 'website', 'type'],
}

SWEEP_ID = 'bpss08tf'
TABLES_ROOT = Path('~/Data/Tables').expanduser()
INPUT_DIR = Path('~/Data/SWDE').expanduser()

THRESHOLD = 0.5


def segments_to_documents(instances: pd.DataFrame) -> pd.DataFrame:
    results = []

    for doc_id, doc_instances in instances.groupby('doc_id'):
        doc_predictions = {'doc_id': doc_id}

        scores = defaultdict(list)

        for feature, predictions in doc_instances.groupby('feature'):
            not_null_predictions = predictions[~((predictions['predicted'] == '') |
                                                 pd.isnull(predictions['predicted']))]

            if not_null_predictions.empty:
                # We could not find any segment with an answer, so we choose the prediction with the lowest confidence
                best_prediction = predictions.loc[predictions['score'].astype(float).idxmin()]
                best_prediction['score'] = 1 - best_prediction['score']
            else:
                # We were able to find segments that contain an answer, so we pick the one with the highest confidence
                best_prediction = not_null_predictions.loc[not_null_predictions['score'].astype(float).idxmax()]

            doc_predictions[feature] = best_prediction['predicted']
            doc_predictions[f'{feature}/score'] = best_prediction['score']

            for metric_name in ['f1', 'em']:
                doc_predictions[f'{feature}/{metric_name}'] = float(best_prediction[metric_name])
                scores[metric_name].append(float(best_prediction[metric_name]))

        for metric_name, metric_scores in scores.items():
            doc_predictions[metric_name] = np.mean(metric_scores)

        results.append(doc_predictions)

    return pd.DataFrame(results)


def read_ground_truth_file(vertical: str, website: str, attribute: str):
    filename = f'{vertical}-{website}-{attribute}.txt'
    with open(INPUT_DIR / 'groundtruth' / vertical / filename) as _file:
        entries = _file.read().splitlines()[2:]

    data = {}

    for entry in entries:
        parts = entry.split('\t')
        data[parts[0]] = parts[2:]

    return data


def get_ground_truths(vertical: str, website: str) -> dict:
    ground_truths = defaultdict(dict)

    for attribute in DOMAINS[vertical]:
        sub_data = read_ground_truth_file(vertical, website, attribute)

        for key, values in sub_data.items():
            ground_truths[key][attribute] = [
                '' if value == '<NULL>' else value
                for value in values
            ]

    return ground_truths


def compare_against_ground_truth(results_df, ground_truths, features):
    scores = {'f1': defaultdict(list), 'em': defaultdict(list)}

    new_rows = []

    for _, row in results_df.iterrows():
        _, website, doc_id = row['doc_id'].split('/')

        new_row = {
            'doc_id': row['doc_id']
        }

        for feature in features:
            a_pred = row[feature]
            a_gold = max(ground_truths[website][doc_id][feature], key=lambda a: compute_f1(a, a_pred))

            scores['f1'][feature].append(compute_f1(a_gold, a_pred))
            scores['em'][feature].append(compute_exact(a_gold, a_pred))

            new_row[f'{feature}/pred'] = a_pred
            new_row[f'{feature}/true'] = a_gold
            new_row[f'{feature}/em'] = compute_exact(a_gold, a_pred)
            new_row[f'{feature}/f1'] = compute_f1(a_gold, a_pred)

        new_rows.append(new_row)

    for feature in features:
        scores['f1'][feature] = np.mean(scores['f1'][feature])
        scores['em'][feature] = np.mean(scores['em'][feature])

    for metric in scores:
        scores[metric]['total'] = np.mean(list(scores[metric].values()))
        scores[metric] = dict(scores[metric])

    return scores, pd.DataFrame(new_rows)


def process_results(df_results, features):
    df_results = df_results.copy()

    for feature in features:
        df_results.loc[df_results[f'{feature}/score'] == '', f'{feature}/score'] = 0
        df_results.loc[df_results[f'{feature}/score'].astype(float) < THRESHOLD, feature] = ''

    return df_results


def main():
    dfs = get_wandb_tables(TABLES_ROOT, SWEEP_ID, table_type='documents')

    avg_scores = {'f1': [], 'em': []}

    for run_name, df in dfs.items():
        # df = segments_to_documents(df)

        domain_name = run_name.split('-')[1]

        ground_truths = {
            (website := identifier.split('-')[1].split('(')[0]): get_ground_truths(domain_name, website)
            for identifier in os.listdir(INPUT_DIR / domain_name)
        }
        df_results = process_results(df, DOMAINS[domain_name])

        scores, new_df = compare_against_ground_truth(df_results, ground_truths, DOMAINS[domain_name])
        pprint(scores)

        new_df.to_csv(f'results-{run_name}.csv', index=False)

        for key in scores:
            avg_scores[key].append(scores[key]['total'])

        print(f'{domain_name}: f1={scores["f1"]["total"]:.2f}, em={scores["em"]["total"]:.2f}')

    for key in avg_scores:
        print(f'Avg {key}: {np.mean(avg_scores[key]):.2f}')


if __name__ == '__main__':
    main()
