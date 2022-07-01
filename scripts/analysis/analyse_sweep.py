from information_extraction.analysis.tables import get_wandb_tables, print_latex_table, aggregate_tables
from information_extraction.evaluation import Evaluator
from information_extraction.data.metrics import compute_f1, compute_exact

SWEEP_ID = 'zax41f1p'


def print_full_overview(evaluator, tables):
    df_agg = aggregate_tables(evaluator, tables, run_name_parts=[1])
    df_agg.index.name = 'Model'

    print_latex_table(df_agg, caption='Performance of our extraction models, compared to several baselines',
                      label='table:full_performance', highlight_axis='index')


def main():
    tables = get_wandb_tables(SWEEP_ID)

    evaluator = Evaluator({'f1': compute_f1, 'em': compute_exact})
    print_full_overview(evaluator, tables)


if __name__ == '__main__':
    main()
