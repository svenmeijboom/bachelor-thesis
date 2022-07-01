import pandas as pd

from information_extraction.analysis.mappings import CLOSED_TO_OPEN_MAPPING
from information_extraction.analysis.tables import print_latex_table


def format_val(val):
    if val is None or val == '-':
        return '-'
    elif len(val) == 1:
        return f'\\texttt{{{val[0]}}}'
    else:
        return '\\vtop{{{}}}'.format(' '.join([
            f'\\hbox{{\\strut \\texttt{{{x}}}}}' for x in val
        ]))


if __name__ == '__main__':
    names = {
        'movie': 'Movie',
        'nbaplayer': 'NBA player',
        'university': 'University',
    }

    for key, mapping in CLOSED_TO_OPEN_MAPPING.items():
        df = pd.DataFrame(mapping).T.fillna('-')
        df.index.name = 'Website'

        table = df.style.format(format_val).format_index('\\verb|{}|', axis='columns')
        print_latex_table(table, f'ClosedIE to OpenIE mapping for the {names[key]} domain',
                          f'table:{key}_mapping', format_float=False)
