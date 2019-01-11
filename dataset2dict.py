#!/usr/bin/env python
import pandas as pd
import dill
import click
import os
from collections import defaultdict
import datetime
import numpy as np
import multiprocessing as MP

_df_indexed = None

TIME_SCALE = 24 * 60 * 60


def _column_worker(params):
    idx, success_prob = params
    if idx == 0:
        return _df_indexed.groupby(level=[0, 1]).p_recall.transform(
            lambda x: np.cumsum([0] + [1 if r >= success_prob else 0 for r in x])[:-1])
    elif idx == 1:
        return _df_indexed.groupby(level=[0, 1]).p_recall.transform(
            lambda x: np.cumsum([0] + [1 if r < success_prob else 0 for r in x][:-1]))
    elif idx == 2:
        return _df_indexed.groupby(level=[0, 1]).p_recall.transform(
            lambda x: np.arange(len(x)))


def add_user_lexeme_columns(success_prob):
    """Adds 'n_correct', 'n_wrong', 'n_total' column to the data-frame."""

    if "history_seen" in _df_indexed.columns:
         _df_indexed['n_correct'] = _df_indexed['history_correct']
         _df_indexed['n_total'] = _df_indexed['history_seen']
         _df_indexed['n_wrong']= _df_indexed['n_total']-_df_indexed['n_correct']
         return
    print("No meta info on total number of exercises")
    with MP.Pool(3) as pool:
        n_correct, n_wrong, n_total = pool.map(_column_worker,
                                               [(ii, success_prob)
                                                for ii in range(3)])

    _df_indexed['n_correct'] = n_correct
    _df_indexed['n_wrong']   = n_wrong
    _df_indexed['n_total']   = n_total


def convert_csv_to_dict(csv_path,
                        dictionary_output,
                        max_days,
                        success_prob,
                        force=False):
    """Pre-process the CSV file and save as a dictionary."""

    if os.path.exists(dictionary_output) and not force:
        print('{} already exists and not being forced to over-write it.'.format(dictionary_output))
        return

    start_time = datetime.datetime.now()

    def elapsed():
        return (datetime.datetime.now() - start_time).seconds

    df = pd.read_csv(csv_path)

    # Hack to avoid passing df_indexed as a argument to the worker function

    if 'n_correct' not in df.columns:
        print('Calculating n_wrong, n_correct and n_total')
        global _df_indexed
        # Only mergesort is stable sort.
        _df_indexed = df.set_index(['user_id', 'lexeme_id']).sort_values('timestamp').sort_index(kind='mergesort')

        add_user_lexeme_columns(success_prob)

        df = _df_indexed.reset_index().sort_values('timestamp')

    # Drop all intervals larger than 30 days.
    df = df[df.delta < TIME_SCALE * max_days]

    # results = dill.load(open(results_path, 'rb'))

    # map_lexeme        = results['map_lexeme']
    # alpha             = results['alpha']
    # beta              = results['beta']
    # lexeme_difficulty = results['lexeme_difficulty']

    # n_0 = [lexeme_difficulty[map_lexeme[x]] for x in df.lexeme_id]
    # df['n_0'] = np.abs(n_0)
    # df['n_t'] = df['n_0'] * (alpha[0] ** df['n_correct']) * (beta[0] ** df['n_wrong'])
    # df['m_t'] = np.exp(-df['n_t'] * df['delta'] / TIME_SCALE)

    op_dict = defaultdict(lambda: defaultdict(lambda: []))

    for ii in range(df.shape[0]):
        row = df.iloc[ii]
        u_id, l_id = row.user_id, row.lexeme_id
        delta = row.delta / TIME_SCALE

        op_dict[u_id][l_id].append({
            'delta_scaled' : delta,
            'n_wrong'      : row.n_wrong,
            'n_correct'    : row.n_correct,
            'p_recall'     : row.p_recall,
            # 'n_0'          : row.n_0,
            'timestamp'    : row.timestamp,
            # 'm_t'          : row.m_t,
            # 'n_t'          : row.n_t,
            'user_id'      : u_id,
            'lexeme_id'    : l_id
        })

        if ii % 100000 == 0:
            print('Done {:0.2f}%\tElapsed = {} sec'.format(100. * ii / df.shape[0], elapsed()))

    print('Writing {} ...'.format(dictionary_output))
    dill.dump(op_dict, open(dictionary_output, 'wb'))
    print('Done.')


@click.command()
@click.argument('csv_file')
@click.argument('output_dill')
@click.option('--success_prob', 'success_prob', default=0.6, type=float, help='At what recall probability is the trial considered successful.')
@click.option('--max_days', 'max_days', default=30, help='Maximum number of days before a revision.')
@click.option('--force/--no-force', 'force', default=False, help='Force overwrite of existing files.')
def run(csv_file, output_dill, success_prob, max_days, force):
    """Converts the CSV_FILE from Duolingo format to a dictionary and saves it in OUTPUT_DILL
    after reading the results of Half-Life regression from RESULTS_PATH."""
    convert_csv_to_dict(csv_path=csv_file, dictionary_output=output_dill,
                        max_days=max_days, success_prob=success_prob,
                        force=force)


if __name__ == '__main__':
    run()
