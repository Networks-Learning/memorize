"""Script to convert Duolingo dataset to have one entry per session with the student."""
import click
import pandas as pd
import numpy as np


@click.command()
@click.argument('in_csv')
@click.argument('out_csv')
def cmd(in_csv, out_csv):
    """Read in the raw data from IN_CSV, process it to keep only one entry
    per session with binary recall signal, and save the output in OUT_CSV."""

    print('Reading input ...')
    raw_df = pd.read_csv(in_csv)

    print('Processing data ...')
    raw_df.sort_values("timestamp", inplace=True)

    sess_seen = raw_df.groupby("lexeme_id").mean()['session_seen']
    first_review = raw_df.groupby(["lexeme_id", "user_id"]).first()
    first_review['history_seen'] /= sess_seen
    first_review['history_correct'] /= sess_seen

    new_df = raw_df.copy()
    new_df['session_correct'] = (raw_df['session_correct'] / raw_df['session_seen']) > 0.99
    new_df['session_seen'] = 1
    new_df.sort_values(["user_id", "lexeme_id", "timestamp"], inplace=True)
    new_df.reset_index(inplace=True)
    new_df.drop("index", axis=1, inplace=True)

    current_u = None
    current_l = None
    result = []
    for _index, row in new_df.iterrows():
        if current_u != row['user_id'] or current_l != row['lexeme_id']:
            history_seen = int(np.round(row['history_seen'] / sess_seen[row['lexeme_id']]))
            history_correct = int(np.round(row['history_correct'] / sess_seen[row['lexeme_id']]))
            row['history_seen'] = history_seen
            row['history_correct'] = history_correct
        else:
            row['history_seen'] = history_seen
            row['history_correct'] = history_correct
        result.append(row)
        current_u = row['user_id']
        current_l = row['lexeme_id']
        history_seen += row['session_seen']
        history_correct += row['session_correct']

    cleaned_df = pd.DataFrame(result)
    cleaned_df['session_correct'] = cleaned_df['session_correct'].astype("int")

    print('Writing output ...')
    cleaned_df.to_csv(out_csv, index=False)
    print('Done.')


if __name__ == '__main__':
    cmd()
