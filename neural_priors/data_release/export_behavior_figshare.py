"""Export all trial-wise behaviour of the released participants as ONE long table
for figshare, plus a codebook README.

Rows = trials of three tasks, per participant x session x block:
- examples:   15 labelled example displays before each block (self-paced viewing)
- practice:   30 estimation trials with feedback before each block (no fMRI)
- estimation: 4 runs x 30 trials per block, in the scanner (the analysed task)

Only the released participants (the paper's sample) are exported; subject labels
match OpenNeuro. The in-scanner rows are checked against Subject.get_behavioral_data()
(the table every analysis uses).

Usage:
    python -m neural_priors.data_release.export_behavior_figshare \
        --bids_folder /data/ds-neuralpriors --out_dir <dir>
"""
import argparse
import os
import os.path as op

import numpy as np
import pandas as pd

from neural_priors.data_release.release import get_subjects

COLUMNS = ['subject', 'session', 'block', 'range', 'task', 'run', 'trial', 'n', 'response',
           'error', 'response_time', 'start_marker_position']


def read_log(bids_folder, subject, session, task, run):
    fn = op.join(bids_folder, 'sourcedata', 'behavior', f'sub-{subject}', f'ses-{session}',
                 f'sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv')
    return numeric_responses(pd.read_csv(fn, sep='\t'))


def numeric_responses(d):
    """Some logs contain key presses (e.g. 'escape') in `response` on trial 0
    (instruction screens), which makes the column textual. Coerce; any
    non-numeric response on a real trial is an error."""
    num = pd.to_numeric(d['response'], errors='coerce')
    bad = num.isna() & d['response'].notna() & (d['trial_nr'] > 0)
    assert not bad.any(), d.loc[bad]
    d['response'] = num
    return d


def first(d, event_type):
    return d[(d['event_type'] == event_type) & (d['trial_nr'] > 0)].groupby('trial_nr').first()


def estimation_trials(d):
    stim, resp, fb = first(d, 'stimulus'), first(d, 'response'), first(d, 'feedback')
    out = pd.DataFrame({'trial': stim.index.astype(int), 'n': stim['n'].values,
                        'start_marker_position': stim['start_marker_position'].values})
    out['response'] = fb['response'].reindex(stim.index).values
    # The click ends the response screen and starts the feedback screen.
    rt = (fb['onset'].reindex(stim.index) - resp['onset'].reindex(stim.index)).values
    out['response_time'] = np.where(out['response'].notna().values, rt, np.nan)
    return out


def practice_trials(d):
    stim, resp, fb = first(d, 'stimulus'), first(d, 'response'), first(d, 'feedback')
    out = pd.DataFrame({'trial': stim.index.astype(int), 'n': stim['n'].values})
    out['response'] = fb['response'].reindex(stim.index).values
    out['response_time'] = (fb['onset'].reindex(stim.index) - resp['onset'].reindex(stim.index)).values
    out['start_marker_position'] = np.nan   # not logged in the practice task
    return out


def example_trials(d):
    stim = d[(d['event_type'] == 'stim') & (d['trial_nr'] > 0)].groupby('trial_nr').first()
    return pd.DataFrame({'trial': stim.index.astype(int), 'n': stim['n'].values})


def export(bids_folder):
    rows = []
    for subject in get_subjects():
        for session in [1, 2]:
            for block, runs in [(1, range(1, 5)), (2, range(5, 9))]:
                est = []
                for run in runs:
                    t = estimation_trials(read_log(bids_folder, subject, session, 'estimation_task', run))
                    assert len(t) == 30, (subject, session, run, len(t))
                    est.append(t.assign(task='estimation', run=run))
                est = pd.concat(est)
                rng = 'wide' if (est['n'] > 25).any() else 'narrow'
                first_run = runs[0]
                prac = practice_trials(read_log(bids_folder, subject, session, 'feedback', first_run))
                exam = example_trials(read_log(bids_folder, subject, session, 'examples', first_run))
                assert len(prac) == 30 and len(exam) == 15, (subject, session, block, len(prac), len(exam))
                lo, hi = (10, 25) if rng == 'narrow' else (10, 40)
                for part in (exam, prac, est):
                    assert part['n'].between(lo, hi).all(), (subject, session, block)
                block_df = pd.concat([exam.assign(task='examples', run=np.nan),
                                      prac.assign(task='practice', run=np.nan), est])
                rows.append(block_df.assign(subject=f'sub-{subject}', session=session, block=block, range=rng))

    df = pd.concat(rows, ignore_index=True)
    df['error'] = df['response'] - df['n']
    for c in ['run', 'n', 'response', 'error', 'start_marker_position']:
        df[c] = df[c].astype('Int64')
    df['response_time'] = df['response_time'].round(4)
    task_order = pd.Categorical(df['task'], ['examples', 'practice', 'estimation'], ordered=True)
    df = df.assign(_t=task_order).sort_values(['subject', 'session', 'block', '_t', 'run', 'trial'],
                                              kind='stable').drop(columns='_t')
    return df[COLUMNS].reset_index(drop=True)


def check_against_analysis_table(df, bids_folder):
    from neural_priors.utils.data import get_all_behavioral_data
    ref = get_all_behavioral_data(bids_folder=bids_folder).reset_index()
    est = df[df['task'] == 'estimation']
    assert len(ref) == len(est), (len(ref), len(est))
    ref = ref.assign(subject='sub-' + ref['subject'].astype(str)).sort_values(['subject', 'session', 'run', 'trial_nr'])
    est = est.sort_values(['subject', 'session', 'run', 'trial'])
    assert (ref['subject'].values == est['subject'].values).all()
    assert np.array_equal(ref['n'].values.astype(float), est['n'].astype(float).values)
    assert np.allclose(ref['response'].values.astype(float), est['response'].astype(float).values, equal_nan=True)
    assert (ref['range'].values == est['range'].values).all()
    print(f'OK: {len(est)} in-scanner trials identical to get_all_behavioral_data()')


CODEBOOK = """# Behavioural data: {title}

{authors}

Paper (preprint): {preprint} · Code: {code} · fMRI data: OpenNeuro (see paper)

One table, `neural_priors_behavior.tsv` (tab-separated, `n/a` = missing), one row per trial, for all {n_sub} participants of the paper (labels identical to the OpenNeuro dataset). {n_rows} rows.

## Design

Participants estimated the number of dots in briefly shown dot clouds. Each participant did two sessions; each session had two blocks, one in which numerosities were drawn uniformly from 10-25 (`narrow`) and one from 10-40 (`wide`). Each block consisted of, in this order:

1. `examples`: 15 dot clouds shown together with their numerosity (the first two are the range limits). No response.
2. `practice`: 30 estimation trials with feedback (the true number was shown after the response), outside the fMRI data. No response deadline.
3. `estimation`: 4 fMRI runs of 30 trials each (runs 1-4 in block 1, 5-8 in block 2), without feedback. Dot cloud 0.6 s, delay 4-6 s, response slider for at most 3 s. These are the trials analysed in the paper.

Responses were given by moving a marker along a slider and clicking.

## Columns

| column | description |
|---|---|
| subject | participant label (`sub-XX`) |
| session | 1 or 2 |
| block | 1 or 2 (order within the session) |
| range | `narrow` (10-25) or `wide` (10-40) |
| task | `examples`, `practice` or `estimation` (see above) |
| run | fMRI run (1-8) for `estimation` trials; `n/a` otherwise |
| trial | trial number within the run (estimation), practice set or example set |
| n | number of dots shown |
| response | reported numerosity; `n/a` for examples and for estimation trials without a response within 3 s |
| error | response - n |
| response_time | seconds from response-slider onset to the click; `n/a` if no response |
| start_marker_position | initial (random) slider-marker position, in numerosity units (estimation trials only) |
"""


def main(bids_folder, out_dir):
    from neural_priors.data_release.write_metadata import AUTHORS, CODE, PREPRINT, TITLE
    os.makedirs(out_dir, exist_ok=True)
    df = export(bids_folder)
    check_against_analysis_table(df, bids_folder)
    fn = op.join(out_dir, 'neural_priors_behavior.tsv')
    df.to_csv(fn, sep='\t', index=False, na_rep='n/a')
    with open(op.join(out_dir, 'README.md'), 'w') as f:
        f.write(CODEBOOK.format(title=TITLE, authors=', '.join(AUTHORS), preprint=PREPRINT, code=CODE,
                                n_sub=df['subject'].nunique(), n_rows=len(df)))
    print(df.groupby(['task']).size())
    print(f'wrote {fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--out_dir', default='/data/ds-neuralpriors/derivatives/figshare_behavior')
    args = parser.parse_args()
    main(args.bids_folder, args.out_dir)
