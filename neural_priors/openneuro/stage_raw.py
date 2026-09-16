"""Stage the raw BIDS data of one subject for the OpenNeuro release.

- BOLD + (synthesised PEPOLAR) EPI fieldmaps: header text cleared, gzipped,
  byte-identical voxel data, into TARGET.
- T1w: header text cleared + gzipped into WORK/undefaced (NOT into TARGET);
  deface_subject.py writes the defaced version into TARGET.
- events.tsv: rebuilt from the PsychoPy logs (sourcedata/behavior) with
  duration / response time columns, and checked against the events.tsv files
  the analyses used (same onsets, numerosities and responses).
- Nothing else (no physio, no eye tracking, no raw behaviour, no FLAIR).

Usage: python stage_raw.py 01
"""
import argparse
import glob
import json
import os.path as op
import re

import numpy as np
import pandas as pd

from neural_priors.openneuro.release import (SOURCE, TARGET, WORK, TASK, TASK_OLD,
                                             copy_nifti_clean_header, get_subjects,
                                             makedirs_for, verify_same_image)

SESSIONS = [1, 2]
RUNS = range(1, 9)


def stage_bold_and_fmaps(subject, session):
    src_sub = op.join(SOURCE, f'sub-{subject}', f'ses-{session}')
    for run in RUNS:
        src = op.join(src_sub, 'func', f'sub-{subject}_ses-{session}_task-{TASK_OLD}_run-{run}_bold.nii')
        dst = op.join(TARGET, f'sub-{subject}', f'ses-{session}', 'func',
                      f'sub-{subject}_ses-{session}_task-{TASK}_run-{run}_bold.nii.gz')
        copy_nifti_clean_header(src, dst)
        verify_same_image(src, dst)
        with open(src.replace('.nii', '.json')) as f:
            meta = json.load(f)
        # Everything but the phase-encoding direction is identical across runs
        # and lives in the top-level task-<label>_bold.json.
        with open(dst.replace('.nii.gz', '.json'), 'w') as f:
            json.dump({'PhaseEncodingDirection': meta['PhaseEncodingDirection']}, f, indent=2)
        print(f'bold  {op.basename(dst)}')

    for src in sorted(glob.glob(op.join(src_sub, 'fmap', '*_epi.nii'))):
        dst = op.join(TARGET, f'sub-{subject}', f'ses-{session}', 'fmap', op.basename(src) + '.gz')
        copy_nifti_clean_header(src, dst)
        verify_same_image(src, dst)
        with open(src.replace('.nii', '.json')) as f:
            meta = json.load(f)
        m = re.fullmatch(rf'ses-{session}/func/sub-{subject}_ses-{session}_task-{TASK_OLD}_run-(\d)_bold\.nii',
                         meta['IntendedFor'])
        assert m, meta['IntendedFor']
        new = {'PhaseEncodingDirection': meta['PhaseEncodingDirection'],
               'TotalReadoutTime': meta['TotalReadoutTime'],
               'IntendedFor': f'bids::sub-{subject}/ses-{session}/func/'
                              f'sub-{subject}_ses-{session}_task-{TASK}_run-{m.group(1)}_bold.nii.gz'}
        with open(dst.replace('.nii.gz', '.json'), 'w') as f:
            json.dump(new, f, indent=2)
        print(f'fmap  {op.basename(dst)}')


def stage_t1w(subject, session):
    srcs = sorted(glob.glob(op.join(SOURCE, f'sub-{subject}', f'ses-{session}', 'anat', '*_T1w.nii')))
    assert srcs, f'no T1w for sub-{subject} ses-{session}'
    for src in srcs:
        dst = op.join(WORK, 'undefaced', 'raw', f'sub-{subject}', f'ses-{session}', 'anat', op.basename(src) + '.gz')
        copy_nifti_clean_header(src, dst)
        verify_same_image(src, dst)
        print(f'T1w (undefaced, WORK only)  {op.basename(dst)}')


def build_events(subject, session, run):
    fn = op.join(SOURCE, 'sourcedata', 'behavior', f'sub-{subject}', f'ses-{session}',
                 f'sub-{subject}_ses-{session}_task-estimation_task_run-{run}_events.tsv')
    d = pd.read_csv(fn, sep='\t')
    # Onsets relative to the trigger_2 scanner pulse (as in prepare/make_events_files.py)
    t0 = d.loc[d['event_type'] == 'trigger_2', 'onset'].iloc[0]
    d['onset'] = d['onset'] - t0
    d = d[(d['trial_nr'] > 0) & (d['onset'] > 0.0)]

    def one(event_type):
        return d[d['event_type'] == event_type].groupby('trial_nr').first()

    stim, resp, fb = one('stimulus'), one('response'), one('feedback')
    trials = stim.index.intersection(resp.index)[:30]
    assert len(trials) == 30, (fn, len(trials))
    rng = 'wide' if (stim['n'] > 25).any() else 'narrow'

    rows = []
    for t in trials:
        trial_nr = (run - 1) * 30 + int(t)
        response = fb.loc[t, 'response'] if t in fb.index else np.nan
        rt = fb.loc[t, 'response_time'] if t in fb.index else np.nan
        common = dict(trial_nr=trial_nr, range=rng, n=int(stim.loc[t, 'n']), response=response,
                      response_time=rt, start_marker_position=stim.loc[t, 'start_marker_position'])
        rows.append(dict(onset=stim.loc[t, 'onset'], duration=stim.loc[t, 'duration'],
                         trial_type='stimulus', **common))
        # Response screen lasts until the response is given (or times out),
        # i.e. until the feedback screen appears.
        resp_end = fb.loc[t, 'onset'] if t in fb.index else np.nan
        rows.append(dict(onset=resp.loc[t, 'onset'], duration=resp_end - resp.loc[t, 'onset'],
                         trial_type='response', **common))

    events = pd.DataFrame(rows).sort_values('onset', kind='stable').reset_index(drop=True)
    return events


def check_against_analysed_events(subject, session, run, events):
    old = pd.read_csv(op.join(SOURCE, f'sub-{subject}', f'ses-{session}', 'func',
                              f'sub-{subject}_ses-{session}_task-{TASK_OLD}_run-{run}_events.tsv'), sep='\t')
    assert len(old) == len(events), (len(old), len(events))
    assert (old['trial_type'].values == events['trial_type'].values).all()
    assert np.allclose(old['onset'].values, events['onset'].values, atol=1e-6)
    s_old, s_new = old[old.trial_type == 'stimulus'], events[events.trial_type == 'stimulus']
    assert np.array_equal(s_old['n'].values, s_new['n'].values)
    r_old, r_new = old[old.trial_type == 'response'], events[events.trial_type == 'response']
    assert np.allclose(r_old['response'].values.astype(float), r_new['response'].values.astype(float),
                       equal_nan=True)
    assert np.array_equal(old['trial_nr'].values, events['trial_nr'].values)


def stage_events(subject, session):
    for run in RUNS:
        events = build_events(subject, session, run)
        check_against_analysed_events(subject, session, run, events)
        dst = op.join(TARGET, f'sub-{subject}', f'ses-{session}', 'func',
                      f'sub-{subject}_ses-{session}_task-{TASK}_run-{run}_events.tsv')
        makedirs_for(dst)
        events['onset'] = events['onset'].round(4)
        events['duration'] = events['duration'].round(4)
        events['response_time'] = events['response_time'].round(4)
        events['response'] = events['response'].astype('Int64')
        events['start_marker_position'] = events['start_marker_position'].astype('Int64')
        events.to_csv(dst, sep='\t', index=False, na_rep='n/a')
        print(f'events {op.basename(dst)}')


def main(subject):
    subject = f'{int(subject):02d}'
    assert subject in get_subjects(), f'sub-{subject} is not part of the release'
    for session in SESSIONS:
        stage_events(subject, session)
        stage_t1w(subject, session)
        stage_bold_and_fmaps(subject, session)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject')
    args = parser.parse_args()
    main(args.subject)
