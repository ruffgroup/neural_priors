"""Write the dataset-level BIDS metadata of the OpenNeuro release (run once).

Raw: dataset_description.json, README, CHANGES, participants.tsv/.json,
task-<label>_bold.json, task-<label>_events.json, T1w.json.
Derivatives: derivatives/fmriprep/{dataset_description.json, README,
desc-aseg_dseg.tsv, desc-aparcaseg_dseg.tsv, CITATION.md, CITATION.bib}.

participants.tsv is built from WORK/participants_source.tsv (the lab's table:
numeric participant_id, age, Sex), restricted to the released subjects.
"""
import collections
import glob
import json
import os.path as op
import re

import nibabel as nib

import pandas as pd

from neural_priors.data_release.release import (SOURCE, TARGET, WORK, TASK, TASK_NAME, TASK_OLD,
                                             copy_file, get_subjects)

TITLE = 'Distributed range adaptation in human parietal encoding of numbers'
AUTHORS = ['Arthur Prat-Carrabin', 'Gilles de Hollander', 'Saurabh Bedi', 'Samuel J. Gershman',
           'Christian C. Ruff']
PREPRINT = 'https://doi.org/10.1101/2025.09.25.675916'
CODE = 'https://github.com/ruffgroup/neural_priors'

TASK_DESCRIPTION = (
    'On each trial participants saw a cloud of dots (0.6 s), and after a jittered delay (4-6 s) '
    'reported its numerosity by moving a marker along a response slider and clicking '
    '(maximum 3 s). The slider with the chosen value then stayed on screen for 0.5 s (or "Too late!" was shown). Each session consisted of '
    'two blocks of 4 runs of 30 trials; in one block numerosities were drawn uniformly from the '
    'narrow range (10-25), in the other from the wide range (10-40). Before each block, participants '
    'saw 15 labelled examples and completed 30 practice trials with feedback (no fMRI data were '
    'recorded for these).')


def json_dump(obj, fn):
    with open(fn, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write('\n')


def check_bold_sidecars():
    """All released BOLD runs share every sidecar field except PhaseEncodingDirection."""
    common = None
    for sub in get_subjects():
        for fn in sorted(glob.glob(op.join(SOURCE, f'sub-{sub}', 'ses-[12]', 'func',
                                           f'*_task-{TASK_OLD}_run-*_bold.json'))):
            with open(fn) as f:
                meta = json.load(f)
            meta.pop('PhaseEncodingDirection')
            if common is None:
                common = meta
            assert meta == common, fn
    return common


def write_run_repetition_times():
    """RepetitionTime per run, from the NIfTI header (the scanner-reported value).

    The working dataset's sidecars all say 2.298 s, but for some sessions the scanner
    reported 2.286 s (or 2.292 s) in the image header; bids-validator requires the two
    to agree. SliceTiming stays at the nominal values used for preprocessing (all < TR).
    """
    trs = collections.Counter()
    for fn in sorted(glob.glob(op.join(TARGET, 'sub-*', 'ses-*', 'func', f'*_task-{TASK}_run-*_bold.nii.gz'))):
        hdr = nib.load(fn).header
        assert hdr.get_xyzt_units()[1] == 'sec', fn
        tr = round(float(hdr.get_zooms()[3]), 4)
        side = fn.replace('.nii.gz', '.json')
        with open(side) as f:
            meta = json.load(f)
        meta['RepetitionTime'] = tr
        json_dump(meta, side)
        trs[tr] += 1
    print('RepetitionTime per run:', dict(trs))


# Acquisition parameters as reported in the paper's Methods ("MRI data acquisition").
SCANNER = {
    'Manufacturer': 'Philips',
    'ManufacturersModelName': 'Achieva',
    'MagneticFieldStrength': 3,
    'ReceiveCoilName': '32-channel head coil',
    'InstitutionName': 'University of Zurich',
    'InstitutionalDepartmentName': 'Laboratory for Social and Neural Systems Research (SNS-Lab), '
                                   'Zurich Center for Neuroeconomics',
}
EPI_SEQUENCE = {
    'PulseSequenceType': 'Gradient-echo EPI',
    'ScanningSequence': 'GR',
    'SequenceVariant': 'NONE',
    'MRAcquisitionType': '2D',
    'EchoTime': 0.030,
    'FlipAngle': 90,
    'SliceThickness': 2.5,
    'SpacingBetweenSlices': 3.0,
}
T1W_SEQUENCE = {
    'PulseSequenceType': 'MPRAGE',
    'ScanningSequence': 'GR\\IR',
    'SequenceVariant': 'MP',
    'MRAcquisitionType': '3D',
    'RepetitionTimePreparation': 2.8,
    'InversionTime': 1.0986,
    'FlipAngle': 8,
    'ParallelReductionFactorInPlane': 2,
}
# EchoTime / RepetitionTimeExcitation of the T1w differ between sessions (PAR headers,
# see acquisition_parameters.tsv): written per image by write_t1w_sidecars().
PAR_TABLE = op.join(op.dirname(__file__), 'acquisition_parameters.tsv')
INSTRUCTIONS = ('Estimate the number of dots and report it with the slider. '
                'Your accuracy will influence your monetary bonus.')


def write_t1w_sidecars():
    """Per-image T1w timing from the PAR headers (constant within a session)."""
    par = pd.read_csv(PAR_TABLE, sep='\t', dtype={'subject': str})
    t1 = par[par['scan'] == 'T1w'].groupby(['subject', 'session'])[['tr_ms', 'te_ms']].agg(set)
    n = 0
    for fn in sorted(glob.glob(op.join(TARGET, 'sub-*', 'ses-*', 'anat', '*_T1w.nii.gz'))):
        sub, ses = re.search(r'sub-(\d\d)_ses-(\d)_', op.basename(fn)).groups()
        trs, tes = t1.loc[(sub, int(ses))]
        assert len(trs) == 1 and len(tes) == 1, (fn, trs, tes)
        json_dump({'RepetitionTimeExcitation': round(trs.pop() / 1000, 5),
                   'EchoTime': round(tes.pop() / 1000, 5)}, fn.replace('.nii.gz', '.json'))
        n += 1
    print(f'T1w sidecars: {n}')


def write_raw():
    json_dump({
        'Name': TITLE,
        'BIDSVersion': '1.10.0',
        'DatasetType': 'raw',
        'License': 'CC0',
        'Authors': AUTHORS,
        'HowToAcknowledge': f'Please cite the accompanying paper ({PREPRINT}).',
        'ReferencesAndLinks': [PREPRINT, CODE],
    }, op.join(TARGET, 'dataset_description.json'))

    common = check_bold_sidecars()
    assert common['MagneticFieldStrength'] == SCANNER['MagneticFieldStrength']
    json_dump({
        'TaskName': TASK_NAME,
        'TaskDescription': TASK_DESCRIPTION,
        'Instructions': INSTRUCTIONS,
        **SCANNER,
        **EPI_SEQUENCE,
        'SliceTiming': common['SliceTiming'],
        # SENSE 1.5 per the paper's Methods and the 2021 protocol sheet; the original
        # conversion template said 2 (not recorded in the PAR headers).
        'ParallelReductionFactorInPlane': 1.5,
        'TotalReadoutTime': common['TotalReadoutTime'],
    }, op.join(TARGET, f'task-{TASK}_bold.json'))

    write_run_repetition_times()

    json_dump({**SCANNER, **T1W_SEQUENCE}, op.join(TARGET, 'T1w.json'))
    write_t1w_sidecars()
    # Synthesised PEPOLAR fieldmaps are volumes of the BOLD runs themselves.
    json_dump({**SCANNER, **EPI_SEQUENCE}, op.join(TARGET, 'epi.json'))

    json_dump({
        'trial_type': {'Description': 'Event type. Onsets are relative to the first recorded volume '
                                      'of the run; duration is the dot-cloud display (stimulus) or the '
                                      'time the response slider was on screen, i.e. until the click or '
                                      'the 3-s time-out (response).',
                       'Levels': {'stimulus': 'Dot-cloud presentation.',
                                  'response': 'Response slider on screen.'}},
        'StimulusPresentation': {'SoftwareName': 'PsychoPy (with exptools2)',
                                 'SoftwareRRID': 'SCR_006571',
                                 'OperatingSystem': 'Windows'},
        'trial_nr': {'Description': 'Trial number within the session (1-240); run r holds trials '
                                    '30*(r-1)+1 ... 30*r.'},
        'range': {'Description': 'Numerosity range of the run (constant within a run).',
                  'Levels': {'narrow': 'Numerosities drawn uniformly from 10-25.',
                             'wide': 'Numerosities drawn uniformly from 10-40.'}},
        'n': {'Description': 'Number of dots shown on this trial.'},
        'response': {'Description': 'Numerosity reported with the slider; n/a if no response '
                                    'within 3 s.'},
        'response_time': {'Description': 'Response time (s) from response-screen onset to the '
                                         'click; n/a if no response.', 'Units': 's'},
        'start_marker_position': {'Description': 'Initial (random) position of the slider marker, '
                                                 'in numerosity units.'},
    }, op.join(TARGET, f'task-{TASK}_events.json'))

    src = pd.read_csv(op.join(WORK, 'participants_source.tsv'), sep='\t')
    src['participant_id'] = src['participant_id'].map(lambda x: f'sub-{int(x):02d}')
    released = [f'sub-{s}' for s in get_subjects()]
    part = src.set_index('participant_id').loc[released].reset_index()
    part = part.rename(columns={'Sex': 'sex'})[['participant_id', 'age', 'sex']]
    part['age'] = part['age'].astype(int)
    part.to_csv(op.join(TARGET, 'participants.tsv'), sep='\t', index=False)
    json_dump({'age': {'Description': 'Age at participation', 'Units': 'years'},
               'sex': {'Description': 'Self-reported sex', 'Levels': {'F': 'female', 'M': 'male'}}},
              op.join(TARGET, 'participants.json'))

    with open(op.join(TARGET, 'README'), 'w') as f:
        f.write(README.format(title=TITLE, n=len(released), task=TASK, preprint=PREPRINT, code=CODE))
    with open(op.join(TARGET, 'CHANGES'), 'w') as f:
        f.write('1.0.0 unreleased\n  - Initial release.\n')


def write_derivatives():
    out = op.join(TARGET, 'derivatives', 'fmriprep')
    json_dump({
        'Name': 'fMRIPrep outputs used in: ' + TITLE,
        'BIDSVersion': '1.10.0',
        'DatasetType': 'derivative',
        'License': 'CC0',
        'GeneratedBy': [{'Name': 'fMRIPrep', 'Version': '23.2.1',
                         'CodeURL': 'https://github.com/nipreps/fmriprep'},
                        {'Name': 'FreeSurfer', 'Version': '7.3.2'}],
        'SourceDatasets': [{'URL': 'bids::'}],
        'HowToAcknowledge': f'Please cite the accompanying paper ({PREPRINT}) and fMRIPrep '
                            '(https://doi.org/10.1038/s41592-018-0235-4).',
    }, op.join(out, 'dataset_description.json'))
    for name in ['desc-aseg_dseg.tsv', 'desc-aparcaseg_dseg.tsv']:
        copy_file(op.join(SOURCE, 'derivatives', 'fmriprep', name), op.join(out, name))
    for name in ['CITATION.md', 'CITATION.bib']:
        copy_file(op.join(SOURCE, 'derivatives', 'fmriprep', 'logs', name), op.join(out, name))
    with open(op.join(out, 'README'), 'w') as f:
        f.write(DERIV_README.format(task=TASK))


README = """{title}
{underline}

Raw BIDS data of {n} healthy adult participants who estimated the number of dots in visual displays while undergoing 3T fMRI (Philips; EPI, 2.5 x 2.5 x 3 mm voxels, TR = 2.29-2.30 s; the exact scanner-reported TR of each run is in its JSON sidecar), in two contexts that differed in the range of possible numerosities (narrow: 10-25, wide: 10-40).

Paper (preprint): {preprint}
Analysis code: {code}

Design
------
Two sessions per participant, each with 8 runs of 30 trials (task-{task}); runs 1-4 and 5-8 form two blocks, one per numerosity range (see the `range` column of the events files). See task-{task}_bold.json (TaskDescription) and task-{task}_events.json for the trial structure and all event columns.

Participants
------------
The released participants are exactly the analysed sample (sub-01 ... sub-41 without sub-11 and sub-23, who completed only one session). Subject labels are those used in the paper and code. Pilot participants are not included.

Data
----
- anat: T1-weighted images, two separate MPRAGE acquisitions per session (sub-19 and sub-31 have an extra repeat); per-image timing is in each sidecar. All T1-weighted images were defaced with the pydeface method (face mask registered with FSL FLIRT; voxels under the mask set to zero) and every image was visually checked.
- func: BOLD runs.
- fmap: The acquisition had no dedicated fieldmaps. Runs alternate the phase-encoding direction; each `_epi` image consists of 5 volumes taken from a neighbouring run with the opposite phase-encoding direction (see IntendedFor), for PEPOLAR susceptibility-distortion correction.
- derivatives/fmriprep: fMRIPrep 23.2.1 outputs used by the analyses (see its README).

Physiological recordings, eye-tracking data and the raw PsychoPy logs are not included. Trial-wise behaviour (including the practice blocks) is additionally available as a single table on figshare (see the paper's data availability statement).
"""

DERIV_README = """fMRIPrep derivatives
====================

Selected outputs of fMRIPrep 23.2.1 (FreeSurfer 7.3.2), run with `--output-spaces MNI152NLin2009cAsym T1w fsaverage fsnative --dummy-scans 4 --no-submm-recon`. Only the outputs used by the paper's analyses are included:

- sub-XX/anat: preprocessed T1w (DEFACED, same method as the raw T1w images), brain/ribbon masks, tissue segmentation and probability maps, T1w <-> fsnative transforms, GIfTI surfaces.
- sub-XX/ses-Y/func: BOLD in T1w space (desc-preproc_bold), its brain mask and reference image, confound regressors.
- sourcedata/freesurfer/sub-XX: surf/ and label/, plus skull-stripped and label volumes from mri/. Volumes that contain the face (orig, rawavg, nu, T1) and the scripts/stats/touch folders are not included. Surface-file creation stamps were replaced by "created by freesurfer".

Not included: MNI152NLin2009cAsym, fsaverage and fsLR outputs, fieldmap/head-motion intermediates, HTML reports and figures (which show non-defaced anatomy), logs, and the fsaverage template subject (available with FreeSurfer).

Task label `task` was renamed to `{task}` to match the raw dataset.
"""
README = README.replace('{underline}', '=' * len('Distributed range adaptation in human parietal encoding of numbers'))


if __name__ == '__main__':
    write_raw()
    write_derivatives()
    print('done')
