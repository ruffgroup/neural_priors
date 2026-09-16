"""Stage the fMRIPrep (+ FreeSurfer) derivatives of one subject for the release.

Only what the paper's analyses use, selected by explicit ALLOWLISTS (anything
not matched is left out, so new/unexpected fMRIPrep outputs can never slip in):
- anat: T1w-space masks/segmentations, fsnative<->T1w transforms, GIfTI surfaces.
  desc-preproc_T1w is NOT skull-stripped (it shows the face): it goes to
  WORK/undefaced and only its defaced version reaches TARGET (deface_subject.py).
- func: T1w-space preprocessed BOLD, its brain mask and boldref, confounds.
- No MNI152 / fsaverage / fsLR outputs, no fieldmap/HMC intermediates, no HTML
  reports / figures / logs (reports show head slices; logs hold paths).
- FreeSurfer: surf/ + label/ + skull-stripped/label volumes from mri/ only.
  T1.mgz, orig*, rawavg, nu* contain the face and are never copied; scripts/,
  stats/, touch/, tmp/, trash/ (usernames, hostnames, command lines) neither.
  Surface files get their 'created by <user> on <date>' stamp replaced.

Usage: python stage_fmriprep.py 01
"""
import argparse
import glob
import json
import os.path as op
import re

import nibabel as nib
import numpy as np

from neural_priors.openneuro.release import (SOURCE, TARGET, WORK, TASK, TASK_NAME, TASK_OLD,
                                             copy_file, get_subjects, makedirs_for, rename_task)

FMRIPREP = op.join(SOURCE, 'derivatives', 'fmriprep')
OUT = op.join(TARGET, 'derivatives', 'fmriprep')

ANAT_ALLOW = [
    r'sub-(\d\d)_desc-preproc_T1w\.json',
    r'sub-(\d\d)_desc-(brain|ribbon)_mask\.(json|nii\.gz)',
    r'sub-(\d\d)_dseg\.nii\.gz',
    r'sub-(\d\d)_label-(CSF|GM|WM)_probseg\.nii\.gz',
    r'sub-(\d\d)_from-(fsnative_to-T1w|T1w_to-fsnative)_mode-image_xfm\.txt',
    r'sub-(\d\d)_hemi-[LR]_(white|pial|midthickness|sphere|desc-reg_sphere)\.surf\.gii',
    r'sub-(\d\d)_hemi-[LR]_(sulc|thickness)\.shape\.gii',
]
ANAT_UNDEFACED = r'sub-(\d\d)_desc-preproc_T1w\.nii\.gz'

FUNC_ALLOW = [
    rf'sub-(\d\d)_ses-[12]_task-{TASK_OLD}_run-\d_space-T1w_desc-preproc_bold\.(json|nii\.gz)',
    rf'sub-(\d\d)_ses-[12]_task-{TASK_OLD}_run-\d_space-T1w_desc-brain_mask\.nii\.gz',
    rf'sub-(\d\d)_ses-[12]_task-{TASK_OLD}_run-\d_space-T1w_boldref\.nii\.gz',
    rf'sub-(\d\d)_ses-[12]_task-{TASK_OLD}_run-\d_desc-confounds_timeseries\.(json|tsv)',
]

FS_MRI_ALLOW = ['aseg.mgz', 'aparc+aseg.mgz', 'aparc.a2009s+aseg.mgz', 'aparc.DKTatlas+aseg.mgz',
                'wmparc.mgz', 'brainmask.mgz', 'brain.mgz', 'brain.finalsurfs.mgz', 'norm.mgz',
                'wm.mgz', 'filled.mgz', 'ribbon.mgz', 'lh.ribbon.mgz', 'rh.ribbon.mgz',
                'transforms/talairach.xfm', 'transforms/talairach.lta']
FS_SURF_ALLOW = [f'{h}.{s}' for h in ['lh', 'rh'] for s in
                 ['white', 'pial', 'smoothwm', 'inflated', 'orig', 'sphere', 'sphere.reg',
                  'fsaverage.sphere.reg', 'midthickness', 'curv', 'sulc', 'thickness', 'area',
                  'volume', 'avg_curv']]
FS_TRIANGLE_SURFACES = ['white', 'pial', 'smoothwm', 'inflated', 'orig', 'sphere', 'sphere.reg',
                        'fsaverage.sphere.reg', 'midthickness']
NEUTRAL_STAMP = 'created by freesurfer'


def fullmatch_any(patterns, name):
    return any(re.fullmatch(p, name) for p in patterns)


def rewrite_json(src, dst):
    with open(src) as f:
        meta = json.load(f)
    text = json.dumps(meta, indent=2)
    # Raw-data references follow the release naming (task label, gzipped raw).
    text = text.replace(f'_task-{TASK_OLD}_', f'_task-{TASK}_')
    text = re.sub(r'(_bold|_T1w|_epi)\.nii"', r'\1.nii.gz"', text)
    text = text.replace('"TaskName": "Risk task"', f'"TaskName": "{TASK_NAME}"')
    makedirs_for(dst)
    with open(dst, 'w') as f:
        f.write(text + '\n')


def stage_fmriprep(subject):
    src_sub = op.join(FMRIPREP, f'sub-{subject}')
    n = 0
    for src in sorted(glob.glob(op.join(src_sub, 'anat', '*'))):
        name = op.basename(src)
        if re.fullmatch(ANAT_UNDEFACED, name):
            dst = op.join(WORK, 'undefaced', 'fmriprep', f'sub-{subject}', 'anat', name)
            copy_file(src, dst)
            print(f'preproc_T1w (undefaced, WORK only) {name}')
            continue
        if not fullmatch_any(ANAT_ALLOW, name):
            continue
        dst = op.join(OUT, f'sub-{subject}', 'anat', name)
        rewrite_json(src, dst) if name.endswith('.json') else copy_file(src, dst)
        n += 1

    for src in sorted(glob.glob(op.join(src_sub, 'ses-[12]', 'func', '*'))):
        name = op.basename(src)
        if not fullmatch_any(FUNC_ALLOW, name):
            continue
        session = re.search(r'_ses-(\d)_', name).group(1)
        dst = op.join(OUT, f'sub-{subject}', f'ses-{session}', 'func', rename_task(name))
        rewrite_json(src, dst) if name.endswith('.json') else copy_file(src, dst)
        n += 1

    n_bold = len(glob.glob(op.join(OUT, f'sub-{subject}', 'ses-*', 'func', '*_desc-preproc_bold.nii.gz')))
    assert n_bold == 16, f'sub-{subject}: {n_bold} preprocessed BOLD runs'
    print(f'fmriprep: {n} files')


def stage_freesurfer(subject):
    src_sub = op.join(FMRIPREP, 'sourcedata', 'freesurfer', f'sub-{subject}')
    out_sub = op.join(OUT, 'sourcedata', 'freesurfer', f'sub-{subject}')

    for rel in FS_MRI_ALLOW:
        copy_file(op.join(src_sub, 'mri', rel), op.join(out_sub, 'mri', rel))

    for src in sorted(glob.glob(op.join(src_sub, 'label', '*'))):
        if re.search(r'\.(label|annot|ctab)$', src):
            copy_file(src, op.join(out_sub, 'label', op.basename(src)))

    for name in FS_SURF_ALLOW:
        src, dst = op.join(src_sub, 'surf', name), op.join(out_sub, 'surf', name)
        if name.split('.', 1)[1] in FS_TRIANGLE_SURFACES:
            coords, faces, meta = nib.freesurfer.read_geometry(src, read_metadata=True)
            makedirs_for(dst)
            nib.freesurfer.write_geometry(dst, coords, faces, create_stamp=NEUTRAL_STAMP,
                                          volume_info=meta)
            c2, f2 = nib.freesurfer.read_geometry(dst)
            assert np.array_equal(c2, coords) and np.array_equal(f2, faces), dst
        else:
            copy_file(src, dst)
    print('freesurfer: done')


def main(subject):
    subject = f'{int(subject):02d}'
    assert subject in get_subjects(), f'sub-{subject} is not part of the release'
    stage_fmriprep(subject)
    stage_freesurfer(subject)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject')
    args = parser.parse_args()
    main(args.subject)
