"""Deface all face-bearing anatomical images of one subject.

Inputs (WORK/undefaced, header-cleaned copies, never uploaded):
- raw T1w:            WORK/undefaced/raw/sub-XX/ses-Y/anat/*_T1w.nii.gz
- fMRIPrep T1w:       WORK/undefaced/fmriprep/sub-XX/anat/sub-XX_desc-preproc_T1w.nii.gz

Method = pydeface's (same template, face mask and FLIRT calls as pydeface
2.1.0, also used for OpenNeuro ds007508), but the mask is applied at the
stored-voxel level: voxels under the warped face mask are set to 0 in the
file's own datatype, and header + all other voxels are copied byte-for-byte
(pydeface itself re-saves through nibabel, which can re-quantise scaled
integer data). Any voxel the (trilinearly interpolated) mask touches at all
(mask < 0.99) is removed, i.e. slightly more conservative than pydeface.

Outputs: defaced images into TARGET, the removal masks + FLIRT matrices into
WORK/deface_masks (for QC), and one summary TSV per subject in WORK/qc.
"""
import argparse
import glob
import os
import os.path as op
import subprocess
import tempfile
from importlib.resources import files

import nibabel as nib
import numpy as np
import pandas as pd

from neural_priors.data_release.release import (TARGET, WORK, copy_nifti_clean_header, get_subjects,
                                             makedirs_for, verify_same_image)

TEMPLATE = str(files('pydeface').joinpath('data/mean_reg2mean.nii.gz'))
FACEMASK = str(files('pydeface').joinpath('data/facemask.nii.gz'))
MASK_THRESHOLD = 0.99


def run(cmd):
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def warp_facemask(infile, tmpdir):
    mat = op.join(tmpdir, 'template_to_input.mat')
    run(['flirt', '-in', TEMPLATE, '-ref', infile, '-omat', mat,
         '-out', op.join(tmpdir, 'template_reg.nii.gz'), '-cost', 'mutualinfo'])
    warped = op.join(tmpdir, 'facemask_warped.nii.gz')
    run(['flirt', '-in', FACEMASK, '-ref', infile, '-applyxfm', '-init', mat, '-out', warped])
    return warped, mat


def deface(src, dst, mask_dst):
    img = nib.load(src)
    slope, inter = img.header.get_slope_inter()
    assert inter in (None, 0.0) or np.isnan(inter), f'{src}: scl_inter={inter}, raw 0 would not be 0'
    assert img.ndim == 3, img.shape

    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmpdir:
        warped, mat = warp_facemask(src, tmpdir)
        mask = nib.load(warped)
        assert mask.shape == img.shape and np.allclose(mask.affine, img.affine, atol=1e-3)
        removed = np.asanyarray(mask.dataobj) < MASK_THRESHOLD

        copy_nifti_clean_header(src, dst, data_transform=lambda a: np.where(removed, 0, a))
        verify_same_image(src, dst, allow_changed_voxels=removed)

        makedirs_for(mask_dst)
        out = nib.Nifti1Image(removed.astype(np.uint8), img.affine)
        out.to_filename(mask_dst)
        os.replace(mat, mask_dst.replace('_removed.nii.gz', '_flirt.mat'))

    data = np.asanyarray(img.dataobj)
    head = data > np.percentile(data[data > 0], 20)
    return dict(n_removed=int(removed.sum()),
                frac_volume_removed=float(removed.mean()),
                frac_head_voxels_removed=float((removed & head).sum() / head.sum()))


def main(subject):
    subject = f'{int(subject):02d}'
    assert subject in get_subjects()
    jobs = []
    for src in sorted(glob.glob(op.join(WORK, 'undefaced', 'raw', f'sub-{subject}', 'ses-*', 'anat', '*_T1w.nii.gz'))):
        rel = op.relpath(src, op.join(WORK, 'undefaced', 'raw'))
        jobs.append(('raw', src, op.join(TARGET, rel)))
    for src in sorted(glob.glob(op.join(WORK, 'undefaced', 'fmriprep', f'sub-{subject}', 'anat', '*_desc-preproc_T1w.nii.gz'))):
        rel = op.relpath(src, op.join(WORK, 'undefaced', 'fmriprep'))
        jobs.append(('fmriprep', src, op.join(TARGET, 'derivatives', 'fmriprep', rel)))
    n_raw = sum(j[0] == 'raw' for j in jobs)
    assert n_raw >= 4 and len(jobs) == n_raw + 1, jobs

    rows = []
    for kind, src, dst in jobs:
        mask_dst = op.join(WORK, 'deface_masks', kind, op.basename(src).replace('.nii.gz', '_removed.nii.gz'))
        stats = deface(src, dst, mask_dst)
        rows.append(dict(subject=subject, kind=kind, file=op.relpath(dst, TARGET), **stats))
        print(rows[-1], flush=True)

    out = op.join(WORK, 'qc', f'sub-{subject}_deface.tsv')
    makedirs_for(out)
    pd.DataFrame(rows).to_csv(out, sep='\t', index=False)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject')
    args = parser.parse_args()
    main(args.subject)
