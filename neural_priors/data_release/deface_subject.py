"""Deface all face-bearing anatomical images of one subject.

Inputs (WORK/undefaced, header-cleaned copies, never uploaded):
- fMRIPrep T1w:  WORK/undefaced/fmriprep/sub-XX/anat/sub-XX_desc-preproc_T1w.nii.gz
- raw T1w:       WORK/undefaced/raw/sub-XX/ses-Y/anat/*_T1w.nii.gz

1. The fMRIPrep T1w is defaced with pydeface's method (same template, face mask
   and FLIRT calls as pydeface 2.1.0). Any voxel the trilinearly interpolated
   mask touches (mask < 0.99) is removed.
2. Raw T1ws are NOT registered to the template individually: on these oblique
   sagittal Philips images that registration failed for several subjects (the
   mask landed on the back of the head, leaving the face intact). Instead the
   fMRIPrep-space removal mask is carried into every raw T1w with fMRIPrep's own
   from-orig_to-T1w coregistration (direction validated by image correlation),
   with voxels outside the fMRIPrep grid treated as removed.

In both cases the mask is applied at the stored-voxel level: voxels are set to 0
in the file's own datatype, header + all other voxels copied byte-for-byte.
Outputs: defaced images into TARGET, removal masks into WORK/deface_masks, one
summary TSV per subject in WORK/qc.
"""
import argparse
import glob
import os
import re
import shutil
import os.path as op
import subprocess
import tempfile
from importlib.resources import files

import nibabel as nib
import numpy as np
import nitransforms as nt
import pandas as pd

from neural_priors.data_release.release import (SOURCE, TARGET, WORK, copy_nifti_clean_header, get_subjects,
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


def apply_and_save(src, dst, removed, mask_dst, affine):
    copy_nifti_clean_header(src, dst, data_transform=lambda a: np.where(removed, 0, a))
    verify_same_image(src, dst, allow_changed_voxels=removed)
    makedirs_for(mask_dst)
    nib.Nifti1Image(removed.astype(np.uint8), affine).to_filename(mask_dst)


def check_img(img, src):
    slope, inter = img.header.get_slope_inter()
    assert inter in (None, 0.0) or np.isnan(inter), f'{src}: scl_inter={inter}, raw 0 would not be 0'
    assert img.ndim == 3, img.shape


def head_stats(img, removed):
    data = np.asanyarray(img.dataobj)
    head = data > np.percentile(data[data > 0], 20)
    return dict(n_removed=int(removed.sum()),
                frac_volume_removed=float(removed.mean()),
                frac_head_voxels_removed=float((removed & head).sum() / head.sum()))


def deface_template_registration(src, dst, mask_dst):
    img = nib.load(src)
    check_img(img, src)
    if op.exists(mask_dst):
        # Re-use the mask that was already visually approved (FLIRT is deterministic,
        # but this makes the released image provably the approved one).
        removed = np.asanyarray(nib.load(mask_dst).dataobj) > 0
        assert removed.shape == img.shape
        apply_and_save(src, dst, removed, mask_dst + '.tmp.nii.gz', img.affine)
        os.remove(mask_dst + '.tmp.nii.gz')
        return removed, dict(method='pydeface-template (existing mask)', **head_stats(img, removed))
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmpdir:
        warped, mat = warp_facemask(src, tmpdir)
        mask = nib.load(warped)
        assert mask.shape == img.shape and np.allclose(mask.affine, img.affine, atol=1e-3)
        removed = np.asanyarray(mask.dataobj) < MASK_THRESHOLD
        shutil.move(mat, mask_dst.replace('_removed.nii.gz', '_flirt.mat'))
    apply_and_save(src, dst, removed, mask_dst, img.affine)
    return removed, dict(method='pydeface-template', **head_stats(img, removed))


def resample(xfm, moving, reference, order, cval=0.0):
    from nitransforms.resampling import apply
    return np.asanyarray(apply(xfm, moving, reference=reference, order=order, cval=cval,
                               prefilter=False).dataobj)


def deface_from_fmriprep(src, dst, mask_dst, subject, preproc_fn, removed_t1w):
    img = nib.load(src)
    check_img(img, src)
    name = op.basename(src)
    ses, run = re.search(r'_ses-(\d)_run-(\d)_T1w', name).groups()
    xfm = nt.linear.load(op.join(SOURCE, 'derivatives', 'fmriprep', f'sub-{subject}', f'ses-{ses}', 'anat',
                                 f'sub-{subject}_ses-{ses}_run-{run}_from-orig_to-T1w_mode-image_xfm.txt'),
                         fmt='itk')
    raw = nib.Nifti1Image(np.asanyarray(img.dataobj).astype(np.float32), img.affine)
    pre = nib.load(preproc_fn)
    pre = nib.Nifti1Image(np.asanyarray(pre.dataobj).astype(np.float32), pre.affine)
    rawdata = np.asanyarray(raw.dataobj)
    head = rawdata > np.percentile(rawdata[rawdata > 0], 20)

    # Direction check: fMRIPrep T1w resampled into the raw grid must match the raw image.
    rs = {}
    for label, t in [('inverse', ~xfm), ('forward', xfm)]:
        res = resample(t, pre, raw, order=1)
        rs[label] = (np.corrcoef(res[head], rawdata[head])[0, 1], t)
    label = max(rs, key=lambda k: rs[k][0])
    r, t = rs[label]
    if r <= 0.5:
        return None   # caller falls back to a sibling image on the same grid

    mask_img = nib.Nifti1Image(removed_t1w.astype(np.float32), pre.affine)
    removed = resample(t, mask_img, raw, order=1, cval=1.0) > 0.01
    apply_and_save(src, dst, removed, mask_dst, img.affine)
    other = [k for k in rs if k != label][0]
    return dict(method='fmriprep-mask', xfm_direction=label, r_preproc_in_raw=r,
                r_other_direction=rs[other][0], **head_stats(img, removed))


def main(subject):
    subject = f'{int(subject):02d}'
    assert subject in get_subjects()
    rows = []

    preproc_src = op.join(WORK, 'undefaced', 'fmriprep', f'sub-{subject}', 'anat', f'sub-{subject}_desc-preproc_T1w.nii.gz')
    preproc_dst = op.join(TARGET, 'derivatives', 'fmriprep', f'sub-{subject}', 'anat', op.basename(preproc_src))
    removed_t1w, stats = deface_template_registration(
        preproc_src, preproc_dst,
        op.join(WORK, 'deface_masks', 'fmriprep', op.basename(preproc_src).replace('.nii.gz', '_removed.nii.gz')))
    rows.append(dict(subject=subject, kind='fmriprep', file=op.relpath(preproc_dst, TARGET), **stats))
    print(rows[-1], flush=True)

    raws = sorted(glob.glob(op.join(WORK, 'undefaced', 'raw', f'sub-{subject}', 'ses-*', 'anat', '*_T1w.nii.gz')))
    assert len(raws) >= 4, raws
    failed = []
    for src in raws:
        dst = op.join(TARGET, op.relpath(src, op.join(WORK, 'undefaced', 'raw')))
        mask_dst = op.join(WORK, 'deface_masks', 'raw', op.basename(src).replace('.nii.gz', '_removed.nii.gz'))
        stats = deface_from_fmriprep(src, dst, mask_dst, subject, preproc_src, removed_t1w)
        if stats is None:
            failed.append((src, dst, mask_dst))
            continue
        rows.append(dict(subject=subject, kind='raw', file=op.relpath(dst, TARGET), **stats))
        print(rows[-1], flush=True)

    # Images that do not look like the fMRIPrep T1w (e.g. sub-19's third ses-2
    # reconstruction, which is mostly noise) get the mask of a successfully
    # handled image of the same session on the identical voxel grid.
    for src, dst, mask_dst in failed:
        img = nib.load(src)
        ses_dir = op.dirname(src)
        sibling = next((r for r in rows if r['kind'] == 'raw'
                        and op.dirname(op.join(WORK, 'undefaced', 'raw', r['file'])) == ses_dir
                        and nib.load(op.join(WORK, 'undefaced', 'raw', r['file'])).shape == img.shape
                        and np.allclose(nib.load(op.join(WORK, 'undefaced', 'raw', r['file'])).affine, img.affine)),
                       None)
        assert sibling is not None, f'{src}: transform check failed and no sibling on the same grid'
        sib_mask = op.join(WORK, 'deface_masks', 'raw', op.basename(sibling['file']).replace('.nii.gz', '_removed.nii.gz'))
        removed = np.asanyarray(nib.load(sib_mask).dataobj) > 0
        apply_and_save(src, dst, removed, mask_dst, img.affine)
        rows.append(dict(subject=subject, kind='raw', file=op.relpath(dst, TARGET),
                         method=f'sibling-mask ({op.basename(sibling["file"])})', **head_stats(img, removed)))
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
