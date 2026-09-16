"""Defacing / de-identification QC for one subject (run after deface_subject.py).

For every defaced image:
- brain clipping: removed voxels inside the fMRIPrep brain mask (and inside the
  mask dilated by 3 mm). For raw T1ws the removal mask is moved into fMRIPrep's
  T1w space with fMRIPrep's own from-orig_to-T1w transform; the transform
  direction is validated by correlating the resampled raw T1w with
  desc-preproc_T1w (reported as `r_orig_vs_preproc`).
- renders: frontal and lateral *depth renders* (first above-threshold voxel along
  each ray, shaded by depth = a poor man's 3D surface render, which is what makes
  a face recognisable) of the defaced image, plus a mid-sagittal slice with the
  removed region (red) and brain mask outline (cyan).

For the other volumes that are released (FreeSurfer mri/*.mgz, fMRIPrep T1w-space
masks/segmentations): nonzero voxels outside the dilated FreeSurfer / fMRIPrep
brain mask, plus a frontal depth render of their union (must look like a brain,
never like a head).
"""
import argparse
import glob
import os.path as op
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import nitransforms as nt
import numpy as np
import pandas as pd
from scipy import ndimage

from neural_priors.data_release.release import SOURCE, TARGET, WORK, get_subjects, makedirs_for

FMRIPREP_SRC = op.join(SOURCE, 'derivatives', 'fmriprep')
FMRIPREP_OUT = op.join(TARGET, 'derivatives', 'fmriprep')


def canonical(img):
    return nib.as_closest_canonical(img)


def depth_render(data, axis, from_end, thr):
    """Depth map of the first voxel above thr along `axis` (RAS array)."""
    above = data > thr
    if from_end:
        above = np.flip(above, axis=axis)
    hit = above.any(axis=axis)
    depth = np.argmax(above, axis=axis).astype(float)
    shade = 1.0 - depth / above.shape[axis]
    shade = np.where(hit, shade, np.nan)
    # emphasise relief: add a directional-light term from the depth gradient
    gy, gx = np.gradient(np.nan_to_num(depth, nan=above.shape[axis]))
    light = np.clip(0.6 + 0.08 * (gx + gy), 0, 1)
    return np.where(hit, 0.35 * shade + 0.65 * light, np.nan)


def head_threshold(data):
    d = data[np.isfinite(data)]
    return 0.1 * np.percentile(d, 99)


def resample(xfm, moving, reference, order):
    try:
        from nitransforms.resampling import apply
        return apply(xfm, moving, reference=reference, order=order)
    except ImportError:
        return xfm.apply(moving, reference=reference, order=order)


def to_float_img(img):
    return nib.Nifti1Image(np.asanyarray(img.dataobj).astype(np.float32), img.affine)


def qc_raw_t1w(subject, row, preproc_undefaced, brain_mask, figdir):
    defaced_fn = op.join(TARGET, row['file'])
    name = op.basename(defaced_fn)
    removed_fn = op.join(WORK, 'deface_masks', 'raw', name.replace('.nii.gz', '_removed.nii.gz'))
    undefaced_fn = op.join(WORK, 'undefaced', 'raw', row['file'])
    ses, run = re.search(r'_ses-(\d)_run-(\d)_T1w', name).groups()
    xfm_fn = op.join(FMRIPREP_SRC, f'sub-{subject}', f'ses-{ses}', 'anat',
                     f'sub-{subject}_ses-{ses}_run-{run}_from-orig_to-T1w_mode-image_xfm.txt')

    undefaced = to_float_img(nib.load(undefaced_fn))
    removed = to_float_img(nib.load(removed_fn))
    xfm = nt.linear.load(xfm_fn, fmt='itk')
    pre = np.asanyarray(preproc_undefaced.dataobj).astype(float)
    brain = np.asanyarray(brain_mask.dataobj) > 0

    # Validate direction: the resampled raw T1w must match desc-preproc_T1w.
    best = None
    for label, t in [('forward', xfm), ('inverse', ~xfm)]:
        res = np.asanyarray(resample(t, undefaced, preproc_undefaced, order=1).dataobj)
        r = np.corrcoef(res[brain], pre[brain])[0, 1]
        if best is None or r > best[1]:
            best = (label, r, t)
    label, r, t = best
    removed_t1w = np.asanyarray(resample(t, removed, preproc_undefaced, order=1).dataobj) > 0.01
    dil = ndimage.binary_dilation(brain, iterations=3)

    defaced = canonical(nib.load(defaced_fn))
    render(defaced, canonical(nib.load(removed_fn)), None, op.join(figdir, name.replace('.nii.gz', '.png')),
           f'sub-{subject} {name}')
    return dict(xfm_direction=label, r_orig_vs_preproc=r,
                brain_voxels_removed=int((removed_t1w & brain).sum()),
                dilated_brain_voxels_removed=int((removed_t1w & dil).sum()))


def qc_preproc_t1w(subject, row, brain_mask, figdir):
    defaced_fn = op.join(TARGET, row['file'])
    name = op.basename(defaced_fn)
    removed_fn = op.join(WORK, 'deface_masks', 'fmriprep', name.replace('.nii.gz', '_removed.nii.gz'))
    removed = np.asanyarray(nib.load(removed_fn).dataobj) > 0
    brain = np.asanyarray(brain_mask.dataobj) > 0
    dil = ndimage.binary_dilation(brain, iterations=3)
    render(canonical(nib.load(defaced_fn)), canonical(nib.load(removed_fn)), canonical(brain_mask),
           op.join(figdir, name.replace('.nii.gz', '.png')), f'sub-{subject} {name}')
    return dict(xfm_direction='same grid', r_orig_vs_preproc=np.nan,
                brain_voxels_removed=int((removed & brain).sum()),
                dilated_brain_voxels_removed=int((removed & dil).sum()))


def render(img, removed_img, brain_img, out, title):
    data = np.asanyarray(img.dataobj).astype(float)
    data = ndimage.gaussian_filter(data, 1.0)
    thr = head_threshold(data)
    front = depth_render(data, axis=1, from_end=True, thr=thr)    # look from anterior (+y)
    side = depth_render(data, axis=0, from_end=False, thr=thr)    # look from the left (-x)
    removed = np.asanyarray(removed_img.dataobj) > 0

    x_mid = data.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(front.T, origin='lower', cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Frontal depth render')
    axes[1].imshow(side.T, origin='lower', cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Lateral depth render (from left)')
    sl = np.asanyarray(img.dataobj)[x_mid].astype(float)
    axes[2].imshow(sl.T, origin='lower', cmap='gray', vmax=np.percentile(sl, 99.5))
    axes[2].imshow(np.ma.masked_where(~removed[x_mid].T, removed[x_mid].T), origin='lower',
                   cmap='autumn', alpha=0.35)
    if brain_img is not None:
        axes[2].contour((np.asanyarray(brain_img.dataobj)[x_mid] > 0).T, levels=[0.5], colors='c', linewidths=0.6)
    axes[2].set_title('Mid-sagittal: removed (red)')
    for ax in axes:
        ax.axis('off')
    fig.suptitle(title, fontsize=9)
    makedirs_for(out)
    fig.savefig(out, dpi=70, bbox_inches='tight')
    plt.close(fig)


def qc_other_volumes(subject, figdir):
    rows = []
    # FreeSurfer volumes, conformed space; reference = brainmask.mgz
    fs = op.join(FMRIPREP_OUT, 'sourcedata', 'freesurfer', f'sub-{subject}', 'mri')
    bm = np.asanyarray(nib.load(op.join(fs, 'brainmask.mgz')).dataobj) > 0
    bm_dil = ndimage.binary_dilation(bm, iterations=3)
    union = None
    for fn in sorted(glob.glob(op.join(fs, '*.mgz'))):
        img = nib.load(fn)
        d = np.asanyarray(img.dataobj) != 0
        rows.append(dict(subject=subject, file=op.relpath(fn, TARGET), n_nonzero=int(d.sum()),
                         n_outside_dilated_brainmask=int((d & ~bm_dil).sum())))
        union = d if union is None else (union | d)
    u = nib.as_closest_canonical(nib.MGHImage(union.astype(np.float32), img.affine))
    fs_front = depth_render(np.asanyarray(u.dataobj), axis=1, from_end=True, thr=0.5)

    # fMRIPrep T1w-space anatomical volumes (other than the defaced preproc T1w)
    anat = op.join(FMRIPREP_OUT, f'sub-{subject}', 'anat')
    brain = np.asanyarray(nib.load(op.join(anat, f'sub-{subject}_desc-brain_mask.nii.gz')).dataobj) > 0
    dil = ndimage.binary_dilation(brain, iterations=3)
    union = None
    for fn in sorted(glob.glob(op.join(anat, '*.nii.gz'))):
        if fn.endswith('_desc-preproc_T1w.nii.gz'):
            continue
        img = nib.load(fn)
        d = np.asanyarray(img.dataobj) > 0
        rows.append(dict(subject=subject, file=op.relpath(fn, TARGET), n_nonzero=int(d.sum()),
                         n_outside_dilated_brainmask=int((d & ~dil).sum())))
        union = d if union is None else (union | d)
    u = nib.as_closest_canonical(nib.Nifti1Image(union.astype(np.float32), img.affine))
    fp_front = depth_render(np.asanyarray(u.dataobj), axis=1, from_end=True, thr=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, im, t in zip(axes, [fs_front, fp_front], ['FreeSurfer mri/*.mgz (union)', 'fMRIPrep anat masks/segs (union)']):
        ax.imshow(im.T, origin='lower', cmap='gray', vmin=0, vmax=1)
        ax.set_title(t, fontsize=9)
        ax.axis('off')
    fig.suptitle(f'sub-{subject}: other released anatomical volumes, frontal depth render', fontsize=9)
    fig.savefig(op.join(figdir, f'sub-{subject}_other_volumes.png'), dpi=70, bbox_inches='tight')
    plt.close(fig)
    return rows


def main(subject):
    subject = f'{int(subject):02d}'
    assert subject in get_subjects()
    figdir = op.join(WORK, 'qc', f'sub-{subject}')
    deface = pd.read_csv(op.join(WORK, 'qc', f'sub-{subject}_deface.tsv'), sep='\t', dtype={'subject': str})

    preproc_undefaced = to_float_img(nib.load(op.join(WORK, 'undefaced', 'fmriprep', f'sub-{subject}', 'anat',
                                                      f'sub-{subject}_desc-preproc_T1w.nii.gz')))
    brain_mask = nib.load(op.join(FMRIPREP_SRC, f'sub-{subject}', 'anat', f'sub-{subject}_desc-brain_mask.nii.gz'))

    rows = []
    for _, row in deface.iterrows():
        if row['kind'] == 'raw':
            extra = qc_raw_t1w(subject, row, preproc_undefaced, brain_mask, figdir)
        else:
            extra = qc_preproc_t1w(subject, row, brain_mask, figdir)
        rows.append({**row.to_dict(), **extra})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(op.join(WORK, 'qc', f'sub-{subject}_qc_deface.tsv'), sep='\t', index=False)

    other = qc_other_volumes(subject, figdir)
    pd.DataFrame(other).to_csv(op.join(WORK, 'qc', f'sub-{subject}_qc_other_volumes.tsv'), sep='\t', index=False)
    print(pd.DataFrame(other).to_string())
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject')
    args = parser.parse_args()
    main(args.subject)
