"""Shared constants and helpers for the OpenNeuro release of ds-neuralpriors.

Layout (all on the cluster share):
- SOURCE : the canonical working dataset. Only ever read, never written.
- TARGET : the upload tree. Only de-identified, defaced data may land here.
- WORK   : scratch for the release. Holds the *undefaced* anatomicals
           (header-cleaned), the pydeface masks and the QC output. It is
           never uploaded and can be deleted once the release is published.
"""
import gzip
import os
import os.path as op
import re
import shutil

import nibabel as nib
import numpy as np
import yaml

SOURCE = '/shares/zne.uzh/gdehol/ds-neuralpriors'
TARGET = '/shares/zne.uzh/gdehol/ds-neuralpriors-openneuro'
WORK = '/shares/zne.uzh/gdehol/ds-neuralpriors-openneuro-work'

# BIDS task label: 'task-task' in the working dataset, renamed for the release.
TASK_OLD = 'task'
TASK = 'numestimation'
TASK_NAME = 'Numerosity estimation'

# Free-text NIfTI-1 header fields; cleared in every released raw image.
NIFTI_TEXT_FIELDS = ['descrip', 'aux_file', 'db_name', 'intent_name']

REPO_SUBJECTS_YML = op.join(op.dirname(op.dirname(__file__)), 'data', 'subjects.yml')


def get_subjects():
    """Released subjects = the paper's sample (two sessions in subjects.yml):
    sub-01..sub-41 without sub-11 and sub-23. Pilots are never listed there."""
    with open(REPO_SUBJECTS_YML) as f:
        mapping = yaml.safe_load(f)
    subjects = sorted(k for k, v in mapping.items() if len(v) > 1)
    assert all(re.fullmatch(r'\d\d', s) for s in subjects), subjects
    return subjects


def rename_task(s):
    return s.replace(f'task-{TASK_OLD}_', f'task-{TASK}_')


def makedirs_for(fn):
    os.makedirs(op.dirname(fn), exist_ok=True)


def _open(fn):
    return gzip.open(fn, 'rb') if fn.endswith('.gz') else open(fn, 'rb')


def copy_nifti_clean_header(src, dst, data_transform=None):
    """Copy a NIfTI-1 image to `dst` (.nii.gz) with the free-text header fields
    blanked, byte-for-byte otherwise.

    The 348-byte header is re-serialised; extensions and the voxel data are
    copied unchanged, so datatype, scl_slope/inter, qform/sform and every voxel
    stay identical (no nibabel round-trip of the data, no rescaling).

    data_transform: optional callable(raw_array) -> raw_array applied to the
    *unscaled* voxel array (used for defacing: zero out face voxels in the
    stored datatype).
    """
    assert dst.endswith('.nii.gz'), dst
    with _open(src) as f:
        raw = f.read()
    hdr = nib.Nifti1Header.from_fileobj(_BytesReader(raw), check=False)
    for field in NIFTI_TEXT_FIELDS:
        hdr[field] = b''
    vox_offset = int(hdr['vox_offset'])
    head = hdr.binaryblock + raw[348:vox_offset]

    if data_transform is None:
        body = raw[vox_offset:]
    else:
        dtype = hdr.get_data_dtype()
        shape = hdr.get_data_shape()
        n = int(np.prod(shape)) * dtype.itemsize
        arr = np.frombuffer(raw[vox_offset:vox_offset + n], dtype=dtype).reshape(shape, order='F')
        new = np.asarray(data_transform(arr.copy()), dtype=dtype)
        assert new.shape == arr.shape
        body = new.tobytes(order='F') + raw[vox_offset + n:]

    makedirs_for(dst)
    tmp = dst + '.part'
    with gzip.open(tmp, 'wb', compresslevel=6) as f:
        f.write(head)
        f.write(body)
    os.replace(tmp, dst)


class _BytesReader:
    def __init__(self, b):
        self.b, self.pos = b, 0

    def read(self, n=-1):
        out = self.b[self.pos:] if n < 0 else self.b[self.pos:self.pos + n]
        self.pos += len(out)
        return out

    def seek(self, pos, whence=0):
        self.pos = pos if whence == 0 else (self.pos + pos if whence == 1 else len(self.b) + pos)

    def tell(self):
        return self.pos


def verify_same_image(src, dst, allow_changed_voxels=None):
    """Assert that dst has the same geometry, datatype, scaling and voxel values
    as src. allow_changed_voxels: boolean array of voxels that may differ
    (defacing); those must be 0 in dst."""
    a, b = nib.load(src), nib.load(dst)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.get_data_dtype() == b.get_data_dtype(), (a.get_data_dtype(), b.get_data_dtype())
    assert np.array_equal(a.header.get_qform(), b.header.get_qform())
    assert np.array_equal(a.header.get_sform(), b.header.get_sform())
    assert a.header.get_slope_inter() == b.header.get_slope_inter() or \
        all(np.isnan(x) for x in a.header.get_slope_inter() + b.header.get_slope_inter())
    ua = np.asanyarray(a.dataobj.get_unscaled())
    ub = np.asanyarray(b.dataobj.get_unscaled())
    if allow_changed_voxels is None:
        assert np.array_equal(ua, ub), f'voxel data differ: {src} vs {dst}'
    else:
        keep = ~allow_changed_voxels
        assert np.array_equal(ua[keep], ub[keep]), f'kept voxels differ: {src} vs {dst}'
        assert not np.any(ub[allow_changed_voxels]), f'removed voxels not zero in {dst}'
    for field in NIFTI_TEXT_FIELDS:
        assert not b.header[field].tobytes().strip(b'\x00'), (dst, field)


def copy_file(src, dst):
    makedirs_for(dst)
    shutil.copyfile(src, dst)  # follows symlinks; no metadata / timestamps
