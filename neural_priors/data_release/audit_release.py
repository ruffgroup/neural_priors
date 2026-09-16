"""Whole-tree de-identification audit of the OpenNeuro upload tree (TARGET).

Blocking checks (exit 1 on failure):
1. Subject folders (raw, fmriprep, freesurfer) are exactly the released subjects.
2. Every file matches an allowlisted path pattern.
3. No old task label / pilot names in file names.
4. Every high-resolution (<= 1.5 mm) 3D image is either a defaced image that
   passed QC (WORK/qc/*_qc_deface.tsv) or a brain-only volume that passed the
   outside-brain-mask check (WORK/qc/*_qc_other_volumes.tsv).
5. Raw NIfTI free-text header fields are empty; no NIfTI header extensions.

Reported for manual review (WORK/audit/):
- regex hits for names, dates/times, file-system paths, e-mail/IP addresses and
  DICOM-style identifying keys, in text files and in the strings of binary
  files (NIfTI headers, MGZ tags, GIfTI metadata, FreeSurfer surface/annot files);
- inventory of all JSON keys; file counts and sizes per type.
"""
import collections
import glob
import gzip
import json
import os
import os.path as op
import re
import sys

import nibabel as nib
import pandas as pd

from neural_priors.data_release.release import NIFTI_TEXT_FIELDS, TARGET, TASK, WORK, get_subjects

SUBJ = r'sub-(\d\d)'
SES = r'ses-[12]'
ALLOWED = [
    r'dataset_description\.json', r'README', r'CHANGES', r'participants\.(tsv|json)',
    rf'task-{TASK}_(bold|events)\.json', r'T1w\.json',
    rf'{SUBJ}/{SES}/anat/sub-\d\d_{SES}_run-\d_T1w\.nii\.gz',
    rf'{SUBJ}/{SES}/func/sub-\d\d_{SES}_task-{TASK}_run-\d_(bold\.(nii\.gz|json)|events\.tsv)',
    rf'{SUBJ}/{SES}/fmap/sub-\d\d_{SES}_dir-(LR|RL)_run-\d_epi\.(nii\.gz|json)',
    r'derivatives/fmriprep/(dataset_description\.json|README|desc-(aseg|aparcaseg)_dseg\.tsv|CITATION\.(md|bib))',
    rf'derivatives/fmriprep/{SUBJ}/anat/sub-\d\d_[A-Za-z0-9_-]+\.(nii\.gz|json|txt|surf\.gii|shape\.gii)',
    rf'derivatives/fmriprep/{SUBJ}/{SES}/func/sub-\d\d_{SES}_task-{TASK}_run-\d_[A-Za-z0-9_-]+\.(nii\.gz|json|tsv)',
    rf'derivatives/fmriprep/sourcedata/freesurfer/{SUBJ}/mri/[A-Za-z0-9.+_-]+\.mgz',
    rf'derivatives/fmriprep/sourcedata/freesurfer/{SUBJ}/mri/transforms/talairach\.(xfm|lta)',
    rf'derivatives/fmriprep/sourcedata/freesurfer/{SUBJ}/label/[A-Za-z0-9.+_-]+\.(label|annot|ctab)',
    rf'derivatives/fmriprep/sourcedata/freesurfer/{SUBJ}/surf/[lr]h\.[A-Za-z.]+',
]
TEXT_EXT = ('.json', '.tsv', '.md', '.bib', '.label', '.ctab', '.xfm', '.lta', '.txt', 'README', 'CHANGES')

PATTERNS = {
    'name': r'(?i)\b(alina|ella|maike|saurabh|jacob|gilles|hollander|gdehol|bedi|prat|carrabin|gershman|ruff)\b',
    'date_iso': r'\b(19|20)\d\d[-/._](0[1-9]|1[0-2])[-/._](0[1-9]|[12]\d|3[01])\b',
    'date_eu': r'\b(0[1-9]|[12]\d|3[01])[-/.](0[1-9]|1[0-2])[-/.](19|20)\d\d\b',
    'date_compact': r'(?<!\d)20(1\d|2[0-6])(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)',
    'weekday_time': r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b[^\n]{0,25}\b\d\d:\d\d',
    'month_year': r'(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+(19|20)\d\d\b',
    'path': r'(/home/|/Users/|/shares/|/scratch/|/storage/|/workflow|idnas|\\\\[A-Za-z])',
    'email': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}',
    'ip': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    'dicom_id': r'(?i)(patient|birth|\bdob\b|AcquisitionDate|AcquisitionTime|ContentDate|SeriesDate|StudyDate|'
                r'InstitutionName|InstitutionAddress|StationName|DeviceSerialNumber|OperatorsName|Physician)',
}
# Expected, reviewed-once hits (authors/citations in the dataset-level docs).
EXPECTED_NAME_FILES = {'dataset_description.json', 'derivatives/fmriprep/dataset_description.json',
                       'README', 'derivatives/fmriprep/CITATION.md', 'derivatives/fmriprep/CITATION.bib'}


def rel_files():
    out = []
    for root, dirs, files in os.walk(TARGET):
        for f in files:
            out.append(op.relpath(op.join(root, f), TARGET))
        for d in dirs:
            if op.islink(op.join(root, d)):
                out.append(op.relpath(op.join(root, d), TARGET) + '  [SYMLINK DIR]')
    return sorted(out)


def nifti_header_bytes(fn):
    with gzip.open(fn, 'rb') as f:
        head = f.read(352)
        hdr = nib.Nifti1Header.from_fileobj(__import__('io').BytesIO(head[:348]), check=False)
        vox_offset = int(hdr['vox_offset'])
        rest = f.read(max(0, vox_offset - 352))
    return hdr, head + rest


def binary_strings(b, minlen=5):
    return [m.group().decode('ascii', 'replace') for m in re.finditer(rb'[\x20-\x7e]{%d,}' % minlen, b)]


def scan_text(rel, text, hits):
    for key, pat in PATTERNS.items():
        for m in re.finditer(pat, text):
            s = max(0, m.start() - 40)
            hits.append(dict(file=rel, pattern=key, match=m.group(), context=text[s:m.end() + 40].replace('\n', ' ')))


def main():
    problems = []
    hits = []
    subjects = {f'sub-{s}' for s in get_subjects()}

    # 1. subject folders
    for base in ['', 'derivatives/fmriprep', 'derivatives/fmriprep/sourcedata/freesurfer']:
        found = {op.basename(d) for d in glob.glob(op.join(TARGET, base, 'sub-*'))}
        if found != subjects:
            problems.append(f'subject folders in "{base or "."}": extra {sorted(found - subjects)}, '
                            f'missing {sorted(subjects - found)}')

    files = rel_files()
    json_keys = collections.Counter()
    sizes = collections.defaultdict(lambda: [0, 0])
    highres = []
    for rel in files:
        fn = op.join(TARGET, rel)
        ext = re.sub(r'^.*?((\.[a-z]+)+|README|CHANGES)$', r'\1', op.basename(rel))
        sizes[ext][0] += 1
        sizes[ext][1] += op.getsize(fn) if op.isfile(fn) else 0

        # 2 + 3. allowlist and names
        if not any(re.fullmatch(p, rel) for p in ALLOWED):
            problems.append(f'not allowlisted: {rel}')
        if 'task-task' in rel or re.search(PATTERNS['name'], rel):
            problems.append(f'bad file name: {rel}')

        if rel.endswith(TEXT_EXT):
            with open(fn, errors='replace') as f:
                text = f.read()
            if rel.endswith('.label'):
                text = text.split('\n', 2)[0]   # coordinates below the header line
            scan_text(rel, text, hits)
            if rel.endswith('.json'):
                with open(fn) as f:
                    json_keys.update(json.load(f).keys())
        elif rel.endswith('.nii.gz'):
            hdr, b = nifti_header_bytes(fn)
            scan_text(rel, '\n'.join(binary_strings(b)), hits)
            if len(b) > 352 and b[348:352] != b'\x00\x00\x00\x00':
                problems.append(f'NIfTI header extension present: {rel}')
            if not rel.startswith('derivatives/'):
                for field in NIFTI_TEXT_FIELDS:
                    if hdr[field].tobytes().strip(b'\x00'):
                        problems.append(f'raw NIfTI {field} not empty: {rel}')
            zooms, shape = hdr.get_zooms(), hdr.get_data_shape()
            if len(shape) == 3 or (len(shape) == 4 and shape[3] == 1):
                if min(zooms[:3]) <= 1.5:
                    highres.append(rel)
        elif rel.endswith('.mgz'):
            with gzip.open(fn, 'rb') as f:
                b = f.read()
            scan_text(rel, '\n'.join(s for s in binary_strings(b, 8)), hits)
            highres.append(rel)
        elif rel.endswith('.gii'):
            img = nib.load(fn)
            meta = [f'{k}={v}' for k, v in img.meta.items()]
            for da in img.darrays:
                meta += [f'{k}={v}' for k, v in da.meta.items()]
            scan_text(rel, '\n'.join(meta), hits)
        else:   # FreeSurfer binaries: surfaces, curv, annot
            with open(fn, 'rb') as f:
                b = f.read()
            scan_text(rel, '\n'.join(binary_strings(b, 6)), hits)

    # 4. high-resolution images must be QC-approved
    qc_deface = pd.concat([pd.read_csv(f, sep='\t') for f in glob.glob(op.join(WORK, 'qc', 'sub-*_qc_deface.tsv'))])
    qc_other = pd.concat([pd.read_csv(f, sep='\t') for f in glob.glob(op.join(WORK, 'qc', 'sub-*_qc_other_volumes.tsv'))])
    approved_defaced = set(qc_deface.loc[(qc_deface['brain_voxels_removed'] == 0) &
                                         (qc_deface['frac_head_voxels_removed'] > 0.01), 'file'])
    approved_other = set(qc_other.loc[qc_other['n_outside_dilated_brainmask'] <= 100, 'file'])
    for rel in highres:
        if rel not in approved_defaced and rel not in approved_other:
            problems.append(f'high-res image without passing QC: {rel}')
    n_defaced_expected = sum(bool(re.search(r'(_T1w|desc-preproc_T1w)\.nii\.gz$', r)) for r in files)
    if len(approved_defaced & set(files)) != n_defaced_expected:
        problems.append(f'defaced images passing QC: {len(approved_defaced & set(files))} of {n_defaced_expected}')

    outdir = op.join(WORK, 'audit')
    os.makedirs(outdir, exist_ok=True)
    hits = pd.DataFrame(hits, columns=['file', 'pattern', 'match', 'context'])
    hits['expected'] = (hits['pattern'] == 'name') & hits['file'].isin(EXPECTED_NAME_FILES)
    hits.to_csv(op.join(outdir, 'identifier_hits.tsv'), sep='\t', index=False)

    with open(op.join(outdir, 'audit_report.md'), 'w') as f:
        f.write(f'# Audit of {TARGET}\n\n{len(files)} files, {sum(v[1] for v in sizes.values()) / 1e9:.1f} GB\n\n')
        f.write('## Blocking problems\n\n' + ('\n'.join(f'- {p}' for p in problems) or 'None') + '\n\n')
        f.write('## Identifier-pattern hits (unexpected), grouped\n\n')
        unexp = hits[~hits['expected']]
        if len(unexp):
            unexp = unexp.assign(kind=unexp['file'].str.replace(r'sub-\d\d', 'sub-XX', regex=True)
                                                  .str.replace(r'ses-\d', 'ses-Y', regex=True)
                                                  .str.replace(r'run-\d', 'run-N', regex=True))
            g = unexp.groupby(['pattern', 'kind', 'match']).agg(n=('file', 'size'), example=('context', 'first'))
            f.write(g.reset_index().to_markdown(index=False) + '\n\n')
        else:
            f.write('None\n\n')
        f.write('## Expected hits (author names in dataset-level docs)\n\n')
        f.write(hits[hits['expected']].groupby(['file', 'match']).size().to_string() + '\n\n')
        f.write('## JSON keys\n\n' + '\n'.join(f'- {k}: {v}' for k, v in sorted(json_keys.items())) + '\n\n')
        f.write('## Files per type\n\n' + '\n'.join(f'- {k}: {v[0]} files, {v[1] / 1e9:.2f} GB'
                                                   for k, v in sorted(sizes.items())) + '\n')
    print(open(op.join(outdir, 'audit_report.md')).read())
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
