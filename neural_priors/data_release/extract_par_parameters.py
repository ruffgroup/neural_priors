"""Tabulate acquisition parameters from the Philips PAR headers (sourcedata/mri).

Reads ONLY technical fields (general-info lines for TR, water-fat shift, EPI factor,
scan resolution, FOV, number of dynamics; first image-definition row for slice
thickness/gap, TE, flip angle, turbo factor). Patient name, examination name,
protocol name and dates are never read or written. Output: one row per PAR file,
keyed by subject / session / acquisition number / scan type.

Usage:
    python -m neural_priors.data_release.extract_par_parameters <sourcedata/mri> <out.tsv>
"""
import argparse
import glob
import os.path as op
import re

import pandas as pd

GENERAL = {
    'tr_ms': r'Repetition time \[msec\]',
    'water_fat_shift_px': r'Water Fat shift \[pixels\]',
    'epi_factor': r'EPI factor',
    'scan_resolution': r'Scan resolution\s+\(x, y\)',
    'fov_ap_fh_rl_mm': r'FOV \(ap,fh,rl\) \[mm\]',
    'n_dynamics': r'Max\. number of dynamics',
    'n_slices': r'Max\. number of slices/locations',
    'technique': r'Technique',
    'scan_mode': r'Scan mode',
}
# 1-based column positions in the PAR v4.2 image-definition rows
ROW = {'slice_thickness_mm': 23, 'slice_gap_mm': 24, 'te_ms': 31, 'flip_angle_deg': 36, 'turbo_factor': 40}


def parse(fn):
    out = {}
    with open(fn, 'r', errors='replace') as f:
        lines = f.read().splitlines()
    for key, pat in GENERAL.items():
        line = next((l for l in lines if l.startswith('.') and re.search(pat, l)), None)
        out[key] = line.split(':', 1)[1].strip() if line else None
    row = next(l for l in lines if re.match(r'^\s*\d', l)).split()
    for key, col in ROW.items():
        out[key] = row[col - 1]
    return out


def main(mri_dir, out_tsv):
    rows = []
    for fn in sorted(glob.glob(op.join(mri_dir, 'SNS_MRI_NJM_S*s*', '*.par'))):
        m = re.search(r'SNS_MRI_NJM_S(\d+)s(\d)', fn)
        # sn_<date>_<time>_<acquisition>_<reconstruction>_<name>.par (date/time not kept)
        a = re.match(r'^sn_\d+_\d+_(\d+)_(\d+)_(.+)\.par$', op.basename(fn))
        if not m or not a or not re.search(r'(t1w3danat|_run\d+_spli)', a.group(3)):
            continue
        rows.append(dict(subject=f'{int(m.group(1)):02d}', session=int(m.group(2)),
                         acquisition=int(a.group(1)),
                         scan='T1w' if 't1w' in a.group(3) else 'bold',
                         run=int(re.search(r'run(\d+)', a.group(3)).group(1)) if 'run' in a.group(3) else None,
                         **parse(fn)))
    df = pd.DataFrame(rows)
    df.to_csv(out_tsv, sep='\t', index=False)
    print(df.drop(columns=['subject', 'session', 'acquisition', 'run']).astype(str).value_counts().to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mri_dir')
    p.add_argument('out_tsv')
    a = p.parse_args()
    main(a.mri_dir, a.out_tsv)
