"""Assemble the defacing QC into one local HTML page for visual review.

Run locally after syncing WORK/qc from the cluster:
    rsync -a sciencecluster:<WORK>/qc/ <local_qc_dir>/
    python -m neural_priors.data_release.make_qc_report <local_qc_dir>
Writes <local_qc_dir>/index.html (images linked relatively). The page shows
defaced participant anatomy: keep it local, never publish it.
"""
import argparse
import glob
import html
import os.path as op

import pandas as pd


def main(qc_dir):
    deface = pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str})
                        for f in sorted(glob.glob(op.join(qc_dir, 'sub-*_qc_deface.tsv')))])
    other = pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str})
                       for f in sorted(glob.glob(op.join(qc_dir, 'sub-*_qc_other_volumes.tsv')))])

    deface['flag'] = ''
    deface.loc[deface['brain_voxels_removed'] > 0, 'flag'] += 'BRAIN CLIPPED; '
    deface.loc[deface['dilated_brain_voxels_removed'] > 0, 'flag'] += 'removal within 3 mm of brain; '
    deface.loc[deface['frac_head_voxels_removed'] < 0.01, 'flag'] += 'almost nothing removed; '
    deface.loc[deface['r_orig_vs_preproc'] < 0.9, 'flag'] += 'transform check r < 0.9; '
    other['flag'] = (other['n_outside_dilated_brainmask'] > 0).map({True: 'voxels outside brain', False: ''})

    parts = ['<html><head><meta charset="utf-8"><title>Defacing QC</title><style>'
             'body{font-family:sans-serif;margin:16px} img{max-width:100%;display:block;margin:4px 0 14px}'
             'table{border-collapse:collapse;font-size:12px} td,th{border:1px solid #ccc;padding:2px 6px}'
             '.flag{color:#b00;font-weight:bold}</style></head><body>',
             f'<h1>Defacing QC</h1><p>{deface["subject"].nunique()} subjects, {len(deface)} defaced images '
             f'({(deface["kind"] == "raw").sum()} raw T1w, {(deface["kind"] == "fmriprep").sum()} fMRIPrep T1w), '
             f'{len(other)} other released anatomical volumes.</p>',
             '<p>For every image check: no recognisable face in the frontal and lateral depth renders '
             '(nose, lips, eyes, chin gone); red region on the mid-sagittal slice stays in front of / below '
             'the brain. Other volumes: the union render must look like a brain, not a head.</p>',
             '<h2>Flagged</h2>']
    flagged = deface[deface['flag'] != '']
    parts.append(flagged.to_html(index=False, float_format='%.3f') if len(flagged) else '<p>None</p>')
    flagged_other = other[other['flag'] != '']
    parts.append('<h3>Other volumes with voxels outside the dilated brain mask</h3>')
    parts.append(flagged_other.to_html(index=False) if len(flagged_other) else '<p>None</p>')
    parts.append('<h2>All metrics</h2>' + deface.to_html(index=False, float_format='%.3f'))

    for subject, rows in deface.groupby('subject'):
        parts.append(f'<h2 id="sub-{subject}">sub-{subject}</h2>')
        for _, r in rows.iterrows():
            png = op.join(f'sub-{subject}', op.basename(r['file']).replace('.nii.gz', '.png'))
            cls = ' class="flag"' if r['flag'] else ''
            parts.append(f'<div{cls}>{html.escape(r["file"])} &mdash; removed {r["frac_head_voxels_removed"]:.1%} '
                         f'of head voxels, brain voxels removed: {r["brain_voxels_removed"]} {html.escape(r["flag"])}</div>'
                         f'<img loading="lazy" src="{png}">')
        parts.append(f'<img loading="lazy" src="sub-{subject}/sub-{subject}_other_volumes.png">')
    parts.append('</body></html>')
    out = op.join(qc_dir, 'index.html')
    with open(out, 'w') as f:
        f.write('\n'.join(parts))
    print(out)
    print(flagged[['file', 'flag']].to_string() if len(flagged) else 'no flagged defaced images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('qc_dir')
    main(parser.parse_args().qc_dir)
