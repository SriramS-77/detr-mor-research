"""Generate every number and plot table the manuscript uses.

Reads the stored eval_results.json / routing_diagnostics_1000.json files and
the complexity JSONs in paper/analysis/out, and writes

* paper/generated/numbers.tex - ``\\val{key}`` lookups (AP in %, 2 d.p.)
* paper/generated/*.dat        - pgfplots tables

so no result is typed into main.tex by hand.

Example::

    python paper/analysis/make_tables.py
"""

import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(REPO, 'paper', 'analysis', 'out')
GEN = os.path.join(REPO, 'paper', 'generated')

R = 'rt_detr_all_results'
EVAL = {
    'rt2': R + '/rt_detr_voc_results (2 blocks)/eval_results.json',
    'rt6': R + '/rt_detr_new_results (6 blocks)/eval_results.json',
    'rt23': R + '/rt_detr_mor_voc_results (2 blocks, 3 recursions)/eval_results.json',
    'rt32': R + '/rt_detr_mor_voc_results (3 blocks, 2 recursions)/eval_results.json',
    'detr': 'detr_voc_results/detr_voc_results_300/eval_results.json',
    'cyc': 'mor_cyclic_voc_results/mor_cyclic_voc_results_300/eval_results.json',
    'seq': 'mor_sequential_voc_results_300/eval_results.json',
}
DIAG = {
    'cyc': 'mor_cyclic_voc_results/mor_cyclic_voc_results_300/routing_diagnostics_1000.json',
    'seq': 'mor_sequential_voc_results_300/routing_diagnostics_1000.json',
    'rt23': R + '/rt_detr_mor_voc_results (2 blocks, 3 recursions)/routing_diagnostics_1000.json',
    'rt32': R + '/rt_detr_mor_voc_results (3 blocks, 2 recursions)/routing_diagnostics_1000.json',
}
RT_COMPLEXITY = {'rt2': '2L', 'rt6': '6L', 'rt23': '2x3', 'rt32': '3x2'}
PAIRS = [('rt23', 'rt6'), ('rt32', 'rt6'), ('rt23', 'rt2'), ('rt32', 'rt2'),
         ('rt6', 'rt2'), ('cyc', 'detr'), ('seq', 'detr')]
IOUS = ['0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85',
        '0.90', '0.95']
SHORT_CLASS = {'aeroplane': 'plane', 'diningtable': 'table',
               'motorbike': 'mbike', 'pottedplant': 'plant',
               'tvmonitor': 'tv'}


def load(path):
    with open(os.path.join(REPO, path), encoding='utf-8') as handle:
        return json.load(handle)


def pct(x):
    return 100.0 * x


def fmt(x, digits=2, sign=False):
    text = '{:{}.{}f}'.format(x, '+' if sign else '', digits)
    return text.replace('-', '\\textminus ') if sign or x < 0 else text


def main():
    os.makedirs(GEN, exist_ok=True)
    vals = {}
    ev = {k: load(p) for k, p in EVAL.items()}
    classes = [c for c, v in ev['rt6']['class_ap'].items() if v is not None]

    # --- headline AP and per-IoU AP --------------------------------------
    for k, d in ev.items():
        vals[k + '.ap'] = fmt(pct(d['ap']))
        for t in IOUS:
            vals['{}.ap{}'.format(k, t[2:])] = fmt(pct(d['per_iou'][t]['mean_ap']))
    for a, b in PAIRS:
        key = 'd.{}-{}'.format(a, b)
        vals[key + '.ap'] = fmt(pct(ev[a]['ap'] - ev[b]['ap']), sign=True)
        for t in IOUS:
            diff = ev[a]['per_iou'][t]['mean_ap'] - ev[b]['per_iou'][t]['mean_ap']
            vals['{}.ap{}'.format(key, t[2:])] = fmt(pct(diff), sign=True)
        wins = sum(ev[a]['class_ap'][c] > ev[b]['class_ap'][c] for c in classes)
        vals['win.{}-{}'.format(a, b)] = str(wins)
        vals['loss.{}-{}'.format(a, b)] = str(len(classes) - wins)
        # range of per-IoU differences between IoU 0.70 and 0.85
        mid = [pct(ev[a]['per_iou'][t]['mean_ap'] - ev[b]['per_iou'][t]['mean_ap'])
               for t in ('0.70', '0.75', '0.80', '0.85')]
        vals[key + '.midmin'] = fmt(min(mid), sign=True)
        vals[key + '.midmax'] = fmt(max(mid), sign=True)
        diffs = [pct(ev[a]['per_iou'][t]['mean_ap'] - ev[b]['per_iou'][t]['mean_ap'])
                 for t in IOUS]
        vals[key + '.iouminabs'] = fmt(min(abs(x) for x in diffs))
        vals[key + '.ioumaxabs'] = fmt(max(abs(x) for x in diffs))
    vals['nclasses'] = str(len(classes))
    # unsigned magnitudes for prose such as "loses 1.50 AP"
    for key in [k for k in vals if k.startswith('d.')]:
        diff_text = vals[key].replace('\\textminus ', '-')
        vals['a.' + key[2:]] = fmt(abs(float(diff_text)))

    # --- complexity --------------------------------------------------------
    rt = {}
    for k, name in RT_COMPLEXITY.items():
        res = load('paper/analysis/out/complexity_rtdetr_{}.json'.format(name))['results'][0]
        p, g = res['params_m'], res['gflops']
        dec_p = p['decoder'] + p['heads_post']
        rt[k] = dict(params=p['total'], dec=dec_p, gflops=g['total'],
                     decg=g['decoder'], vproj=g['decoder.value_proj'])
        vals[k + '.params'] = fmt(p['total'])
        vals[k + '.decparams'] = fmt(dec_p)
        vals[k + '.gflops'] = fmt(g['total'], 1)
        vals[k + '.decgflops'] = fmt(g['decoder'])
        vals[k + '.vprojgflops'] = fmt(g['decoder.value_proj'])
        vals[k + '.querygflops'] = fmt(g['decoder'] - g['decoder.value_proj'])
        vals[k + '.backbone'] = fmt(p['backbone'])
    for k in ('rt23', 'rt32', 'rt2'):
        vals[k + '.decred'] = fmt(100 * (1 - rt[k]['dec'] / rt['rt6']['dec']), 0)
        vals[k + '.totred'] = fmt(100 * (1 - rt[k]['params'] / rt['rt6']['params']), 1)
    # value_proj is identical across recursions of a shared block with a fixed
    # memory; caching it once per unique block would leave this much decoder
    # compute (2x3: 2 unique blocks)
    per_block_vproj = rt['rt2']['vproj'] / 2
    vals['rt23.cachedgflops'] = fmt(rt['rt23']['decg'] - rt['rt23']['vproj']
                                    + 2 * per_block_vproj)
    vals['rt32.cachedgflops'] = fmt(rt['rt32']['decg'] - rt['rt32']['vproj']
                                    + 3 * per_block_vproj)

    dc = load('paper/analysis/out/complexity_detr.json')
    for k, name in (('detr', 'DETR'), ('cyc', 'MoR-cyclic'),
                    ('seq', 'MoR-sequential')):
        p, g = dc[name]['params'], dc[name]['gflops']
        vals[k + '.params'] = fmt(p['total'] / 1e6)
        vals[k + '.trainable'] = fmt(p['trainable'] / 1e6)
        vals[k + '.tfparams'] = fmt(p['transformer'] / 1e6)
        vals[k + '.tfgflops'] = fmt(g['encoder'] + g['decoder'])
        vals[k + '.encgflops'] = fmt(g['encoder'])
        vals[k + '.decgflops'] = fmt(g['decoder'])
        vals[k + '.gflops'] = fmt(g['total'], 1)
        vals[k + '.routers'] = str(p['routers'])
    for k, name in (('cyc', 'MoR-cyclic'), ('seq', 'MoR-sequential')):
        base, mor = dc['DETR'], dc[name]
        vals[k + '.trainred'] = fmt(100 * (1 - mor['params']['trainable']
                                           / base['params']['trainable']), 0)
        tf = lambda d: d['gflops']['encoder'] + d['gflops']['decoder']  # noqa: E731
        vals[k + '.tfgflopsred'] = fmt(100 * (1 - tf(mor) / tf(base)), 0)

    # --- routing probes (first 1,000 test images) ---------------------------
    probe_rows = []
    for k, path in DIAG.items():
        d = load(path)
        maps = d['maps']
        base = maps['default']['ap']
        vals['probe.{}.default'.format(k)] = fmt(pct(base))
        names = {'full': 'full_both', 'random': 'random_both',
                 'oracle': 'oracle_decoder'} if k in ('cyc', 'seq') else \
            {'full': 'full', 'random': 'random', 'oracle': 'oracle'}
        row = [k]
        for short, name in names.items():
            delta = pct(maps[name]['ap'] - base)
            vals['probe.{}.{}'.format(k, short)] = fmt(delta, sign=True)
            row.append('{:.3f}'.format(delta))
        if k in ('cyc', 'seq'):
            for extra in ('full_encoder', 'full_decoder', 'random_decoder'):
                vals['probe.{}.{}'.format(k, extra.replace('_', ''))] = fmt(
                    pct(maps[extra]['ap'] - base), sign=True)
        probe_rows.append(row)
        deltas = [abs(pct(m['ap'] - base)) for n, m in maps.items()
                  if n != 'default']
        vals['probe.{}.maxabs'.format(k)] = fmt(max(deltas))
        vals['probe.{}.capacity'.format(k)] = '\\textrightarrow{}'.join(
            str(c) for c in d['decoder_capacity'])
        vals['probe.{}.nobj'.format(k)] = '{:,}'.format(d['num_matched_objects']).replace(',', '{,}')
        for row_d in d['decoder_depth']:
            key = 'depth.{}.{}'.format(k, row_d['depth'])
            vals[key + '.share'] = fmt(100 * row_d['share'], 0)
            vals[key + '.iou'] = fmt(row_d['mean_iou'])
            vals[key + '.iou75'] = fmt(100 * row_d['frac_iou_ge_0.75'], 0)
            vals[key + '.prob'] = fmt(row_d['mean_gt_prob'])
            vals[key + '.objs'] = fmt(row_d['mean_objects_in_image'], 1)
        for row_f in d.get('encoder_foreground', []):
            key = 'fg.{}.{}'.format(k, row_f['stage'])
            vals[key + '.ratio'] = fmt(row_f['fg_over_bg'])
            vals[key + '.fg'] = fmt(100 * row_f['fg_kept_rate'], 0)
            vals[key + '.bg'] = fmt(100 * row_f['bg_kept_rate'], 0)
            vals[key + '.cap'] = fmt(100 * row_f['capacity_fraction'], 0)
        if d.get('encoder_foreground'):
            ratios = [r['fg_over_bg'] for r in d['encoder_foreground']]
            vals['fg.{}.min'.format(k)] = fmt(min(ratios))
            vals['fg.{}.max'.format(k)] = fmt(max(ratios))
    vals['probe.maxabs.all'] = fmt(max(float(vals['probe.{}.maxabs'.format(k)])
                                       for k in DIAG))

    # --- optional bootstrap intervals --------------------------------------
    boot_path = os.path.join(OUT, 'bootstrap.json')
    if os.path.exists(boot_path):
        with open(boot_path, encoding='utf-8') as handle:
            boot = json.load(handle)
        for pair, metrics in boot['pairs'].items():
            for metric, ci in metrics.items():
                key = 'ci.{}.{}'.format(pair, metric)
                vals[key + '.lo'] = fmt(ci['lo'], sign=True)
                vals[key + '.hi'] = fmt(ci['hi'], sign=True)
                vals[key + '.p'] = '{:.3f}'.format(ci['p_le_0'])
                if ci['lo'] > 0 or ci['hi'] < 0:
                    vals[key + '.sig'] = '1'  # interval excludes zero
        vals['ci.B'] = str(boot['B'])

    # --- write numbers.tex ---------------------------------------------------
    lines = ['% Generated by paper/analysis/make_tables.py - do not edit.',
             '\\makeatletter',
             '\\newcommand{\\val}[1]{\\ifcsname val@#1\\endcsname'
             '\\csname val@#1\\endcsname\\else'
             '\\textbf{\\textcolor{red}{??#1}}\\PackageWarning{numbers}'
             '{Undefined value #1}\\fi}',
             '\\newcommand{\\hasval}[1]{\\ifcsname val@#1\\endcsname'
             '\\expandafter\\@firstoftwo\\else\\expandafter\\@secondoftwo\\fi}']
    for key in sorted(vals):
        lines.append('\\expandafter\\def\\csname val@{}\\endcsname{{{}}}'.format(
            key, vals[key]))
    lines.append('\\makeatother')
    with open(os.path.join(GEN, 'numbers.tex'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')

    # --- pgfplots tables -----------------------------------------------------
    with open(os.path.join(GEN, 'periou.dat'), 'w', encoding='utf-8') as handle:
        handle.write('iou rt2 rt6 rt23 rt32 detr cyc seq '
                     'drt2 drt23 drt32 dcyc dseq\n')
        for t in IOUS:
            a = {k: pct(ev[k]['per_iou'][t]['mean_ap']) for k in ev}
            row = [a[k] for k in ('rt2', 'rt6', 'rt23', 'rt32', 'detr', 'cyc',
                                  'seq')]
            row += [a['rt2'] - a['rt6'], a['rt23'] - a['rt6'],
                    a['rt32'] - a['rt6'], a['cyc'] - a['detr'],
                    a['seq'] - a['detr']]
            handle.write(t + ' ' + ' '.join('{:.3f}'.format(x) for x in row)
                         + '\n')
    with open(os.path.join(GEN, 'perclass.dat'), 'w', encoding='utf-8') as handle:
        handle.write('cls drt23 drt32 dcyc dseq\n')
        for c in classes:
            handle.write('{} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(
                SHORT_CLASS.get(c, c),
                pct(ev['rt23']['class_ap'][c] - ev['rt6']['class_ap'][c]),
                pct(ev['rt32']['class_ap'][c] - ev['rt6']['class_ap'][c]),
                pct(ev['cyc']['class_ap'][c] - ev['detr']['class_ap'][c]),
                pct(ev['seq']['class_ap'][c] - ev['detr']['class_ap'][c])))
    with open(os.path.join(GEN, 'depth.dat'), 'w', encoding='utf-8') as handle:
        handle.write('stage cycshare cyciou cycobjs rtshare rtiou rtobjs\n')
        cyc = {r['depth']: r for r in load(DIAG['cyc'])['decoder_depth']}
        rt23 = {r['depth']: r for r in load(DIAG['rt23'])['decoder_depth']}
        for s in (1, 2, 3):
            handle.write('{} {:.2f} {:.4f} {:.3f} {:.2f} {:.4f} {:.3f}\n'.format(
                s, 100 * cyc[s]['share'], cyc[s]['mean_iou'],
                cyc[s]['mean_objects_in_image'], 100 * rt23[s]['share'],
                rt23[s]['mean_iou'], rt23[s]['mean_objects_in_image']))
    with open(os.path.join(GEN, 'probes.dat'), 'w', encoding='utf-8') as handle:
        handle.write('model full random oracle\n')
        for row in probe_rows:
            handle.write(' '.join(row) + '\n')
    # The same tables as inline pgfplots macros, so the manuscript needs only
    # numbers.tex (Springer asks for a single .tex source).
    inline = ['% pgfplots tables (same content as the .dat files)']
    for name, macro in (('periou', 'tblPerIoU'), ('perclass', 'tblPerClass'),
                        ('depth', 'tblDepth'), ('probes', 'tblProbes')):
        with open(os.path.join(GEN, name + '.dat'), encoding='utf-8') as handle:
            rows = [line.strip() for line in handle if line.strip()]
        inline.append('\\pgfplotstableread[row sep=\\\\]{{{}\\\\}}\\{}'.format(
            ' \\\\ '.join(rows), macro))
    with open(os.path.join(GEN, 'numbers.tex'), 'a', encoding='utf-8') as handle:
        handle.write('\n'.join(inline) + '\n')
    print('wrote {} values to {}'.format(len(vals), GEN))


if __name__ == '__main__':
    main()
