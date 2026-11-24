"""Render paper v0.6 figures and table fragments from the frozen Windows audit."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import numpy as np


HERE = Path(__file__).resolve().parent
SCALES = ('10k', '50k', '100k', '250k', '500k', 'full')
LABELS = ('10K', '50K', '100K', '250K', '500K', 'FULL')
K = (1, 3, 5, 7)
COLORS = ('#164e72', '#12766c', '#936112', '#783d66')
MARKERS = ('o', 's', '^', 'D')


def records(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def figure(rows=1, cols=1, width=7.2, height=3.5):
    return plt.subplots(rows, cols, figsize=(width, height), layout='constrained')


def export(out, name, fig):
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{ext}', dpi=240)
    plt.close(fig)


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def effect(row):
    f = lambda k: float(row[k])
    return [f'{f("diff"):+.2f}', f'[{f("cluster_lo"):+.2f}, {f("cluster_hi"):+.2f}]',
            f'{f("cluster_p"):.4f}', f'{f("cluster_bh"):.4f}']


def replace_tables(paper, tables):
    import re
    text = paper.read_text(encoding='utf-8')
    for label, block in tables.items():
        pattern = rf'^(<!-- begin {label} -->)\n.*?^(<!-- end {label} -->)'
        text, count = re.subn(pattern, lambda m: m[1] + '\n' + block + '\n' + m[2], text, flags=re.M | re.S)
        if count != 1:
            raise ValueError(f'Expected exactly one block for {label}, found {count}')
    paper.write_text(text, encoding='utf-8')


def bars(ax, entries, names, *, show_xlabel=True):
    for i, (row, name) in enumerate(zip(entries, names, strict=True)):
        a, b, c = (float(row[k]) for k in ('diff', 'cluster_lo', 'cluster_hi'))
        ax.errorbar(a, i, xerr=[[a-b], [c-a]], fmt='o', color=COLORS[0], capsize=3, markersize=4)
    ax.set_yticks(np.arange(len(names)), labels=names)
    ax.invert_yaxis()
    ax.axvline(0, color='#5f6b73', lw=.9)
    if show_xlabel:
        ax.set_xlabel('Paired CIDEr difference (points)')
    ax.grid(axis='x', color='#e3e9ed', linewidth=.7)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, default=HERE.parents[1] / 'reports/local_independent_review/results')
    p.add_argument('--out', type=Path, default=HERE)
    p.add_argument('--paper', type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.spines.top':False,
                         'axes.spines.right':False, 'savefig.facecolor':'white', 'pdf.fonttype':42})
    summary = json.loads((args.audit/'summary.json').read_bytes())
    top = records(args.audit/'top1_churn_downstream.csv')
    pairs = {(r['split'],r['family'],r['test']):r for r in records(args.audit/'paired_contrasts.csv') if r['metric']=='official_ptb'}
    val = summary['val497']['means']['official_ptb']
    cell = lambda s,t,k: val[f'postfix_{s}_{t}{k}']
    marks = np.arange(6)

    fig, axes = figure(1,2, height=3.35)
    for ax,t in zip(axes, ('CF','D'), strict=True):
        for i,k in enumerate(K):
            response = [cell(s,t,k)-cell('10k',t,k) for s in SCALES]
            ax.plot(marks, response, color=COLORS[i], marker=MARKERS[i], lw=1.5, label=f'k = {k}', ms=4)
        ax.axhline(0,color='#6b7479',lw=.8)
        ax.set(xticks=marks, xticklabels=LABELS, ylim=(-3.3,3.3), xlabel='External corpus scale', title=f'{t} ordering')
        ax.grid(axis='y', color='#e3e9ed')
    axes[0].set_ylabel('CIDEr change from 10K (points)')
    axes[1].legend(loc='upper left', frameon=False, ncol=2)
    export(args.out, 'fig1_scale_curves', fig)

    fig, axes = figure(1,2, height=3.35)
    for i,t in enumerate(('CF','D')):
        z={r['scale_vs_10k']:r for r in top if r['ordering']==t and r['budget']=='3'}
        axes[0].plot(marks, [0]+[float(z[s]['selected_xclip_delta']) for s in SCALES[1:]], color=COLORS[i], marker=MARKERS[i], label=t)
        rs=[pairs['val497','exploratory',f'{s}-10k/{t}/k3'] for s in SCALES[1:]]
        d, lo, hi = ([float(r[key]) for r in rs] for key in ('diff','cluster_lo','cluster_hi'))
        axes[1].errorbar(marks[1:]+(.045 if i else -.045), d, [np.subtract(d,lo),np.subtract(hi,d)],
                         color=COLORS[i], marker=MARKERS[i], capsize=2, lw=1.05, label=t)
    for ax in axes:
        ax.axhline(0, color='#6b7479', lw=.8)
        ax.set(xticks=marks, xticklabels=LABELS, xlabel='External corpus scale')
        ax.legend(frameon=False)
    axes[0].set_ylabel('Selected top-1 X-CLIP change from 10K')
    axes[1].set_ylabel('D3 or CF3 CIDEr change from 10K (points)')
    export(args.out, 'fig2_access_utility', fig)

    fig, axes = figure(1,2, height=2.9)
    for ax, split,title in zip(axes,('test2690','msvd670'),('MSR-VTT test2690','MSVD test670'),strict=True):
        keys=('D1-D3','D3-D7','D1-D7')
        bars(ax,[pairs[split,'benchmark',k] for k in keys],keys)
        ax.set(title=title, xlim=(-1,21))
    export(args.out, 'fig3_cross_benchmark', fig)

    fig, (ax,diag) = figure(1,2, width=8.4, height=4.0)
    order=(1,3,7)
    for i,t in enumerate(('D','Oracle')):
        y=[cell('full','D',k) if t=='D' else val[f'oracle{k}'] for k in order]
        ax.plot(order,y,label=t,color=COLORS[i],marker=MARKERS[i],lw=1.6)
    ax.set(xlabel='Evidence budget k', ylabel='CIDEr (points)', xticks=order, ylim=(0,80), title='(a) Observed caption utility')
    ax.legend(frameon=False,loc='lower left')
    diag.set(xlim=(0,1), ylim=(0,1), title='(b) Changes bundled in Oracle3 → Oracle7')
    diag.axis('off')
    for y,label in ((.75,'Four additional captions'),(.53,'Different set composition and\nlower lexical compatibility'),(.31,'Four additional visual inputs\nto compact fusion')):
        patch=FancyBboxPatch((.04,y),.90,.13,boxstyle='round,pad=.015',edgecolor='#4b6472',facecolor='#edf4f7')
        diag.add_patch(patch)
        diag.text(.49,y+.065,label,ha='center',va='center')
    diag.text(.49,.13,'One observed joint input contrast',ha='center',weight='bold',color=COLORS[0])
    export(args.out, 'fig4_oracle', fig)

    names={
       'text_main':'Text count main effect','visual_main':'Visual count main effect','text_x_visual':'Text × visual','source_x_budget':'Source × budget',
       'posrev-D7':'Reverse text vs D7','postop3last-D7':'Top 3 last vs D7','goodtail7-D7':'GoodTail vs D7',
       'hardwrong7-D7':'HardWrong vs D7','hardwrong7-easywrong7':'HardWrong vs EasyWrong','head_minus_tail_removal':'LOO head − tail removal',
    }
    fig, ax = figure(height=4.2)
    data=[pairs['val150_loo','loo',k] if k=='head_minus_tail_removal' else pairs['val497','mechanism',k] for k in names]
    bars(ax,data,list(names.values()))
    ax.set_xlim(-5.6,7.0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    export(args.out,'fig5_diagnostics',fig)

    if args.paper:
        num = ('Effect','95% source CI','Source p','Source BH q')
        grid_rows=[[l,*[f'{cell(s,t,k):.2f}' for t in ('CF','D') for k in K]] for s,l in zip(SCALES,LABELS,strict=True)]
        benchmark=[(disp, s) for s,disp in (('test2690','MSR-VTT'),('msvd670','MSVD'))]
        context_rows=[[name,k,f'{summary[s]["means"]["official_ptb"][k]:.2f}'] for name,s in benchmark
                      for k in ('D1','D3','D5','D7','Adaptive','B') if k in summary[s]['means']['official_ptb']]
        inference_rows=[[name,k,*effect(pairs[s,'benchmark',k])] for name,s in benchmark for k in ('D1-D3','D3-D7','D1-D7','Adaptive-D1')]
        oracle_rows=[[k,*effect(pairs['val497','oracle4',k])] for k in ('Oracle1-D1','Oracle3-D3','Oracle7-D7','Oracle7-Oracle3')]
        ctrl=[[names.get(k,k),*effect(r)] for (s,f,k),r in pairs.items() if f in ('loo','mechanism')]
        replace_tables(args.paper, {'grid':table(['Scale',*[f'{t}{k}' for t in ('CF','D') for k in K]],grid_rows),
           'test_means':table(['Benchmark','Condition','CIDEr'],context_rows),
           'test_inference':table(['Benchmark','Contrast',*num],inference_rows),
           'oracle':table(['Contrast',*num],oracle_rows),'controls':table(['Contrast',*num],ctrl)})


if __name__=='__main__':
    main()
