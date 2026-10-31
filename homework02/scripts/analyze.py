#!/usr/bin/env python3
"""Summarize actual benchmark samples and render report inputs; never invent data."""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def tex(value):
    replacements = {'\\':r'\textbackslash{}','_':r'\_','%':r'\%','&':r'\&',
                    '#':r'\#','$':r'\$','{':r'\{','}':r'\}'}
    return ''.join(replacements.get(ch,ch) for ch in str(value))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--report-dir',type=Path,default=Path(__file__).resolve().parents[1]/'report')
    args=parser.parse_args()
    metadata=json.loads((args.dataset/'metadata.json').read_text(encoding='utf-8'))
    environment=(args.dataset/'environment.txt').read_text(encoding='utf-8')
    allocation=re.search(r'job=\S+ cpus=(\d+)',environment)
    if allocation:
        metadata['allocated_cpus']=int(allocation.group(1))
        metadata['status_note']+=' Allocation: '+allocation.group(1)+' CPUs and 4 GiB RAM.'
    samples=defaultdict(list)
    checksums=defaultdict(set)
    repetitions=defaultdict(set)
    with (args.dataset/'raw.csv').open(newline='',encoding='utf-8') as file:
        for row in csv.DictReader(file):
            algorithm=row['algorithm']; n=int(row['n']); threads=int(row['threads'])
            seconds=float(row['seconds']); repeat=int(row['repeat'])
            if algorithm not in ('serial','doubling','tree') or seconds<=0:
                raise ValueError('Unexpected algorithm or nonpositive elapsed time')
            key=(algorithm,n,threads)
            if repeat in repetitions[key]: raise ValueError('Duplicate trial ID')
            repetitions[key].add(repeat); samples[key].append(seconds)
            checksums[n].add(row['checksum'])
    if not samples: raise ValueError('No benchmark samples')
    if any(len(values)!=1 for values in checksums.values()): raise ValueError('Checksum mismatch')
    stats=[]
    for (algorithm,n,p),values in sorted(samples.items()):
        if len(values)!=metadata['repeats']: raise ValueError('Incomplete benchmark configuration')
        q1,median,q3=np.quantile(values,[.25,.5,.75])
        stats.append(dict(algorithm=algorithm,n=n,threads=p,repeats=len(values),
                          median_seconds=float(median),q1_seconds=float(q1),q3_seconds=float(q3)))
    bykey={(s['algorithm'],s['n'],s['threads']):s for s in stats}
    sizes=sorted({s['n'] for s in stats})
    if sizes!=sorted(metadata['sizes']): raise ValueError('Missing input size')
    for n in sizes:
        serial=bykey['serial',n,1]['median_seconds']
        for s in stats:
            if s['n']==n:
                s['speedup']=serial/s['median_seconds']
                s['efficiency']=s['speedup']/s['threads']
    for algorithm in ('doubling','tree'):
        for n in sizes:
            found={s['threads'] for s in stats if s['algorithm']==algorithm and s['n']==n}
            if found!=set(metadata['threads']): raise ValueError('Missing thread configuration')
    output=args.dataset/'summary.csv'
    with output.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(stats[0])); writer.writeheader(); writer.writerows(stats)
    (args.dataset/'summary.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
    report=args.report_dir; report.mkdir(parents=True,exist_ok=True)
    n=max(sizes); serial=bykey['serial',n,1]['median_seconds']
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.bbox':'tight'})
    fig,ax=plt.subplots(figsize=(3.3,2.05),layout='constrained')
    best={}
    for algorithm,label,color,marker in [('doubling',r'$O(n\log n)$ doubling','#b95028','o'),
                                          ('tree',r'$O(n)$ tree','#146b75','s')]:
        rows=sorted((s for s in stats if s['algorithm']==algorithm and s['n']==n),key=lambda s:s['threads'])
        xs=np.array([s['threads'] for s in rows]); ys=np.array([s['speedup'] for s in rows])
        low=np.array([serial/s['q3_seconds'] for s in rows]); high=np.array([serial/s['q1_seconds'] for s in rows])
        ax.plot(xs,ys,label=label,color=color,marker=marker,markersize=4,linewidth=1.3)
        ax.fill_between(xs,low,high,color=color,alpha=.14)
        best[algorithm]=max(rows,key=lambda s:s['speedup'])
    ax.axhline(1,color='#777777',linestyle='--',linewidth=.8,label='Serial baseline')
    ax.set_xscale('log',base=2); ax.set_xticks(metadata['threads'],labels=metadata['threads'])
    ax.set_xlabel('OpenMP threads'); ax.set_ylabel(r'Speedup $T_{serial}/T_p$')
    ax.set_title('n = '+f'{n:,}'+': '+metadata['platform_short'],fontsize=8)
    ax.grid(axis='y',alpha=.2); ax.legend(fontsize=6.8,loc='best',frameon=False)
    ax.set_ylim(bottom=0)
    fig.savefig(report/'scaling.pdf'); fig.savefig(report/'scaling.png',dpi=220)
    plt.close(fig)
    platform_display=metadata['platform'].replace('.talapas.uoregon.edu','')
    lines=[r'\newcommand{\ExperimentPlatform}{'+tex(platform_display)+'}',
           r'\newcommand{\ExperimentCPU}{'+tex(metadata['cpu'])+'}',
           r'\newcommand{\ExperimentCompiler}{'+tex(metadata['compiler'])+'}',
           r'\newcommand{\RunStatus}{'+tex(metadata['status_note'])+'}',
           r'\newcommand{\MaxSize}{'+f'{n:,}'+'}',
           r'\newcommand{\TrialCount}{'+str(metadata['repeats'])+'}',
           r'\newcommand{\ConfigurationCount}{'+str(len(stats))+'}',
           r'\newcommand{\MeasurementCount}{'+str(sum(map(len,samples.values())))+'}',
           r'\newcommand{\ThreadCounts}{'+', '.join(map(str,metadata['threads']))+'}',
           r'\newcommand{\SerialTime}{'+f'{serial*1000:.2f}'+'}',
           r'\newcommand{\TreeTime}{'+f"{best['tree']['median_seconds']*1000:.2f}"+'}',
           r'\newcommand{\TreeSpeedup}{'+f"{best['tree']['speedup']:.2f}"+'}',
           r'\newcommand{\TreeThreads}{'+str(best['tree']['threads'])+'}',
           r'\newcommand{\DoublingTime}{'+f"{best['doubling']['median_seconds']*1000:.2f}"+'}',
           r'\newcommand{\DoublingSpeedup}{'+f"{best['doubling']['speedup']:.2f}"+'}',
           r'\newcommand{\DoublingThreads}{'+str(best['doubling']['threads'])+'}']
    size_text=[]
    for size in sizes:
        if size>0 and size&(size-1)==0:
            size_text.append('2^{'+str(size.bit_length()-1)+'}')
        else:
            size_text.append(str(size))
    lines.append(r'\newcommand{\ProblemSizes}{'+', '.join(size_text)+'}')
    largest_p=max(metadata['threads'])
    lines.append(r'\newcommand{\HighThreadFinding}{At '+str(largest_p)+' threads, tree and doubling speedups were '+
                 f"{bykey['tree',n,largest_p]['speedup']:.2f}"+' and '+
                 f"{bykey['doubling',n,largest_p]['speedup']:.2f}"+', respectively.}')
    parallel=[s for s in stats if s['algorithm']!='serial']
    finding=('Neither parallel method\'s median time beat serial at any tested input size. '
             if max(s['speedup'] for s in parallel)<=1 else
             'Parallel speedup depends on input size and the chosen thread count. ')
    last_tree=bykey['tree',n,largest_p]
    if last_tree['median_seconds']>best['tree']['median_seconds']*1.05:
        percent=100*(last_tree['median_seconds']/best['tree']['median_seconds']-1)
        finding+=f"Increasing tree threads from {best['tree']['threads']} to {largest_p} raised median runtime by {percent:.0f}%. "
    finding+='Lower total work alone did not guarantee proportional thread scaling.'
    lines.append(r'\newcommand{\FindingSummary}{'+tex(finding)+'}')
    if metadata['talapas_verified']:
        lines.append(r'\newcommand{\MachineLimit}{The node has 128 physical cores across two sockets and eight NUMA domains. The shared first-touch policy and core binding can affect serial/parallel ratios; this experiment does not isolate placement effects or characterize other nodes.}')
    else:
        lines.append(r'\newcommand{\MachineLimit}{WSL exposes 16 physical cores and 32 logical CPUs; the 32-thread point uses SMT. Desktop activity, virtualization, and cache effects limit generalization to Talapas.}')
    (report/'numbers.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    rows=[]
    for size in sizes:
        winners=[]
        for algorithm in ('doubling','tree'):
            winners.append(max((s for s in stats if s['algorithm']==algorithm and s['n']==size),key=lambda s:s['speedup']))
        rows.append(f"{size:,} & {winners[0]['speedup']:.2f} ({winners[0]['threads']}) & {winners[1]['speedup']:.2f} ({winners[1]['threads']}) "+r'\\')
    table=(r'\begin{tabular}{rcc}'+'\n'+r'\toprule'+'\n'+
           r'$n$ & $O(n\log n)$ doubling & $O(n)$ tree \\'+'\n'+r'\midrule'+'\n'+
           '\n'.join(rows)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n')
    (report/'table.tex').write_text(table,encoding='utf-8')
    (report/'dataset.json').write_text(json.dumps({'source_csv':str((args.dataset/'raw.csv').resolve()),
        'metadata':metadata,'configurations':len(stats),'samples':sum(map(len,samples.values()))},indent=2)+'\n',encoding='utf-8')
    print(f'Summarized {sum(map(len,samples.values()))} samples across {len(stats)} configurations.')
    print('Dataset platform:',metadata['platform'])


if __name__=='__main__': main()
