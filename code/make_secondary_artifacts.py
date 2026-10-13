"""Tables and vector figures from secondary_results.json, without manual data."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from make_figures import save,tidy,card,arrow,BLUE,TEAL,GOLD,RED,INK,GREY,LIGHT
ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'results/secondary_results.json').read_text())
LABELS=list(D['datasets']['nih']['discrimination']['Mean'])
COHORTS=[('nih','NIH original test'),('iu','IU original test'),('nih_replication','NIH new patients')]

def architecture():
    f,a=plt.subplots(figsize=(7.16,5.15));f.subplots_adjust(0,0,1,1);a.set(xlim=(0,100),ylim=(0,76));a.axis('off')
    a.text(2,74,'a   Executed output boundary',fontsize=10.8,fontweight='bold')
    card(a,2,49,14,18,'Request','Unit ID\n+ findings',color=GREY)
    card(a,20,49,15,18,'Planner','Calls registered\nCXR experts',color=GREY)
    card(a,39,59,19,10,'Expert A','CheXpert-trained',color=BLUE,fs=9.4)
    card(a,39,46,19,10,'Expert B','MIMIC-trained',color=TEAL,fs=9.4)
    card(a,63,49,17,18,'Final gate','Set intersection\n+ risk threshold',color=BLUE,face='#EAF2F9',fs=9.4)
    card(a,85,59,13,10,'Release','Binary finding',color=TEAL,fs=9.4)
    card(a,85,46,13,10,'Defer','No label value',color=GOLD,fs=9.4)
    for p,q in [((16.3,58),(19.5,58)),((35.2,62),(38.5,64)),((35.2,54),(38.5,51)),((58.2,64),(62.5,61)),((58.2,51),(62.5,55)),((80.2,61),(84.5,64)),((80.2,54),(84.5,51))]:arrow(a,p,q)
    a.text(48.5,43.6,'Same unit and finding; distinct expert identities',ha='center',fontsize=8.8,color=GREY)
    a.text(2,39.8,'b   Original disjoint data partitions',fontsize=9.9,fontweight='bold')
    card(a,2,24,28,12,'Conformal calibration','NIH 846 | IU 1,182\nClass-conditional sets; alpha = 0.10',color=BLUE,fs=9.4)
    card(a,34,24,31,12,'Risk calibration','NIH 634 | IU 886\nExact bounds over 308 candidate rules',color=TEAL,fs=9.4)
    card(a,69,24,29,12,'Original test','NIH 636 | IU 887\nRelease, errors and positive recall',color=GREY,fs=9.4)
    a.text(2,20,'c   New patient replication; all fitted components frozen',fontsize=9.9,fontweight='bold')
    a.add_patch(plt.Rectangle((2,6.5),96,10.5,fc=LIGHT,ec='#CFD9E1',lw=.7))
    a.text(5,13.3,'2,244 new-shard images',fontsize=8.8,fontweight='bold')
    a.text(5,9.2,'1,966 unique patient IDs',fontsize=8.8,color=GREY)
    arrow(a,(34,11.7),(40,11.7))
    a.text(43,13.3,'384 overlapping patients excluded',fontsize=8.8,fontweight='bold')
    a.text(43,9.2,'1,582 new patients; one hash-selected image each',fontsize=8.8,color=GREY)
    a.text(2,2.3,'Certificate scope: independent units, fixed pipeline, common distribution and specified annotation target.',fontsize=8.8,color=GREY)
    save(f,'fig1_architecture')

def discrimination():
    f,axes=plt.subplots(1,3,figsize=(7.16,3.9));f.subplots_adjust(left=.165,right=.99,top=.80,bottom=.25,wspace=.15)
    yy=np.arange(7)[::-1];colors=[BLUE,TEAL,GOLD]
    for k,((ds,title),a) in enumerate(zip(COHORTS,axes)):
        for ei,(name,c) in enumerate(zip(['CheXpert','MIMIC','Mean'],colors)):
            for y,l in zip(yy,LABELS):
                r=D['datasets'][ds]['discrimination'][name][l];v=r['auroc'];lo,hi=r['auroc_ci']
                a.errorbar(v,y+(ei-1)*.18,xerr=[[v-lo],[hi-v]],fmt=['o','s','D'][ei],ms=2.6,lw=.65,color=c,capsize=1)
        a.axvline(.5,color=GREY,lw=.7,ls='--');a.set(xlim=(0,1.03),ylim=(-.6,6.6),yticks=yy,xticks=[0,.5,1],xlabel='AUROC')
        a.set_yticklabels(LABELS if k==0 else []);a.tick_params(axis='y',length=0);tidy(a)
        a.set_title(chr(97+k)+'   '+title,loc='left',fontsize=8.8,fontweight='bold',pad=14)
    f.legend([Line2D([],[],color=c,marker=m,lw=.8,ms=3) for c,m in zip(colors,['o','s','D'])],['CheXpert-trained','MIMIC-trained','Mean score'],loc='upper center',bbox_to_anchor=(.56,.995),frameon=False,ncol=3,fontsize=8.8)
    f.text(.165,.035,'Pointwise 95% unit-bootstrap intervals; sparse positive targets widen uncertainty.',fontsize=8.8,color=GREY)
    save(f,'fig7_discrimination')

def matched():
    f,axes=plt.subplots(1,2,figsize=(7.16,3.6));f.subplots_adjust(left=.24,right=.98,top=.76,bottom=.25,wspace=.28)
    methods=[('Mean / matched release','Mean at 0.5'),('CC-Arbiter','CC-Arbiter'),('Youden / matched release','Calibration Youden')]
    coords=[5,4,3,1.5,.5,-.5];yt=[]
    for g,ds in enumerate(['nih','nih_replication']):
        pol=D['datasets'][ds]['policies']
        for j,(key,label) in enumerate(methods):
            y=coords[g*3+j];r=pol[key];color=[GREY,BLUE,GOLD][j]
            for a,metric in zip(axes,['selective_risk','positive_recall_released']):
                v=r[metric]*100;lo,hi=np.array(r['ci'][metric])*100
                a.errorbar(v,y,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',ms=4,color=color,lw=1,capsize=2)
                suffix=f"{r['errors']}/{r['accepted']}" if metric=='selective_risk' else f"{r['true_positive_released']}/{r['positive_targets']}"
                a.text(max(hi,v)+.5,y,suffix,va='center',fontsize=8.8,color=color)
            yt.append(label)
    for a in axes:a.set(yticks=coords,ylim=(-1.1,6.3));tidy(a)
    axes[0].set_yticklabels(yt,fontsize=8.8);axes[1].set_yticklabels([]);axes[1].tick_params(axis='y',length=0)
    axes[0].set(xlim=(-.2,20),xlabel='Error among released labels (%)')
    axes[1].set(xlim=(-.2,12),xlabel='Positive recall after deferral (%)')
    axes[0].set_title('a   Error at the same release budget',loc='left',fontweight='bold',fontsize=8.8,pad=15)
    axes[1].set_title('b   Positive annotations recovered',loc='left',fontweight='bold',fontsize=8.8,pad=15)
    for a in axes:
        a.text(0,5.85,'Original NIH: 407 releases',fontsize=8.8,fontweight='bold')
        a.text(0,2.35,'New NIH: 1,195 releases',fontsize=8.8,fontweight='bold')
        a.axhline(2.05,lw=.6,color='#CCD6DF')
    f.text(.24,.95,'Matched per-label release counts; selection uses scores and unit IDs only',fontsize=8.8,fontweight='bold')
    f.text(.24,.035,'Retrospective batch comparison; intervals condition on fitted thresholds and masks.',fontsize=8.8,color=GREY)
    save(f,'fig8_matched_replication')

def pct(v):return '--' if v is None else f'{v*100:.1f}'
def interval(v,ci):return '--' if v is None else f'{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]'
def write(name,lines):(ROOT/'paper'/name).write_text('\n'.join(lines)+'\n',encoding='utf8')

def tables():
    methods=[('Always negative','Always negative'),('Mean score','Mean at 0.5'),('Mean / calibration Youden','Mean: calibration Youden'),('Mean / calibration minimum error','Mean: calibration min. error'),('Risk gate (mean)','Mean risk gate'),('CC-Arbiter','CC-Arbiter'),('Mean / matched release','Mean: matched release'),('Youden / matched release','Youden: matched release')]
    a=[r'\begin{tabular}{lrrrrr}\toprule',r'Policy & Released & Errors & Error (\%) & TP & Recall (\%)\\\midrule']
    for key,label in methods:
        r=D['datasets']['nih_replication']['policies'][key]
        a.append(f"{label} & {r['accepted']:,} & {r['errors']:,} & {pct(r['selective_risk'])} & {r['true_positive_released']} & {pct(r['positive_recall_released'])} "+r'\\')
    write('table_replication.tex',a+[r'\bottomrule\end{tabular}'])
    a=[]
    for ds,title in COHORTS:
        a.extend([r'\subsection*{'+title+'}',r'\begin{center}\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{llrrr}\toprule',r'Finding & Score & Positive $n$ & AUROC [95\% CI] & AP [95\% CI]\\\midrule'])
        for label in LABELS:
            for ei,expert in enumerate(['CheXpert','MIMIC','Mean']):
                r=D['datasets'][ds]['discrimination'][expert][label]
                a.append(f"{label if ei==0 else ''} & {expert} & {r['positive'] if ei==0 else ''} & {interval(r['auroc'],r['auroc_ci'])} & {interval(r['ap'],r['ap_ci'])} "+r'\\')
        a.extend([r'\bottomrule\end{tabular}\end{center}',r'\noindent '+f"Denominator: {D['datasets'][ds]['units']:,} units. AP is average precision. Its prevalence reference and valid bootstrap counts are available in JSON. Intervals are pointwise."])
        if ds!='iu':a.append(r'\clearpage')
    write('supp_discrimination.tex',a)
    fitted=json.loads((ROOT/'results/secondary_thresholds.json').read_text())
    a=[r'\begin{center}\small\setlength{\tabcolsep}{5pt}\begin{tabular}{llrrrr}\toprule',r'Dataset & Finding & $n_+$ & $n_-$ & Youden & Min. error\\\midrule']
    for ds in ['nih','iu']:
        for r in fitted[ds]:a.append(f"{ds.upper()} & {r['label']} & {r['positive_calibration']} & {r['negative_calibration']} & {r['youden']:.6f} & {r['minimum_error']:.6f} "+r'\\')
    write('supp_thresholds.tex',a+[r'\bottomrule\end{tabular}\end{center}'])
    a=[r'\begin{center}\footnotesize\setlength{\tabcolsep}{4pt}\begin{tabular}{llrrrr}\toprule',r'Cohort & Policy & Released & Errors & TP & Neg. recovery (\%)\\\midrule']
    for ds,title in COHORTS:
        for key,label in methods:
            r=D['datasets'][ds]['policies'][key]
            a.append(f"{('NIH new' if ds=='nih_replication' else ds.upper())} & {label} & {r['accepted']:,} & {r['errors']:,} & {r['true_positive_released']} & {pct(r['specificity_after_deferral'])} "+r'\\')
        a.append(r'\addlinespace')
    write('supp_secondary_policies.tex',a+[r'\bottomrule\end{tabular}\end{center}'])
    a=[r'\begin{center}\small\begin{tabular}{llrr}\toprule',r'Dataset & Finding & Maximum accepted $n$ & Minimum upper bound\\\midrule']
    for ds in ['nih','iu']:
        for label,r in D['certification_feasibility']['0.1']['per_dataset'][ds].items():
            a.append(f"{ds.upper()} & {label} & {r['max_candidate_n']} & {r['minimum_candidate_bound']:.3f} "+r'\\')
    write('supp_feasibility.tex',a+[r'\bottomrule\end{tabular}\end{center}'])
    with (ROOT/'results/secondary_policy_summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['cohort','policy','units','opportunities','accepted','errors','conditional_error','TP','positive_recall','specificity_after_deferral'])
        for ds,r in D['datasets'].items():
            for key,m in r['policies'].items():w.writerow([ds,key,m['units'],m['opportunities'],m['accepted'],m['errors'],m['selective_risk'],m['true_positive_released'],m['positive_recall_released'],m['specificity_after_deferral']])

if __name__=='__main__':architecture();discrimination();matched();tables()
