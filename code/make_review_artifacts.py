"""Transparent positive attrition and failed routing feasibility, from saved outputs."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from make_figures import save,tidy,BLUE,TEAL,GOLD,RED,GREY,INK
ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'results/review_results.json').read_text())
A={}
primary=json.loads((ROOT/'results/results.json').read_text())
for ds in ['nih','iu']:
    pp=primary['datasets'][ds]['targets']['0.1']['policies']
    A[ds]={'positive_annotations':pp['CC-Arbiter']['positive_targets'],
           'mean_true_positive':pp['Mean score']['true_positive_released'],
           'intersection_true_positive':pp['CP intersection']['true_positive_released'],
           'intersection_positive_proposals':sum(x['accepted_predicted_positive'] for x in pp['CP intersection']['per_label'].values()),
           'gated_true_positive':pp['CC-Arbiter']['true_positive_released'],
           'gated_positive_releases':sum(x['accepted_predicted_positive'] for x in pp['CC-Arbiter']['per_label'].values())}
(ROOT/'results/positive_attrition.json').write_text(json.dumps(A,indent=2),encoding='utf8')

def main():
    fig,axes=plt.subplots(1,2,figsize=(7.16,3.9),gridspec_kw={'width_ratios':[1.05,1]})
    fig.subplots_adjust(left=.2,right=.985,top=.80,bottom=.27,wspace=.5)
    a,b=axes;yy=np.arange(3)[::-1];keys=['mean_true_positive','intersection_true_positive','gated_true_positive']
    for ds,offset,c in [('nih',.16,BLUE),('iu',-.16,TEAL)]:
        den=A[ds]['positive_annotations'];vv=[A[ds][k] for k in keys]
        a.barh(yy+offset,np.array(vv)/den*100,height=.28,color=c,label=ds.upper())
        for y,v in zip(yy+offset,vv):a.text(v/den*100+2,y,f'{v}/{den}',va='center',fontsize=8.8,color=c)
    a.set(yticks=yy,yticklabels=['Mean at 0.5','CP intersection','CC-Arbiter'],xlim=(0,120),xticks=[0,50,100],xlabel='Positive annotations correctly released (%)')
    a.set_title('a   Automatic positive release',loc='left',fontweight='bold',fontsize=8.8,pad=30);tidy(a)
    a.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=2,fontsize=8.8,frameon=False)
    b.add_patch(Rectangle((.025,.025),.075,.075,facecolor='#E5F3ED',edgecolor='none'))
    for method,c,marker,label in [('stacker',BLUE,'o','Logistic stacker'),('max_mean',GOLD,'s','Maximum mean')]:
        cc=D['calibration'][method]['candidates'];b.scatter([x['miss_fraction_upper'] for x in cc],[x['unflagged_positive_upper'] for x in cc],s=11,facecolors='none',edgecolors=c,marker=marker,lw=.8,label=label)
    b.axvline(.1,color=RED,lw=.8,ls='--');b.axhline(.1,color=RED,lw=.8,ls='--')
    b.set(xscale='log',yscale='log',xlim=(.025,1.15),ylim=(.025,1.15),xlabel='Missed / positive: upper bound',ylabel='Positive / unflagged: upper bound')
    for axis in [b.xaxis,b.yaxis]:axis.set_ticks([.05,.1,.2,.5,1]);axis.set_ticklabels(['5%','10%','20%','50%','100%'])
    b.tick_params(which='minor',bottom=False,left=False);b.text(.033,.044,'Required\nregion:\nnone qualify',fontsize=8.8,color=TEAL)
    b.set_title('b   Review-routing constraints',loc='left',fontweight='bold',fontsize=8.8,pad=34)
    b.legend(loc='upper center',bbox_to_anchor=(.5,1.22),ncol=1,fontsize=8.8,frameon=False);tidy(b)
    fig.text(.2,.085,'a: seven finding targets. b: any of seven annotations per patient; 1,278 calibration patients.',fontsize=8.8,color=GREY)
    fig.text(.2,.035,'No threshold passed both bounds; both rules flagged all 1,279 test patients.',fontsize=8.8,color=GREY)
    save(fig,'fig9_positive_review')
    lines=[r'\begin{tabular}{lrrrrr}\toprule',r'Routing rule & Flagged & Positive flagged & Sensitivity (\%) & Unflagged & Positive/unflagged\\\midrule']
    names={'stacker':'Constrained stacker','max_mean':'Constrained maximum mean','stacker_fixed_0.5':'Stacker at 0.5','max_mean_fixed_0.5':'Maximum mean at 0.5','always_flag':'Always flag'}
    for method,label in names.items():
        r=D['policies'][method];risk=f"{r['unflagged_positive']}/{r['unflagged']}" if r['unflagged'] else '--'
        lines.append(f"{label} & {r['flagged']:,} & {r['flagged_positive']}/{r['positive_patients']} & {100*r['sensitivity']:.1f} & {r['unflagged']:,} & {risk} "+r'\\')
    (ROOT/'paper/table_review.tex').write_text('\n'.join(lines+[r'\bottomrule\end{tabular}']),encoding='utf8')
    with (ROOT/'results/review_candidate_bounds.csv').open('w',newline='') as f:
        keys=list(D['calibration']['stacker']['candidates'][0]);w=csv.DictWriter(f,fieldnames=['method']+keys);w.writeheader()
        for method,r in D['calibration'].items():
            for row in r['candidates']:w.writerow({'method':method,**row})

if __name__=='__main__':main()
