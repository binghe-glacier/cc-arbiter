"""Publication vector figures; all statistical panels read final results.json."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'paper/figures'
D=json.loads((ROOT/'results/results.json').read_text()); LABELS=D['labels']
BLUE='#28639A'; TEAL='#168276'; GOLD='#B67A23'; RED='#B34C52'; INK='#233746'; GREY='#687887'; LIGHT='#EEF3F7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.8,'axes.labelsize':8.8,'axes.titlesize':9.9,'xtick.labelsize':8.8,'ytick.labelsize':8.8,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9CAAB5','axes.linewidth':.65,'text.color':INK,'axes.labelcolor':INK,
 'xtick.color':GREY,'ytick.color':GREY,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
def save(fig,name):
    for ext in ['pdf','svg','png']: fig.savefig(OUT/f'{name}.{ext}',dpi=400,facecolor='white')
    plt.close(fig)
def card(ax,x,y,w,h,title,body='',color=BLUE,face='white',fs=9.4):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1,rounding_size=1',ec=color,fc=face,lw=.9))
    ax.text(x+w/2,y+h-2,title,ha='center',va='top',fontsize=fs,fontweight='bold',color=color)
    if body: ax.text(x+w/2,y+h/2-1.2,body,ha='center',va='center',fontsize=fs-.4,linespacing=1.4)
def arrow(ax,a,b,color=GREY,ls='-'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8.5,lw=.9,color=color,linestyle=ls,shrinkA=0,shrinkB=0))
def tidy(ax,axis='x'):
    ax.grid(axis=axis,color='#E5EBF0',lw=.6); ax.set_axisbelow(True); ax.tick_params(length=2)
def arch():
    f,a=plt.subplots(figsize=(7.16,4.45)); f.subplots_adjust(0,0,1,1);a.set(xlim=(0,100),ylim=(0,64));a.axis('off')
    a.text(2,62,'a   Enforced execution path',fontsize=10.8,fontweight='bold')
    card(a,2,37,14,18,'Request','Unit ID\n+ findings',color=GREY)
    card(a,20,37,15,18,'Planner','Calls registered\nCXR experts',color=GREY)
    card(a,39,47,19,10,'Expert A','CheXpert-trained',color=BLUE,fs=9.4)
    card(a,39,34,19,10,'Expert B','MIMIC-trained',color=TEAL,fs=9.4)
    card(a,63,37,17,18,'Evidence gate','Set intersection\n+ certified threshold',color=BLUE,face='#EAF2F9',fs=9.4)
    card(a,85,47,13,10,'Release','Binary finding',color=TEAL,fs=9.4)
    card(a,85,34,13,10,'Defer','No label released',color=GOLD,fs=9.4)
    arrow(a,(16.3,46),(19.5,46)); arrow(a,(35.2,50),(38.5,52));arrow(a,(35.2,42),(38.5,39))
    arrow(a,(58.2,52),(62.5,49),BLUE);arrow(a,(58.2,39),(62.5,43),TEAL)
    arrow(a,(80.2,49),(84.5,52),TEAL);arrow(a,(80.2,42),(84.5,39),GOLD)
    a.text(48.5,31.5,'Same unit, same finding; two distinct experts',ha='center',fontsize=8.8,color=GREY)
    a.text(88,30.5,'Final structured output',ha='center',fontsize=8.8,color=GREY)
    a.text(2,26.5,'b   Independent calibration and evaluation',fontsize=9.9,fontweight='bold')
    card(a,2,9,28,13,'Conformal split (40%)','NIH 846 patients | IU 1,182 reports\nClass-conditional sets; alpha = 0.10',color=BLUE,fs=9.4)
    card(a,34,9,31,13,'Risk split (30%)','NIH 634 patients | IU 886 reports\nExact upper bounds; fixed threshold grid',color=TEAL,fs=9.4)
    card(a,69,9,29,13,'Locked test (30%)','NIH 636 patients | IU 887 reports\nRelease, errors and positive retention',color=GREY,fs=9.4)
    a.text(2,5.5,'Finite-sample risk bound: i.i.d. units, fixed pipeline and specified annotation target.',fontsize=8.8,fontweight='bold')
    a.text(2,2.3,'The final gate replaces planner prose. Unsupported, missing or uncertified evidence yields deferral.',fontsize=8.8,color=GREY)
    save(f,'fig1_architecture')
def coverage():
    f,(a,b)=plt.subplots(1,2,figsize=(7.16,2.65),gridspec_kw={'width_ratios':[1.05,1]}); f.subplots_adjust(left=.17,right=.97,bottom=.2,top=.79,wspace=.43)
    yy=np.arange(4)[::-1]; names=[]; cov=[]; one=[]; broad=[]
    for ds,label in [('nih','NIH'),('iu','IU')]:
        for expert,tag in [('CheXpert expert','CheXpert'),('MIMIC expert','MIMIC')]:
            q=D['datasets'][ds]['coverage'][expert]; names.append(label+' / '+tag);cov.append(q['coverage']);one.append(q['singleton']);broad.append(1-q['singleton']-q['empty'])
    a.axvline(.9,color=GOLD,ls='--',lw=1)
    for i,y in enumerate(yy):
        a.plot(cov[i],y,'o',color=BLUE if i%2==0 else TEAL,ms=5);a.text(cov[i]+.0008,y,f'{100*cov[i]:.1f}%',va='center',fontsize=8.8)
    a.set(yticks=yy,yticklabels=names,xlim=(.88,.917),ylim=(-.5,3.5),xlabel='Pooled empirical set coverage')
    a.set_xticks([.88,.90,.915],['88%','90%','91.5%']);tidy(a)
    a.set_title('a   Coverage against annotation targets',loc='left',pad=16,fontweight='bold',fontsize=8.8)
    b.barh(yy,one,color=TEAL,height=.5,label='Singleton');b.barh(yy,broad,left=one,color='#CDD8E1',height=.5,label='Both labels')
    for y,v in zip(yy,one): b.text(v+.015,y,f'{100*v:.1f}%',va='center',fontsize=8.8,color=INK)
    b.set(xlim=(0,1),yticks=[],ylim=(-.5,3.5),xlabel='Fraction of expert prediction sets')
    b.set_xticks([0,.5,1],['0%','50%','100%']); b.spines['left'].set_visible(False)
    b.set_title('b   Informative versus broad sets',loc='left',pad=16,fontweight='bold',fontsize=8.8)
    b.legend(loc='upper center',bbox_to_anchor=(.5,1.14),ncol=2,frameon=False,fontsize=8.8)
    f.text(.17,.03,'NIH: 4,452 and IU: 6,209 label opportunities per expert. Pooled summaries are descriptive.',fontsize=8.8,color=GREY)
    save(f,'fig2_coverage')
def main_results():
    methods=['Always negative','CheXpert expert','MIMIC expert','Mean score','Expert agreement','CP intersection','Risk gate (mean)','CC-Arbiter']
    f,axes=plt.subplots(1,4,figsize=(7.16,3.65),gridspec_kw={'width_ratios':[1,1,1,1]});f.subplots_adjust(left=.2,right=.98,bottom=.24,top=.75,wspace=.18)
    y=np.arange(len(methods))[::-1]
    for i,ds in enumerate(['nih','iu']):
        pol=D['datasets'][ds]['targets']['0.1']['policies'];a,b=axes[2*i:2*i+2]
        for yy,name in zip(y,methods):
            r=pol[name];c=BLUE if name=='CC-Arbiter' else TEAL if name=='Risk gate (mean)' else '#AABAC8'
            a.barh(yy,r['release']*100,color=c,height=.5)
            if r['selective_risk'] is None:
                b.text(5,yy,'Undefined',fontsize=8.8,color=BLUE,va='center')
            else:
                lo,hi=r['ci']['selective_risk'];v=r['selective_risk'];b.errorbar(v*100,yy,xerr=[[max(0,v-lo)*100],[max(0,hi-v)*100]],fmt='o',color=c,ms=3,lw=.8,capsize=1.5)
        for ax in [a,b]:ax.set(xlim=(0,105),ylim=(-.6,7.6),yticks=y,xticks=[0,50,100]);tidy(ax)
        a.set_yticklabels(methods if i==0 else []);b.set_yticklabels([])
        for ax in [b] if i==0 else [a,b]: ax.tick_params(axis='y',length=0)
        b.axvline(10,color=GOLD,ls='--',lw=.75)
        a.set_xlabel('Release (%)');b.set_xlabel('Error / release (%)')
        a.set_title(('a   NIH' if i==0 else 'b   IU'),loc='left',fontweight='bold',pad=14)
    f.text(.2,.95,'Primary risk target: 10%',fontsize=8.8,fontweight='bold')
    f.text(.2,.12,'Bars: released fraction. Points: observed error among released labels with unit-bootstrap 95% CIs.',fontsize=8.8,color=GREY)
    f.text(.2,.065,'Dashed line: calibration target for gated methods. An empty release set has no defined conditional error.',fontsize=8.8,color=GREY)
    save(f,'fig3_selective_performance')
def certificates():
    f,(a,b)=plt.subplots(1,2,figsize=(7.16,3.2),gridspec_kw={'width_ratios':[1.1,1]});f.subplots_adjust(left=.19,right=.97,bottom=.24,top=.76,wspace=.4)
    cert=D['datasets']['nih']['targets']['0.1']['certificates']['intersection']; met=D['datasets']['nih']['targets']['0.1']['policies']['CC-Arbiter']['per_label']
    yy=np.arange(7)[::-1]
    for y,name in zip(yy,LABELS):
        c=cert[name]
        if c['bound'] is None: a.text(.005,y,'No certified threshold',fontsize=8.8,color=GREY,va='center')
        else:
            a.plot(c['bound'],y,'s',color=TEAL,ms=4);a.plot(met[name]['selective_risk'],y,'o',color=BLUE,ms=4)
    a.axvline(.1,color=GOLD,ls='--',lw=.9);a.set(yticks=yy,yticklabels=LABELS,xlim=(0,.12),ylim=(-.7,6.7),xlabel='Conditional error / upper bound')
    a.set_xticks([0,.05,.1],['0%','5%','10%']);tidy(a)
    a.set_title('a   NIH label-specific certification',loc='left',fontweight='bold',pad=19,fontsize=8.8)
    f.legend([Line2D([],[],marker='s',color=TEAL,ls=''),Line2D([],[],marker='o',color=BLUE,ls='')],['Risk-calibration upper bound','Test error among releases'],loc='upper center',bbox_to_anchor=(.56,.995),frameon=False,fontsize=8.8,ncol=2)
    for ds,color,marker in [('nih',BLUE,'o'),('iu',TEAL,'s')]:
        for mode,ls in [('CC-Arbiter','-'),('Risk gate (mean)','--')]:
            xx=[5,10,20]; rr=[D['datasets'][ds]['targets'][str(t)]['policies'][mode]['release']*100 for t in [.05,.1,.2]]
            b.plot(xx,rr,marker=marker,ls=ls,color=color,lw=1.1,ms=4,label=('NIH' if ds=='nih' else 'IU')+' / '+('CC' if mode=='CC-Arbiter' else 'mean gate'))
    b.set(xlim=(3,22),ylim=(-1,65),xticks=[5,10,20],xlabel='Risk target (%)',ylabel='Released label opportunities (%)');tidy(b,axis='y')
    b.set_title('b   Certification versus release',loc='left',fontweight='bold',pad=19,fontsize=8.8)
    b.legend(loc='upper left',frameon=False,fontsize=8.8)
    f.text(.19,.07,'Exact one-sided binomial bounds, corrected over 308 dataset-policy-label-threshold combinations.',fontsize=8.8,color=GREY)
    save(f,'fig4_risk_certificates')
def stress():
    f,axes=plt.subplots(2,2,figsize=(7.16,4.5));f.subplots_adjust(left=.09,right=.98,bottom=.18,top=.83,wspace=.25,hspace=.38)
    colors={'Mean score':GREY,'Risk gate (mean)':TEAL,'CC-Arbiter':BLUE}; clean=D['datasets']['nih']['targets']['0.1']['policies'];st=D['datasets']['nih']['stress']
    for col,kind in enumerate(['single_expert','correlated_experts']):
        rows=[x for x in st if x['mode']==kind]
        for method,c in colors.items():
            xx=[0]+[x['requested_rate']*100 for x in rows]
            risk=[clean[method]['selective_risk']]+[x['policies'][method]['selective_risk'] for x in rows]
            rel=[clean[method]['release']]+[x['policies'][method]['release'] for x in rows]
            axes[0,col].plot(xx,np.array(risk)*100,'o-',color=c,label=method,ms=3,lw=1.2)
            axes[1,col].plot(xx,np.array(rel)*100,'o-',color=c,ms=3,lw=1.2)
        axes[0,col].axhline(10,color=GOLD,ls='--',lw=.8)
        axes[0,col].set_title(('a   One expert flipped' if col==0 else 'b   Both experts flipped together'),loc='left',fontweight='bold',fontsize=9.9)
        axes[0,col].set(ylim=(0,80),yticks=[0,20,40,60,80]);axes[1,col].set(ylim=(0,105),yticks=[0,25,50,75,100],xlabel='Requested fraction of corrupted test units (%)')
        for a in axes[:,col]:a.set(xlim=(-1,51),xticks=[0,10,25,50]);tidy(a,axis='y')
    axes[0,0].set_ylabel('Error among releases (%)');axes[1,0].set_ylabel('Released opportunities (%)')
    f.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,.975),frameon=False,ncol=3,fontsize=8.8)
    f.text(.09,.075,'NIH test set, all 636 patients; clean calibration remains frozen. Flips use no labels or correctness filtering.',fontsize=8.8,color=GREY)
    f.text(.09,.03,'The clean-data certificate does not cover corrupted scores. IU CC-Arbiter remains all-defer at the 10% target.',fontsize=8.8,color=GREY)
    save(f,'fig6_correlated_failures')
def positive():
    f,(a,b)=plt.subplots(1,2,figsize=(7.16,3.5),sharey=True);f.subplots_adjust(left=.2,right=.97,bottom=.21,top=.8,wspace=.2)
    methods=['Always negative','CheXpert expert','MIMIC expert','Mean score','Expert agreement','CP intersection','Risk gate (mean)','CC-Arbiter']; yy=np.arange(8)[::-1]
    for ax,ds,label in [(a,'nih','a   NIH: 232 positive targets'),(b,'iu','b   IU: 193 positive targets')]:
        for y,name in zip(yy,methods):
            r=D['datasets'][ds]['targets']['0.1']['policies'][name];c=BLUE if name=='CC-Arbiter' else TEAL if name=='Risk gate (mean)' else '#AABAC8'
            v=r['positive_recall_released']*100;ax.barh(y,v,color=c,height=.48)
            ax.text(min(v+1.5,101),y,str(r['true_positive_released']),fontsize=8.8,va='center')
        ax.set(xlim=(0,110),yticks=yy,yticklabels=methods,xticks=[0,25,50,75,100],xlabel='Positive targets correctly released (%)');tidy(ax)
        ax.set_title(label,loc='left',fontweight='bold',pad=16,fontsize=9.7)
    b.tick_params(axis='y',length=0)
    f.text(.2,.065,'Numbers show correctly released positive labels. Low overall error can coexist with near-zero positive retention.',fontsize=8.8,color=GREY)
    save(f,'fig5_positive_retention')
if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    for fn in [arch,coverage,main_results,certificates,stress,positive]: fn()
    print('Saved 6 figures as PDF, SVG and 400-dpi PNG.')
