from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]; D=json.loads((ROOT/'results/results.json').read_text()); OUT=ROOT/'paper'
def pct(x,d=1):return '--' if x is None else f'{100*x:.{d}f}'
def main():
    table=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Dataset & Policy & Release (\%) & Errors / released & Risk (\%) & TP / positives\\',r'\midrule']
    csvrows=[]
    for ds in ['nih','iu']:
        for name,m in D['datasets'][ds]['targets']['0.1']['policies'].items():
            table.append(f"{ds.upper()} & {name} & {pct(m['release'])} & {m['errors']} / {m['accepted']} & {pct(m['selective_risk'])} & {m['true_positive_released']} / {m['positive_targets']}"+r'\\')
            csvrows.append({'dataset':ds,'policy':name,**{k:v for k,v in m.items() if k not in ['per_label','ci']}})
        table.append(r'\midrule')
    table[-1]=r'\bottomrule';table.append(r'\end{tabular}');(OUT/'table_results.tex').write_text('\n'.join(table))
    with (ROOT/'results/primary_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=csvrows[0].keys());w.writeheader();w.writerows(csvrows)
    lines=[]
    for ds in ['nih','iu']:
        for name,x in D['datasets'][ds]['coverage'].items():
            lines += [r'\subsection{'+ds.upper()+': '+name+'}',r'\begin{center}\small',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Finding & $n_+$ & Coverage$_+$ (\%) & $n_-$ & Coverage$_-$ (\%) & Singleton (\%) & Mean $|C|$\\',r'\midrule']
            for label,v in x['per_label'].items():
                lines.append(f"{label} & {v['positive']['n']} & {pct(v['positive']['coverage'])} & {v['negative']['n']} & {pct(v['negative']['coverage'])} & {pct(v['singleton'])} & {v['mean_size']:.2f}"+r'\\')
            lines += [r'\bottomrule\end{tabular}\end{center}']
    (OUT/'supp_coverage.tex').write_text('\n'.join(lines))
    lines=[]
    for ds in ['nih','iu']:
        for mode in ['mean','intersection']:
            lines += [r'\subsection{'+ds.upper()+': '+('mean-score gate' if mode=='mean' else 'CC-Arbiter')+'}',r'\begin{center}\small',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Finding & Threshold & $N_{\rm cal}$ & $E_{\rm cal}$ & Upper bound (\%) & $N_{\rm test}$ & $E_{\rm test}$\\',r'\midrule']
            cert=D['datasets'][ds]['targets']['0.1']['certificates'][mode];met=D['datasets'][ds]['targets']['0.1']['policies']['Risk gate (mean)' if mode=='mean' else 'CC-Arbiter']['per_label']
            for label,c in cert.items():
                th='--' if c['threshold'] is None else f"{c['threshold']:.2f}"
                lines.append(f"{label} & {th} & {c['n']} & {c['errors']} & {pct(c['bound'],2)} & {met[label]['accepted']} & {met[label]['errors']}"+r'\\')
            lines += [r'\bottomrule\end{tabular}\end{center}']
    (OUT/'supp_certificates.tex').write_text('\n'.join(lines))
    rows=[]
    for ds in ['nih','iu']:
        for s in D['datasets'][ds]['stress']:
            for name,m in s['policies'].items():rows.append({'dataset':ds,'corruption':s['mode'],'requested_rate':s['requested_rate'],'corrupted_units':s['corrupted_units'],'policy':name,'accepted':m['accepted'],'errors':m['errors'],'release':m['release'],'risk':m['selective_risk'],'positive_recall':m['positive_recall_released']})
    with (ROOT/'results/stress_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
