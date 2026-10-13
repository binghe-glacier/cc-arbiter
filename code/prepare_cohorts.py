"""Build deterministic manifests without evaluating model outcomes."""
from pathlib import Path
import argparse, collections, hashlib, json, tarfile
import xml.etree.ElementTree as ET
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Pneumonia','Pneumothorax']
MESH = {'Pulmonary Atelectasis':'Atelectasis', 'Cardiomegaly':'Cardiomegaly',
        'Consolidation':'Consolidation', 'Pulmonary Edema':'Edema',
        'Pleural Effusion':'Effusion', 'Pneumonia':'Pneumonia', 'Pneumothorax':'Pneumothorax'}
def sha(data): return hashlib.sha256(data).hexdigest()
def write(path, obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')
def split_units(units):
    ids=sorted(units); np.random.default_rng(20261009).shuffle(ids)
    n=len(ids); a=int(.4*n); b=a+int(.3*n)
    for i,key in enumerate(ids): units[key]['split']='conformal' if i<a else ('risk' if i<b else 'test')
    return [units[k] for k in sorted(units)]
def main(data_root):
    excluded=[]; nih={}; source_info=[]
    for fp in sorted((data_root/'chestxray14').glob('*.parquet')):
        try: table=pq.read_table(fp)
        except Exception as e:
            excluded.append({'source':str(fp),'reason':type(e).__name__}); continue
        source_info.append({'path':str(fp),'bytes':fp.stat().st_size,'sha256':sha(fp.read_bytes()),'rows':len(table)})
        for i,row in enumerate(table.to_pylist()):
            key='pid'+str(row['Patient ID']); digest=sha(row['image']['bytes'])
            im={'source':str(fp),'row':i,'key':digest,'original_path':row['image']['path']}
            unit={'id':key,'images':[im],'y':[int(k in row['label']) for k in LABELS],
                  'raw_labels':row['label'],'view':row['View Position'],'age':row['Patient Age'],'sex':row['Patient Gender']}
            if key not in nih or digest<nih[key]['images'][0]['key']: nih[key]=unit
    image_dirs=collections.defaultdict(list)
    for fp in sorted((data_root/'iuxray/images').rglob('*.png')):
        image_dirs[fp.parent.name.split('_')[0]].append(fp)
    iu={}; xml_count=0; missing=0
    archive=data_root/'iuxray/NLMCXR_reports.tgz'
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            if not m.name.endswith('.xml'): continue
            xml_count+=1; doc=ET.fromstring(tf.extractfile(m).read()); el=doc.find('uId')
            if el is None: continue
            key=el.get('id'); terms=[e.text.strip() for mesh in doc.iter('MeSH') for e in mesh if e.tag in ('major','minor') and e.text]
            present={MESH[t.split('/')[0].strip()] for t in terms if t.split('/')[0].strip() in MESH}
            if not image_dirs[key]: missing+=1; continue
            ims=[{'source':str(p),'key':sha(p.read_bytes())} for p in image_dirs[key]]
            # Duplicate copies within a report never contribute twice.
            ims=list({x['key']:x for x in ims}.values())
            iu[key]={'id':key,'images':ims,'y':[int(k in present) for k in LABELS], 'mesh_terms':terms,
                     'license_url':doc.findtext('licenseURL')}
    # Connected components for duplicated images across group identifiers.
    # Retain one unit per component, avoiding image duplication across splits.
    dedup={}
    for name,units in [('nih',nih),('iu',iu)]:
        parent={k:k for k in units}; seen={}
        def find(k):
            while parent[k]!=k: parent[k]=parent[parent[k]]; k=parent[k]
            return k
        for k,u in units.items():
            for im in u['images']:
                h=im['key']
                if h in seen:
                    a,b=find(k),find(seen[h]); parent[max(a,b)]=min(a,b)
                else: seen[h]=k
        drop=[k for k in units if find(k)!=k]; dedup[name]=len(drop)
        for k in drop: excluded.append({'dataset':name,'id':k,'reason':'duplicate image component','retained':find(k)}); del units[k]
    manifest={'labels':LABELS,'mesh_mapping':MESH,'nih':split_units(nih),'iu':split_units(iu)}
    write(ROOT/'data/cohorts.json',manifest)
    write(ROOT/'evidence/cohort_audit.json',{'nih_sources':source_info,'iu_xml_reports':xml_count,'iu_reports_without_local_images':missing,
         'iu_local_image_reports':sum(bool(v) for v in image_dirs.values()),'duplicate_units_removed':dedup,'exclusions':excluded,
         'summary':{k:{s:{'units':sum(u['split']==s for u in manifest[k]),'images':sum(len(u['images']) for u in manifest[k] if u['split']==s),
                   'positives':np.sum([u['y'] for u in manifest[k] if u['split']==s],axis=0).tolist()}
                    for s in ['conformal','risk','test']} for k in ['nih','iu']}})
    write(ROOT/'evidence/protocol_lock.json',{'protocol_sha256':sha((ROOT/'PROTOCOL.md').read_bytes()),'manifest_sha256':sha((ROOT/'data/cohorts.json').read_bytes()),
          'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})
    print(json.dumps(json.loads((ROOT/'evidence/cohort_audit.json').read_text())['summary'],indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,default=Path('D:/论文/MedRAX/data')); main(ap.parse_args().data_root)
