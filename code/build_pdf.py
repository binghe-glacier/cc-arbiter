"""Build the manuscript with Tectonic/XeTeX and locally installed fonts.

System font binaries are build inputs and are NOT redistributed in the archive.
"""
from pathlib import Path
import argparse,os,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tectonic',default='tectonic');ap.add_argument('--font-dir',type=Path,default=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts');args=ap.parse_args()
    dest=ROOT/'paper/local-fonts';dest.mkdir(exist_ok=True)
    for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf','arial.ttf','arialbd.ttf','ariali.ttf']:
        src=args.font_dir/name
        if not src.exists():raise FileNotFoundError(f'Install licensed Times New Roman / Arial fonts or pass --font-dir. Missing: {src}')
        shutil.copy2(src,dest/name)
    (ROOT/'output/pdf').mkdir(parents=True,exist_ok=True)
    for name in ['main','main_review','supplement']:
        with (ROOT/f'evidence/compile_{name}.txt').open('w',encoding='utf8') as log:
            subprocess.run([args.tectonic,name+'.tex','--outdir','../output/pdf','--keep-logs'],cwd=ROOT/'paper',stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()
