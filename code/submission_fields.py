"""Export the exact manuscript title, abstract and keywords for portal entry."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'paper/main.tex').read_text(encoding='utf8')
title=re.search(r'\\title\{([^}]+)\}',s).group(1)
abstract=s.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
abstract=re.sub(r'\\textbf\{([^}]+)\}',r'\1',abstract).replace(r'\noindent','').replace(r'\par','\n\n').replace(r'\%','%').replace('--','-')
abstract='\n\n'.join(' '.join(x.split()) for x in abstract.split('\n\n') if x.strip())
assert '\\' not in abstract
keywords=re.search(r'\\textbf\{Keywords:\}\s*([^\n]+)',s).group(1)
text='Intended journal: BMC Medical Imaging\nArticle type: Research article (verify portal category)\n\nTitle\n'+title+'\n\nAuthors (as in source manuscript; final confirmation pending)\nYaocong Yang; Xiangyong Kong\n\nAbstract\n'+abstract+'\n\nKeywords\n'+keywords+'\n\nNo grant support and no competing interests are declared. Ethics approval was not required (public de-identified data; confirmed by the supervisor on 2026-10-12). Final approval of the submitted version by both authors is still to be recorded. This file is for form entry and has not been submitted.\n'
(ROOT/'submission_fields.txt').write_text(text,encoding='utf8')
print('Exported title, abstract and keywords;',len(abstract.split()),'abstract words.')
