"""Recalculate all published outputs from supplied frozen expert scores."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
for script in ['evaluate.py','make_tables.py','make_figures.py','secondary_analysis.py','make_secondary_artifacts.py','evaluate_review.py','make_review_artifacts.py','submission_fields.py']:
    subprocess.run([sys.executable,str(ROOT/'code'/script)],check=True,cwd=ROOT)
print('Rebuilt results, tables and vector figures. Use replay_graph.py for actual graph checks and build_pdf.py for PDFs.')
