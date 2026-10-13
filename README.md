# CC-Arbiter revision, version 3 (updated 2026-10-12)

Start with `output/pdf/main.pdf` (compact reading copy) and `output/pdf/main_review.pdf` (journal copy with double spacing and line numbers).

Status: no competing interests, no grant support and the author roles are entered in the manuscript. Ethics approval is not required (secondary analysis of public de-identified data; confirmed by the supervisor on 2026-10-12) and is stated accordingly. The code is released under the MIT license (see `LICENSE`) and the archive is published at https://github.com/binghe-glacier/cc-arbiter. The archive is deposited at Zenodo under DOI 10.5281/zenodo.23329663 (https://doi.org/10.5281/zenodo.23329663). Final approval of the submitted version by both authors remains an author decision. Nothing has been submitted or sent.

## Contents

- `output/pdf/main.pdf`, `main_review.pdf`, `supplement.pdf`, `figures.pdf`: manuscript (compact and journal formats), Additional file 1 and a figure gallery.
- `paper/`: editable LaTeX. `main.tex` compiles under XeTeX/Tectonic and under pdfLaTeX (TeX Live 2021), which the journal uses; `fonts.tex` selects the font route by engine. `main_review.tex` wraps `main.tex` with the journal formatting switch.
- `paper/table_cohorts.tex`, `table_outcomes.tex`, `table_subgroups.tex`: cohort composition, released-cell decomposition and the post hoc projection audit. `supp_positive_counts.tex` and `supp_subgroups.tex` are the fuller supplementary versions.
- `code/`: conformal sets, exact conditional-risk calibration, enforced LangGraph output, tests, analysis and rendering.
- `data/unit_scores.jsonl`: newly inferred scores for all 5,071 patient/report units.
- `data/nih_replication_scores.jsonl`: unchanged-expert scores for 1,582 additional NIH patients, excluded from all prior NIH splits.
- `ENHANCEMENT_PROTOCOL.md`: dated secondary analysis plan; original test results were known, additional patient outcomes were not.
- `REVIEW_ROUTING_PROTOCOL.md`, `data/review_scores.jsonl`: separately locked review-routing study with 2,557 further, disjoint NIH patients, 1,278 for calibration and 1,279 for testing.
- Author-side notes (submission package only): `作者声明填写说明.md` gives the exact declarations and what remains for the authors; `cover_letter.md` is the BMC Medical Imaging-oriented letter to confirm before sending.
- `LICENSE`: MIT license for the code in this archive.
- `data/cohorts.json`: original frozen source manifest; `cohorts_portable.json` expresses image paths relative to the source data root.
- `results/`: full estimates, certificates, candidate bounds, per-unit decisions and CSV summaries.
- `evidence/`: hashes, model provenance, numerical checks, test logs and graph replay checks.

## Recompute from frozen scores

Use Python 3.11 and the dependencies in `requirements-analysis.txt` and `requirements-review.txt`. The latter pins the scikit-learn version used to fit/export the separate routing stacker.

```powershell
python code/reproduce.py
python -m unittest discover -s code -p 'test_*.py' -v
python code/replay_graph.py --include-replication
python code/replay_review.py
```

The first command recreates all analyses, tables and nine vector figures without original images, model weights, network calls or an LLM. All 31 tests pass (re-run on 2026-10-12). Original/replication replay covers 6,210 invocations and 43,470 label decisions; additional routing replay covers 5,116 patient-level invocations. Both have zero discrepancies. The graph tests additionally need `requirements-graph.txt`. Original versions are recorded in `evidence/package_versions.json`; the fitted routing score records its scikit-learn version.

## Fresh model inference

The original data layout must include `chestxray14/train-00000.parquet`, `train-00001.parquet`, `iuxray/NLMCXR_reports.tgz` and the report-linked `iuxray/images/` tree. The manifest describes exactly the available local subsets; it is not a full official dataset split. Keep the published frozen split for comparison.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python code/infer.py --data-root 'D:\论文\MedRAX\data'
python code/reproduce.py
```

Inference needs a CUDA-enabled PyTorch installation, TorchXRayVision, PyArrow and the other recorded image dependencies. It downloads the two specified public weights if absent. It does not call a language model. Model outputs retain published operating-point normalization and should not be interpreted as calibrated disease probabilities.

`prepare_cohorts.py --data-root PATH` rebuilds the cohort manifest from raw files. Running it creates a new manifest/lock record, so use it in a new directory if preserving the original analysis lock. Exact image-byte hashes can differ from differently encoded public mirrors even when images appear similar.

## Build PDFs

```powershell
python code/build_pdf.py --tectonic 'D:\论文\CC\.tools\tectonic.exe'
python code/render_pdfs.py
```

The builder uses Tectonic/XeTeX with installed Times New Roman and Arial; `main_review.tex` is built the same way for the journal copy. Under pdfLaTeX the same sources fall back to the standard `mathptmx`/`helvet` fonts, which was verified with TeX Live 2021 (`pdflatex main.tex`, two passes, 21 pages, no errors). Font files are copied locally for the build and are not redistributed in the archive.

## New-patient replication from raw images

These optional steps require Requests, PyArrow, the inference dependencies, public network access and approximately 0.9 GB of raw data. Derived scores are already included for numerical reproduction.

```powershell
python code/download_replication.py
python code/replicate_patients.py --weights PATH_TO_XRV_WEIGHT_CACHE
python code/secondary_analysis.py
python code/make_secondary_artifacts.py
```

The download uses exactly the two predefined public shards at an immutable conversion commit. All 2,116 prior NIH patients are considered for exclusion. Removing 384 overlapping patients and selecting one image per remaining patient yields 1,582 new patients. Models and original NIH conformal/risk states remain frozen. This is same-source replication, not external-center validation. Raw images and model weights are not redistributed.

For the separate version-3 routing cohort, use `python code/download_review.py`, followed by `python code/prepare_review_data.py --weights PATH_TO_XRV_WEIGHT_CACHE`. These fetch the fixed four further shards and exclude all 3,698 prior NIH patients. Then run `evaluate_review.py` and `make_review_artifacts.py`. No additional downloads are needed when using the supplied derived scores.

## Interpretation

The review study is not a repair of automatic positive diagnosis. Both fixed routing scores fail the dual 10% calibration constraints, so all 1,279 new test patients are flagged, including all 267 positive patients. This preserves positives structurally and reduces no flags. No human review or clinical workload reduction was performed. The unflagged conditional risk is undefined. Do not report this as 100% diagnostic accuracy.

At the 10% primary target, CC-Arbiter releases 407/4,452 NIH label opportunities, with 10 errors and only 3/232 correctly retained positive targets (positive predictive value 27.3%). It releases nothing on IU. The mean-score gate releases more. These findings do not establish clinical utility. Certification is conditional on independent identically distributed units and the specified annotation target; IU report independence is unverified. Zero release has undefined conditional risk. Corruption studies intentionally violate the clean-distribution premise.

New-patient replication yields 1,195/11,074 releases, 23 errors and 8/507 correctly retained positives. At equal per-label budgets, mean-score ranking incurs fewer errors in both NIH cohorts (5 versus 10; 14 versus 23), but recovers fewer positives. Calibration-only Youden and minimum-error thresholds expose operating-point dependence. The post hoc AP/PA audit is descriptive and carries no subgroup guarantee. These secondary rules have no 308-family certificate; comparisons and pointwise intervals do not establish clinical superiority.

`PROTOCOL.md` is the internally frozen specification, not a public preregistration. The precision correction, label mapping limitations and difference between clinical truth and annotation targets are documented in the supplement. No old annotation-only LLM comparison is presented as a repaired intervention result.
