# Secondary analysis and patient replication protocol

Version 2, 9 October 2026. This is a dated internal analysis plan, not a registry preregistration. The original test results and preliminary AUROC/AP summaries were already inspected before this plan. All additions on those tests are secondary and exploratory. Original PROTOCOL.md, patient splits, models, conformal sets, candidate family and risk gates remain unchanged.

## Fair operating-point comparisons

Use the mean of the same two released expert scores. Fit one threshold per finding and dataset using ONLY the original conformal-calibration units. Consider every distinct calibration score and an additional threshold just above its maximum; predict positive at score >= threshold. Fit (a) maximum Youden sensitivity + specificity - 1 and (b) minimum empirical classification error. Break ties by the largest threshold. If a class is missing, use the error-minimizing threshold for both rules and record this fallback. No threshold is fitted on risk-calibration or test outcomes. These secondary baselines have no risk certificate and do not extend the original 308-rule confidence family.

For each original test cohort, also compare at exactly the CC-Arbiter 10% policy's per-label release counts. Rank all patients/reports independently for each finding by descending absolute distance of mean score from (a) 0.5 and (b) the calibration-only Youden threshold. Resolve exact ties by ascending SHA256 of the unit ID. Take the same number per finding as CC-Arbiter. Predictions are the corresponding thresholded mean scores. No reference labels enter either selection. These are retrospective batch-budget comparisons, not deployable certified threshold policies. Bootstrap paired differences conditional on the chosen batch masks; do not describe these intervals as population uncertainty for a future adaptive batch rule.

## Discrimination and uncertainty

Report finding-specific AUROC and average precision (AP) for both experts and their mean, along with positive counts and prevalence. Use 2,000 unit bootstrap resamples with seed 20261010; preserve the seven correlated labels within each resampled unit and the pairing of policies. AUROC and AP replicates lacking either target class are undefined and omitted, with valid replicate counts reported. Intervals are descriptive pointwise percentile 95% intervals, without multiplicity-adjusted superiority claims. Report calibrated-threshold sensitivity, specificity, released positive recall, release and error; do not infer probability calibration from score discrimination.

## Newly acquired patient replication

Before evaluating any additional outcomes, fix the two public parquet shards at https://huggingface.co/datasets/BahaaEldin0/NIH-Chest-Xray-14: `default/test/0.parquet` and `default/test/1.parquet` via the repository parquet endpoint. Resolve their immutable conversion commit and file paths if available; record actual download URL, SHA256 and row counts. Do not change shard choice based on findings. If these exact shards cannot be obtained, report the unavailable replication, without replacing it by a favorable subset.

Exclude every patient in the original 2,116-patient NIH manifest, irrespective of original split. Also exclude duplicate image content matching original selected images. Select one image per remaining patient by the smallest SHA256 of its encoded bytes, as in the original protocol; collapse any duplicate-image components across remaining patient IDs. Preserve source rows and patient exclusion counts. Downloaded shards may have their own dataset split names, but here they serve only as a new evaluation cohort. Never use them for fitting, threshold adjustment or model selection.

Run the identical frozen CheXpert/MIMIC DenseNet releases with the existing preprocessing, FP32, TF32 disabled and deterministic inference. Restore the ORIGINAL NIH conformal quantiles and risk-gate states; also apply thresholds fitted on the original conformal units for the secondary baselines. Report all eight primary policies at risk targets .05/.10/.20, secondary threshold rules at full release and matched-budget rules at .10, discrimination and annotation prevalence. Use 2,000 patient bootstraps with the same seed convention. This is patient-disjoint same-source replication, not external-center validation or a clinical adjudication study. Original certificate transfer remains conditional on the same data-generating distribution; disjoint patients alone do not establish it.

## Certification feasibility

Derive the zero-error lower acceptance-count requirement `ceil(log(delta/M)/log(1-r))` for each original target, delta=.05 and M=308. Compare available accepted risk-calibration units with this necessary count, and distinguish insufficient accepted support from observed errors. Add transparent discussion of positive retention and the costs of separate positive-output error certification; do not retroactively change the original target or candidate family.

## Reporting and integrity

Keep primary results unmodified. Store this plan's digest and creation time before new-cohort inference, and hash original score, calibration and result files. Package all secondary code, calibrated thresholds, new score records, audits and analysis JSON. Raw images and model weights are not redistributed. Record failed downloads/inference and any protocol deviations. Revise the manuscript's conclusions to match the results, including negative findings. Do not add fabricated clinical labels, LLM experiments or author declarations.
