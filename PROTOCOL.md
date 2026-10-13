# CC-Arbiter revision protocol

Frozen before revised inference and outcome evaluation on 2026-10-09. This is an internal prospective analysis specification, not a public preregistration. Earlier manuscript results have already been inspected. No test-set model, threshold, endpoint, or split selection is permitted.

## Cohorts and prediction target

Use both complete local NIH ChestX-ray14 parquet shards. Select one image per patient using the smallest SHA-256 of image bytes; no outcome-based filtering. Use all locally available IU report-linked PNGs, including normal and unmapped reports. Aggregate scores by the maximum across all available views within each report. IU has report identifiers, not verified patient identifiers; independence there is an explicit assumption.

Seven findings shared by NIH, the expert output ontologies and IU MeSH mapping: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Pneumonia, Pneumothorax. NIH targets are released text-mined labels. IU targets are the presence of the specified MeSH terms in the associated report; absent term means target 0 (not proof of clinical absence). Preserve report major/minor terms and mapping. Neither target is newly adjudicated clinical truth.

Models: TorchXrayVision densenet121-res224-chex and densenet121-res224-mimic_ch, frozen publicly released weights. Retain XRV operating-point normalization; outputs are scores in [0,1], not calibrated disease probabilities. Both are DenseNet-121; do not claim architectural diversity. Published training sources are CheXpert and MIMIC-CXR, respectively, rather than NIH or IU. Save versions, URLs and weight hashes. Scores are freshly inferred; no predictions from the earlier all-source models are reused.

## Splits

Sort unit IDs and shuffle with NumPy default_rng(20261009), independently per dataset. First floor(0.40 n) units: conformal calibration. Next floor(0.30 n): risk calibration. Rest: locked test. NIH unit = patient, one selected image; IU unit = report, all associated views. No outcome stratification or model training. Save full unit manifests before inference. Exclude only unreadable inputs/missing XML, record every exclusion and rerun neither selection nor split after seeing outcomes.

## Policy

Class-conditional split-conformal sets per expert and label, score |y-p|, alpha=0.10, finite-sample quantile with +infinity for insufficient support. Intersection singleton proposes its sole member. Empty intersection or two-member intersection defers. No comparison of raw nonconformity across models. The scalar uncertainty is 1 - mean expert support for the proposed member; it may exceed 0.5 and then cannot be accepted on the fixed grid. Two policies are risk-calibrated: average-score prediction and intersection-singleton proposal.

For each of 11 thresholds {0,0.05,...,0.50} and seven labels, compute a one-sided Clopper-Pearson upper bound on error among eligible, accepted risk-calibration units. Zero accepted units is uncertified with bound 1. Fix family size M=2 datasets x 2 policies x 7 labels x 11 thresholds=308; use tail probability 0.05/M. The bounds are common across risk targets 0.05,0.10,0.20. Choose the largest certified threshold, otherwise defer all (threshold null, never 0). The claim is a simultaneous high-probability bound on population label-wise conditional error under independent identically distributed units and a fixed pipeline, not on the realized test error or arbitrary clinical truth. Primary target 0.10. No extra numerical risk tolerance.

## Comparisons and metrics

Fixed baselines: always-negative; each expert at 0.5; mean score at 0.5; two-expert agreement at 0.5; CP singleton intersection without risk gate; mean-score risk gate; CC-Arbiter. Report release fraction, errors among releases, erroneous releases per all opportunities, correctly released positives and negatives, and positive-case release/sensitivity to expose majority-class effects. Give label-wise denominators. All-label summaries are descriptive, with 2,000 unit bootstrap replicates, seed 718; no treating seven labels from one unit as independent cases. Conditional risk undefined if no releases. Report set coverage and informativeness by expert and label; positive/negative strata separately.

Stress evaluation: freeze clean calibration and thresholds; flip score vectors p->1-p for 10%, 25%, 50% of test units, either expert B only or both experts with the same mask. Mask uses seed 1309 and is independent of labels/correctness. Report natural and corrupted errors, release and deferral for identical test cases across methods. These are synthetic tool-output failures, not clinical distribution-shift guarantees. No oracle filtering, ground-truth majority, or fabricated third expert.

## Integration and scope

Build graph nodes with the gated class before graph compilation. Evidence indexed by request, image/study and proposition; overwrite repeated calls from the same expert; no cross-image mixing. Release only a structured certified label through a final graph gate; planner prose cannot bypass it. Unsupported/failed experts, missing keys and uncalibrated labels defer. At least two distinct experts required. Final scope is binary finding evidence, not free-text diagnostic answers or VQA correctness. Deterministic graph tests exercise actual LangGraph dispatch and adversarial planner overrides. These software tests do not replace an end-to-end clinical or LLM benchmark.

## Reporting

Replace invalid earlier experiment claims, not retrofit them. Publish negative results and all listed baselines. Do not claim superiority, clinical utility, negligible runtime overhead, universal scaffold transfer or JBHI readiness without supporting evidence. Author contributions, ethics determination, funding and conflicts require author confirmation.
