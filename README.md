# Revision-Aware Uncertainty Estimation for Corporate Cash-Flow Forecasting

This research project forecasts next-quarter operating cash flow and builds prediction intervals using information available at each forecast date. It investigates whether previously observed changes to reported cash-flow values help estimate forecast uncertainty beyond ordinary cash-flow history.

This repository includes the forecasting pipeline, experiment diagnostics, tests, figure-generation tools, and accompanying IEEE manuscript.

## What the project does

1. Downloads and caches SEC EDGAR Company Facts and submissions metadata for a supplied list of company identifiers (CIKs).
2. Extracts USD operating cash-flow facts from 10-Q, 10-K, and amended filings using `us-gaap:NetCashProvidedByUsedInOperatingActivities`. It matches accessions to reporting periods and treats facts as observable on the day after filing.
3. Reconstructs standalone quarterly cash flow from year-to-date disclosures using only vintages available at the forecast origin. The target is the next quarter's first-publication cash flow.
4. Builds eight-quarter cash-flow history features and revision features: change count, normalized magnitude, share of affected periods, recency, and number of revised periods. Cash flows are normalized by the median absolute cash flow in the historical window, with a minimum scale of one dollar.
5. Fits a shared histogram gradient-boosting point forecast and compares four split-conformal interval methods with chronological training, calibration, and test partitions.
6. Writes predictions, evaluation metrics, audit records, figures, and LaTeX tables for an accompanying IEEE-format manuscript.

## Interval methods

| Method in code | How interval uncertainty is estimated |
| --- | --- |
| `constant` | A common radius calibrated from absolute forecast errors. |
| `feature_scale` | A learned error scale using cash-flow history features. |
| `revision_scale` | A learned error scale using cash-flow history plus revision features. |
| `permuted_revision` | A control that shuffles revision features during scale-model training. |

The learned scale models predict `log1p(absolute error)` on a later portion of the training period. Each method calibrates its error scores on a separate calibration set. The default `alpha=0.10` targets 90% coverage. Evaluation includes empirical coverage, normalized interval width, interval score, and point-forecast mean absolute error. Lower interval scores reward narrower intervals while penalizing missed targets.

## Run locally

Use Python 3.10 or newer. Dependencies include NumPy, pandas, scikit-learn, requests, Matplotlib, and pytest; install the versions specified in `requirements.txt`.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

### Offline demonstration

```bash
python src/racfp.py --demo --output outputs/synthetic_demo
python scripts/make_figures.py --results outputs/synthetic_demo
```

This generates artificial data to exercise the workflow. Synthetic charts are watermarked and provide no evidence about performance on SEC filings.

### SEC pilot

Set `SEC_USER_AGENT` to your actual name or organization and contact email before running:

```bash
export SEC_USER_AGENT="Your Name your.email@institution.edu"

python src/racfp.py \
  --ciks-file configs/sample_ciks.txt \
  --cache cache \
  --output outputs/sec_pilot \
  --train-end 2020-12-31 \
  --cal-end 2022-12-31 \
  --test-end 2025-12-31 \
  --alpha 0.10

python scripts/make_figures.py --results outputs/sec_pilot
python diagnose_revision_experiment.py --results outputs/sec_pilot
```

Network access is required for uncached SEC responses. The downloader caches responses and retries failed requests. Review `download_errors.json` for company-level failures. If you change the experiment dates, pass matching dates to the diagnostic script. For a larger study, supply a predefined and documented CIK cohort instead of the convenience pilot list.

The pipeline also accepts a prepared CSV through `--input-panel path/to/panel.csv`. Required fields are checked in `src/racfp.py`; these runs are marked `external_unverified`, and their figures carry an unverified-panel watermark.

## Outputs and figures

Each completed experiment produces `panel.csv`, `predictions.csv`, `metrics.csv`, `sample_counts.json`, `paired_bootstrap.json`, LaTeX tables, and a `run_manifest.json` recording provenance, split dates, and a panel hash. SEC processing additionally writes raw vintages, source hashes, data-audit information, and download errors. The diagnostic script adds `diagnostics.json`.

Figure generation creates coverage and interval-score comparisons, coverage by target year, interval score by prior revision exposure, and a predicted-scale versus observed-error diagnostic. The scale diagnostic is skipped when predicted scales have insufficient variation. Figures are saved under the results directory's `figures/` folder as PDF and PNG files, alongside supporting CSV summaries and a figure manifest.

To copy SEC-origin figures and the results table into the manuscript:

```bash
python scripts/make_figures.py --results outputs/sec_pilot --paper paper
```

The script blocks synthetic and externally supplied panels from automatic manuscript import. SEC provenance alone does not establish data validity; imported results still require author review. Compile `paper/main.tex` with pdfLaTeX and BibTeX, or upload the paper sources to Overleaf.

## Current evidence and limitations

The development project produced both a synthetic demonstration and a completed SEC pilot. The inspected SEC manifest recorded **829 panel rows and 196 test observations**, with status `completed_pipeline_not_independently_audited`. These are development-run observations, not experiments rerun while preparing this repository. Generated outputs and cached SEC responses are excluded from version control; use the commands above to generate them locally.

The saved pilot diagnostics report only **19 revision-exposed scale-training rows** and **4 revision-exposed test rows**. Revision-aware, feature-only, and shuffled-revision models produce identical test error-scale predictions in that run. Consequently, the pilot does not demonstrate an incremental benefit from revision features. Diagnostics also flag point-model labels that became public after the earliest scale-fitting forecast origin; rigorous cross-fitting requires addressing that timing overlap.

Observed numerical changes in SEC facts are not necessarily formal restatements. Historic accounting contexts, fiscal-year boundaries, original filings, and cohort selection require validation. Split-conformal coverage guarantees depend on assumptions such as exchangeability that financial time series may not satisfy. The paired firm-cluster bootstrap is descriptive and conditions on fitted models; it does not account for shared calendar shocks or training uncertainty. The manuscript and results require review before publication claims.
