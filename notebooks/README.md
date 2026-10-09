# Notebook and Reproduction

The only reading/execution notebook is [Home_Credit_Submission.ipynb](Home_Credit_Submission.ipynb). Keep this repository's relative directory layout intact.

## Environment

Reference runtime: Python 3.13.9. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r notebooks/requirements.txt
python -m ipykernel install --user --name msdm5054 --display-name "MSDM5054"
```

Use that kernel in your Jupyter-compatible editor. On Windows, activate the environment using `.venv\Scripts\activate`. LightGBM requires its platform's OpenMP runtime for training; on macOS use a compatible `libomp` installation. No machine-specific runtime binary is shipped.

## Verify Saved Results

Open the notebook, restart its kernel and run all cells. Default execution requires no original dataset and performs zero model fits. It checks file/source hashes and ID/label/fold alignment, recomputes AUC/AP from saved predictions, reproduces the frozen secondary selection rules, displays tables and loads existing images.

This is saved-result verification, not training, model replay or fresh plotting. `support/` contains only its required evidence and redraw/retraining inputs. `source_snapshot/` preserves the actual original training implementations without code changes. The source packet retains original historical protocol hashes; the repository manifest covers this smaller delivery, not the discarded course-package files.

## Redraw Figures

From the repository root:

```sh
python notebooks/redraw_figures.py
```

This regenerates seven technical plots and four main-report figures from authenticated saved CSVs into `figures/redrawn/`, including alignment checks. It performs no training and preserves the checked-in images. The notebook continues to display those checked-in images. Its EDA illustration displays both the label distribution and application missingness; the redraw command regenerates this combined technical figure.

## Retrain From Original Data

Download inputs as described in [data/README.md](../data/README.md). Replace the example paths below with absolute paths. Each output directory must be new and nonexistent.

```sh
python notebooks/reproduce.py data --data-root /path/to/input_root --output /path/to/new_run/data_build
python notebooks/reproduce.py abc --inputs /path/to/new_run/data_build --output /path/to/new_run/abc
python notebooks/reproduce.py secondary --inputs /path/to/new_run/data_build --abc-results /path/to/new_run/abc/results/oof_predictions.csv --output /path/to/new_run/secondary
```

- `data` reconstructs A/B/C and fixed membership from raw CSVs, with zero fits. It checks raw fingerprints and the original information definitions.
- `abc` defaults to all three groups, models and folds: 27 outer fits plus nine LightGBM stopping fits, **36 fits**.
- `secondary` reproduces the 18 frozen configuration/feature combinations: **108 fits**. Supplying `--abc-results` uses newly trained C LR/RF predictions for fusion. No new candidates or holdout-based selection are introduced.

The complete development route totals **144 fits**. It excludes full-development diagnostic fits, viewed-holdout scoring and official-submission model fitting; those displays remain saved historical evidence. Original final-fit code is included for inspection. Retraining outputs do not automatically replace the notebook's saved scores or feed the redraw command.

For a bounded training check, add `--groups A --models LR RF LGB --folds 0` to `abc`, or `--tags C11_base` to `secondary`. These are partial checks, not the full experiment.

### Existing Processed-Input Alternative

If you already have the exact original processed inputs and split:

```sh
python notebooks/train_fixed.py --data-root /path/to/original_project --output /path/to/new_abc_run --groups A B C
```

Required inputs are `processed_data/application_train_processed_v1.csv`, `application_train_processed_v2.csv`, `baseline/splits/membership.csv` and `home-credit-default-risk/application_train.csv`. Exact hashes are checked. The notebook's last cell can invoke this route by explicitly setting `RUN_FIXED_CV=True`, `DATA_ROOT`, and `GROUPS_TO_TRAIN`; its default remains disabled and the example group is A only.

## Evidence Boundaries

Original A/B/C development counts are 292,135 applicants, three common folds; the 15,376-person holdout was previously viewed. A was added after B/C. Original B/C outer models were not saved, so their default check authenticates predictions rather than replaying models. Mean fold AUC/AP is distinct from pooled OOF metrics; SD uses ddof=1. Extra feature trials were not adopted. Raw fusion decreased AUC, and selected rank fusion does not improve all AP folds. Ranking outputs are not calibrated probabilities.

Historical verification on 7 October rebuilt raw features, replayed nine saved A models and performed ten isolated verification fits. That was not complete 144-fit retraining. The 9 October midterm-report finalization and GitHub update verify saved results with zero model fits; they do not change scores or model choices.
