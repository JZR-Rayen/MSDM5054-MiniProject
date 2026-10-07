# MSDM5054 Mini-Project

HKUST MSDM5054 Statistical Machine Learning, Project 1.

## Team Members

- JIANG Zerui
- CHEN Kepeng
- MA Ruiyang
- WU Ziyi

## Topic

Home Credit payment-difficulty ranking using application information and historical credit records.

## Research Questions

1. How does ranking change when application-table information is expanded with basic historical aggregation and then 39 behavioural features, across logistic regression, random forest and LightGBM pipelines?
2. With enhanced history held fixed, what additional gains and trade-offs arise from bounded LightGBM refinement and model fusion?

## Start Here

- **Read the scientific report:** [report/Report.pdf](report/Report.pdf).
- **Inspect the code and verify saved results:** [notebooks/Home_Credit_Submission.ipynb](notebooks/Home_Credit_Submission.ipynb).
- **Run, retrain or redraw:** [notebooks/README.md](notebooks/README.md).
- **Download original inputs:** [data/README.md](data/README.md).

The report explains the scientific question, paired evidence, negative findings and limitations. The notebook supplies detailed technical explanations and saved-prediction verification. Default Run All does **not** retrain models or redraw figures; explicit commands are documented separately.

## Main Development Results

Mean ROC AUC across the same three development folds, with original pipeline configurations fixed within A/B/C:

| Pipeline | A: application-table only | B: basic history | C: enhanced history |
|---|---:|---:|---:|
| Logistic regression | 0.752941 | 0.762607 | 0.773946 |
| Random forest | 0.751752 | 0.762813 | 0.766000 |
| LightGBM | 0.757202 | 0.772092 | 0.780230 |

Refined LightGBM reached 0.781855; the rule-selected 80/20 LGB–LR rank fusion reached 0.782349. These are development results, not independent test estimates. A was retrospective; the holdout had already been viewed. Fold SD is not a confidence interval. CV batch ranking differs from the OOF-CDF transformation used for holdout/test. Full AUC/AP results and their trade-offs are in the report and notebook.

## Repository Structure

- `notebooks/`: one integrated notebook, required helpers, frozen training source and saved-result dependencies.
- `data/`: original-data download and placement instructions; no raw or processed training tables.
- `figures/`: figures displayed in the report/notebook and their redraw code.
- `report/`: the current eight-page course report.

Raw competition data, model archives, backups, review logs, historical delivery versions and ZIP files are excluded. Saved OOF scores remain because the notebook recomputes metrics from them rather than merely displaying a performance table.

## Contributions and AI Use

The recorded modules are data preparation/EDA (WU Ziyi), LR and integration (JIANG Zerui), RF (MA Ruiyang), and LightGBM (CHEN Kepeng). The report details AI assistance, evaluation boundaries and author responsibilities. The authors remain responsible for checking attribution and submitting the coursework.
