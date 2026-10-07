"""Redraw the existing figures from saved plotting inputs; no model fitting."""
from pathlib import Path
import sys, shutil
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'figures/code'))
import plots
plots.BASE=BASE
plots.ROOT=BASE/'no_external_project'
plots.OUT=BASE.parent/'figures/redrawn'
plots.DATA=plots.OUT/'data'
shutil.copytree(BASE/'support/report_data',plots.DATA,dirs_exist_ok=True)
shared_inputs = {'membership.csv': 'membership.csv', 'abc_summary.csv': 'abc/cv_summary.csv', 'abc_deltas.csv': 'abc/paired_deltas.csv', 'abc_folds.csv': 'abc/cv_folds.csv', 'missing.csv': 'data/missing_statistics.csv', 'blend_folds.csv': 'optimization/blend_folds.csv', 'optimization_folds.csv': 'optimization/cv_folds.csv', 'feature_trials.csv': 'optimization/feature_paired_deltas.csv', 'LR_importance.csv': 'interpretation/LR_importance.csv', 'RF_importance.csv': 'interpretation/RF_importance.csv', 'LGB_importance.csv': 'interpretation/LGB_importance.csv'}
for name, relative in shared_inputs.items():
    shutil.copy2(BASE/'support'/relative, plots.DATA/name)
plots.QA=plots.OUT/'qa'
plots.main()
import main_figures
main_figures.B=plots.OUT
main_figures.D=plots.DATA
main_figures.O=plots.OUT
main_figures.Q=plots.QA
main_figures.main()
print('Redrawn seven technical plots and four report figures in figures/redrawn; no models trained.')
