# Figures

The four report figures are `study_design`, `history_gains`, `model_diagnostics` and `refinement_effects` (PDF and PNG). Other PNGs support the technical notebook; they are not additional report figures.

`code/` contains the existing plotting implementation, shared style and panel-alignment check. Run `python notebooks/redraw_figures.py` from the repository root to regenerate figures in `figures/redrawn/` from saved plotting inputs. This does not retrain models or consume newly trained outputs automatically. See [notebooks/README.md](../notebooks/README.md#redraw-figures).
