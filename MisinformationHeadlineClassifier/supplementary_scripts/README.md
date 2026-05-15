# Supplementary Scripts

All optional utility scripts are kept in this single folder. They are not required for the core training/evaluation paths, but they are useful after training or for analysis.

## Main Route Classifier Utilities

- `compare_baselines.py` — compare a trained main-route model against random/class-prior baselines.
- `visualize_dataset.py` — summarize and plot class/topic/length distributions for a 3-route dataset.
- `visualize_training.py` — plot loss/F1/accuracy curves from a training log.
- `generate_results_log.py` — convert a results directory into a readable `results_log.txt`.
- `hyperparameter_search.py` — run grid/random hyperparameter search for the main route classifier.
- `generate_plots.py` — generate learning/evaluation plots for a trained main-route model.
- `scratch_generate_test1_plots_new.py` — cleaned Test1-specific plot generator for the main-route model.

## Hierarchical Model Utilities

- `generate_predictions_md.py` — generate a markdown table of hierarchical predictions.

## Binary Model Utilities

- `plot_roc_models.py` — plot ROC curves for binary fold ensembles.
- `plot_auprc_models.py` — plot peripheral-class AUPRC curves for binary fold ensembles.
- `plot_auprc_micro.py` — plot micro-average AUPRC curves for binary fold ensembles.

