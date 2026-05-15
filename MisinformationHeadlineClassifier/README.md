# Misinformation Headline Classifier

This project builds a BERT-based binary classifier for detecting persuasion routes in news-style headlines. The final model classifies each headline as either:

- `central`: persuasion based on evidence, facts, logic, or substantive reasoning
- `peripheral`: persuasion based on emotional framing, cues, urgency, attention hooks, or non-substantive influence

Neutral headlines were consolidated into the `peripheral` class for the later binary models. This makes the final task focused on whether a headline uses central-route persuasion or relies on Peripheral persuasion cues.

## Main Result

The strongest model is the Combined Binary Model trained on both Health and Technology headlines. It achieved:

- Accuracy: `80.2%`
- Macro F1-score: `76.5%`

This result shows that the model can identify persuasion tactics across topics and performs better than large general-purpose foundation model baselines tested during evaluation, including Gemini at about `72%`.

## Model Variations

Two binary modeling strategies were tested.

### Combined Binary Model

Path: `Binary_model/`

This is the main model. It trains one binary BERT classifier on a combined dataset containing both Health and Technology headlines. The goal is to learn persuasion patterns that generalize across topic areas instead of depending on topic-specific models.

Classes:

- `central`
- `peripheral`

Main files:

- `train_both_models.py`: trains the combined model and topic-specific comparison models using 10-fold cross-validation
- `test_evaluation.py`: evaluates trained fold models on the held-out test set
- `Datasets/Train/training_data_balanced_2.json`: balanced binary training dataset
- `Datasets/Test/test_dataset_preprocessed.json`: binary test dataset

### Separate Routed Binary Model

This approach trains two specialized binary classifiers:

- one for Health headlines
- one for Technology headlines

A Stage 1 topic classifier routes each headline to the correct specialized model. This was tested to see whether topic-specific persuasion patterns performed better than one generalized model.

### Supporting 3-Class Experiments

The repository also keeps earlier 3-class route classification experiments:

- `main_route_classifier/`: one BERT model for `central`, `peripheral`, and `neutral`
- `hierarchical_model/`: topic prediction followed by topic-specific 3-class route prediction

These experiments are useful for comparison, but the final project direction is the binary persuasion-route model.

## Project Structure

```text
clean-headline-classifier/
├── Binary_model/
│   ├── Datasets/
│   │   ├── Train/
│   │   └── Test/
│   ├── train_both_models.py
│   └── test_evaluation.py
├── main_route_classifier/
│   ├── Dataset/
│   ├── scripts/
│   └── src/
├── hierarchical_model/
│   ├── Datasets/
│   ├── scripts/
│   └── src/
├── supplementary_scripts/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Main dependencies:

- `torch`
- `transformers`
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tqdm`

## Train The Main Binary Model

```bash
cd Binary_model
python3 train_both_models.py
```

The script trains and compares:

- random baseline
- Combined Health + Technology binary model
- Health-only binary model
- Technology-only binary model

It uses 10-fold stratified cross-validation and saves fold weights, logs, confusion matrices, and comparison plots.

Expected output folder:

```text
Binary_model/training_results/
```

Important output files include:

- `training_log_combined.json`
- `training_log_health.json`
- `training_log_tech.json`
- `model_comparison.png`
- `summary.txt`
- `weights_combined/`
- `weights_health/`
- `weights_tech/`

## Evaluate The Binary Model

After training, run:

```bash
cd Binary_model
python3 test_evaluation.py
```

This evaluates the trained fold models on the test dataset and compares them with baseline approaches.

Expected output folder:

```text
Binary_model/test_evaluation_results/
```

Important output files include:

- `test_summary.txt`
- `test_performance_plot.png`

## Data Format

The binary model uses JSON records with headline text, topic, and route labels.

Example:

```json
{
  "id": "sample_id",
  "text": "New study finds exercise may reduce heart disease risk",
  "topic": "health",
  "framework1_feature1": 1,
  "framework1_feature2": 0
}
```

Label mapping:

- `framework1_feature1 = 1` means `central`
- `framework1_feature2 = 1` means `peripheral`
- neutral headlines from earlier datasets are treated as `peripheral` in the later binary setup

## Supporting 3-Class Model Commands

These commands are kept for reproducibility of the earlier experiments.

### Train 3-Class Main Route Classifier

```bash
cd main_route_classifier
python3 scripts/train_enhanced.py \
  --data Dataset/BERT_training_3000_v2.json \
  --test-data Dataset/Testing_Dataset/Test1.json \
  --output-dir results_main \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --patience 4
```

### Train 3-Class Hierarchical Model

```bash
cd hierarchical_model
python3 scripts/train_all.py \
  --data Datasets/Train/BERT_training_V3.json \
  --test-data Datasets/Test/TruthSeeker_Test_116.json \
  --output-dir results_hierarchical \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --patience 4
```

## Supplementary Scripts

The `supplementary_scripts/` folder contains utilities for:

- plotting ROC and AUPRC curves
- plotting training curves
- comparing baselines
- exporting prediction tables
- generating readable result logs
- running hyperparameter search

See `supplementary_scripts/README.md` for details.

## Generated Files

Training creates checkpoints, plots, logs, and result folders. These are ignored by Git because they can be large and are reproducible.


