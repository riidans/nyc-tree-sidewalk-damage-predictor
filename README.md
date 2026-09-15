# NYC Tree-Sidewalk Damage Predictor

A machine-learning model pipeline for predicting whether a New York City street tree is associated with sidewalk damage. The full analysis, methodology, experiment results, and references are available in [`report.pdf`](report.pdf).

## Overview

The project uses the 2015 NYC Street Tree Census to prioritize trees for sidewalk inspection. The target is a binary `sidewalk` damage label. Because missing a damaged sidewalk is more costly than sending an inspector, we wanted to prioritize recall and uses the F2 score rather than accuracy as the main metric. 

The final model is a hyperparameter-tuned random forest. On held-out test data, it achieved:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.5829 |
| Precision | 0.4012 |
| Recall | 0.9077 |
| F2 score | 0.7247 |

The model identified 17,065 damaged sidewalks correctly and missed 1,735. The high false-positive count is intentional as the model is designed to flag most potential hazards for inspection.

## Dataset

The source data is the [2015 Street Tree Census - Tree Data](https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/uvpi-gqnh) from NYC OpenData. It contains more than 680,000 tree records with biological, health, geographic, and infrastructure-related attributes.

Our preprocessing workflow:

1. Keeps living trees with a non-missing sidewalk label, producing 652,166 records.
2. Uses an 80% training, 10% development, and 10% test split.
3. Groups less frequent tree species into `Other`, retaining the 20 most common species.
4. Applies class weighting because damaged sidewalks are the minority class.
5. Optimizes classification cutoffs on the development set using F2 score.

## Models and experiments

The project evaluates:

- Logistic regression as an interpretable baseline.
- Random forest using the `ranger` package.
- XGBoost with one-hot encoded categorical variables.
- Feature engineering across size, species, care, and interaction features.
- HDBSCAN/DBOpt-based spatial feature transformations.

The baseline random forest outperformed the baseline logistic regression and XGBoost models. Hyperparameter tuning found the best random forest configuration at `mtry = 6` and `min.node.size = 5`, with 500 trees and a class-weight ratio of approximately 2.48. Feature engineering and spatial clustering did not improve performance enough to justify replacing the original features.

## Repository structure

```text
data_cleanup.R                         Data cleaning and preparation
final_model_training.R                  Final model training workflow
cleaned_tree_data.rds                   Cleaned dataset
final_model.rds                         Saved final model (Git LFS)

logistic_regression/                    Logistic regression baselines
random_forest/                          Random forest baselines and tuning
feature_engineering/                   Feature engineering experiments
clustering/                             DBOpt and HDBSCAN experiments
```

## Requirements

The modeling workflow is written primarily in R. Install the packages required by the scripts, including:

- `ranger`
- `xgboost`
- `pROC`
- `caret`
- `dplyr`
- `data.table`
- `hdbscan`

The clustering utilities also include Python scripts and require the Python dependencies used by those scripts and DBOpt.

## Running the project

1. Place the NYC Street Tree Census CSV in the repository root.
2. Run `data_cleanup.R` to create the cleaned data file.
3. Run the scripts in `logistic_regression/` and `random_forest/` for baseline and tuning experiments, or run `final_model_training.R` to train the final model.
4. Use the scripts in `feature_engineering/` and `clustering/` for the additional experiments described in the report.

Large model and dataset artifacts are stored with [Git LFS](https://git-lfs.com/).

