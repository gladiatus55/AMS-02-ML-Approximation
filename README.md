# AMS-02 ML Approximation

This repository contains a machine learning framework for approximating AMS-02 (Alpha Magnetic Spectrometer) data using various regression models.

## Overview

The project implements multiple machine learning models to approximate AMS-02 data across different energy bins. The supported models include:

- XGBoost
- Linear Regression
- Random Forest
- Support Vector Regression (SVR)
- Stochastic Gradient Descent Regression (SGDR)
- Transformer (separate python file)

## Features

- Multiple model support with automatic hyperparameter tuning
- Feature selection based on correlation analysis
- Performance metrics including R², RMSE, and etaRMS
- Visualization of model performance across energy bins
- Detailed logging and result tracking
- Output organization with timestamped directories

## Requirements

Required packages are listed in requirements.txt. The main dependencies include:

```
pandas
numpy
matplotlib
scikit-learn
xgboost
```

## Usage

```bash
python jgr-ml-models-ams02.py [OPTIONS]
```

### Command Line Arguments

- `--models`: Specify models (can be multiple separated by space) to run (choices: 'xgboost', 'linear', 'random_forest', 'svr', 'sgdr', 'all')
- `--train`: Path to training data CSV file (default: "train.csv")
- `--test_x`: Path to X test data CSV file (default: "X_test.csv")
- `--test_y`: Path to y test data CSV file (default: "y_test.csv")
- `--top_n`: Use top N correlated features (default: 0, disabled)
- `--specific_features`: Use specific features (default: None)

### Examples

Run all models:
```bash
python jgr-ml-models-ams02.py --models all
```

Run only XGBoost and SVR models:
```bash
python jgr-ml-models-ams02.py --models xgboost svr
```

Use top 10 correlated features:
```bash
python jgr-ml-models-ams02.py --top_n 10
```

Use specific features:
```bash
python jgr-ml-models-ams02.py --specific_features NM_Oulu "sigma-phi V, degrees" "R (Sunspot No.)" --models linear
```

## Output Structure

Results are saved in the outputs directory with timestamped subdirectories:
```
outputs/
  YYYYMMDD_HHMMSS/
    correlation_results.csv
    model_type/
      bin_X_prediction.csv
      final_prediction.csv
      gridsearch_params.txt
      results.txt
      model_type_r2_scores.png
```

## Model Performance

Each model's performance is evaluated using:
- R² score (coefficient of determination)
- Root Mean Squared Error (RMSE)
- etaRMS (percentage RMS error)

Performance metrics are visualized across all energy bins and saved as PNG files.

## License

Distributed under the GPLv3 License. See LICENSE file for more information.