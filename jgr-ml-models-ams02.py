# from sklearn.experimental import enable_halving_search_cv # noqa
# # now you can import normally from model_selection
# from sklearn.model_selection import HalvingGridSearchCV
# from sklearn.pipeline import make_pipeline
import math

import argparse
import os
import time
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description='ML models for AMS-02 data approximation')
    parser.add_argument('--models', nargs='+', choices=['xgboost', 'linear', 'random_forest', 'svr', 'sgdr', 'all'],
                      default=['all'], help='Specify models to run (default: all)')
    parser.add_argument('--train', type=str, default="train.csv", 
                      help='Path to training data CSV file (default: train.csv)')
    parser.add_argument('--test_x', type=str, default="X_test.csv", 
                      help='Path to X test data CSV file (default: X_test.csv)')
    parser.add_argument('--test_y', type=str, default="y_test.csv", 
                      help='Path to y test data CSV file (default: y_test.csv)')
    parser.add_argument('--top_n', type=int, default=0, 
                      help='Use top N correlated features (default: 0, disabled)')
    parser.add_argument('--specific_features', nargs='+', 
                      help='Use specific features (default: None)')
    return parser.parse_args()


def create_output_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def create_model_subdirectory(output_dir, model_type):
    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def compute_correlations(data, feature_cols, bin_cols):
    correlation_results = []

    for bin_index, bin_name in enumerate(bin_cols):
        for feature_index, feature_name in enumerate(feature_cols):
            correlation_value = data[bin_name].corr(data[feature_name])
            correlation_results.append([
                bin_index, bin_name, feature_index, feature_name, correlation_value
            ])

    correlation_df = pd.DataFrame(correlation_results, columns=['Bin Index', 'Bin Name', 'Feature Index', 'Feature Name', 'Correlation'])
    return correlation_df

def run_model_with_grid_search(model_type, X_train, y_train, X_test, y_test):
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
        param_grid = {
            'max_depth': [3, 4],#, 5, 6, 7, 8, 9, 10, None],
            'eta': [0.01, 0.03, 0.05],#, 0.1, 0.15, 0.2],
            'n_estimators': [100, 200, 500, 600, 750, 800, 900, 1000]
        }
    elif model_type == 'linear':
        # Use a pipeline for Linear Regression with Standard Scaler
        pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('regressor', LinearRegression())])
        param_grid = {
            'regressor__fit_intercept': [True, False],
        }
        model = pipeline
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
        }
    elif model_type == 'svr':
        pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('regressor', SVR())])
        param_grid = {
            'regressor__C': [0.1, 1, 10],
            'regressor__gamma': ['scale', 'auto'],
            'regressor__kernel': ['linear', 'rbf']
        }
        model = pipeline
    elif model_type == 'sgdr':
        pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('regressor', SGDRegressor(max_iter=1000, tol=9e-3))])
        param_grid = {
            'regressor__alpha': [0.0001, 0.001, 0.01],
            'regressor__penalty': ['l2', 'l1', 'elasticnet'],
            'regressor__learning_rate': ['constant', 'optimal', 'adaptive']
        }
        model = pipeline
    else:
        raise ValueError(f"Model type {model_type} is not supported.")

    if param_grid != None:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, return_train_score=True)
        #grid_search = HalvingGridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, return_train_score=True)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        mean_train_score = np.mean(grid_search.cv_results_['mean_train_score'])
        print(f"R² Score on Train Data: {mean_train_score:.4f}")
    else: 
        print(f"Shape of X_train: {X_train.shape}")
        print(f"Shape of y_train: {y_train.shape}")

        X = np.vstack(X_train)
        y = np.vstack(y_train)
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")

        model.fit(X, y)
        mean_train_score = model.score(X, y)
        print(f"Training R² score: {mean_train_score:.4f}")
        best_params = {}    
    
    r2, mse, etaRMS, y_pred = print_score_on_test_data(best_model, X_test, y_test)
    
    return r2, mse, etaRMS, y_pred, best_params, mean_train_score, param_grid

def print_score_on_test_data(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = np.sqrt(mean_squared_error(y_test, y_pred))

    # RSS (Residual Sum of Squares} evaluation
    y_test_np = np.array(y_test)
    RSS = 0
    etaRMS = 0
    avg_y_test = 0
    avg_y_test_n = 0
    for ij in range(len(y_test)):
        tem = ((y_test_np[ij] - y_pred[ij])*(y_test_np[ij] - y_pred[ij]))
        RSS = RSS + tem
        etaRMS = etaRMS + (tem/(y_test_np[ij]*y_test_np[ij]))
        avg_y_test = avg_y_test + y_test_np[ij]
        avg_y_test_n = avg_y_test_n + 1

    avg_y_test = avg_y_test/avg_y_test_n
    etaRMS = etaRMS/avg_y_test_n
    etaRMS = 100. * math.sqrt(etaRMS)

    print(f"R² Score on Test Data: {r2:.4f}, RMSE: {mse:.4f}, etaRMS: {etaRMS:.4f}")
    return r2, mse, etaRMS, y_pred

def plot_r2_scores(r2_scores, bin_cols, model_type, model_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(bin_cols)), r2_scores, marker='o', linestyle='-')
    plt.xlabel('Bin Index')
    plt.ylabel('R² Score')
    plt.title(f'R² Scores for Each Bin ({model_type.capitalize()})')
    plt.xticks(range(len(bin_cols)), bin_cols, rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(model_dir, f"{model_type}_r2_scores.png")
    plt.savefig(plot_path)

def main():
    args = parse_arguments()

    if 'all' in args.models:
        model_types = ['xgboost', 'linear', 'random_forest', 'svr', 'sgdr']
    else:
        model_types = args.models

    use_top_n_features = args.top_n > 0
    top_n_features_count = args.top_n

    use_specific_features = args.specific_features is not None
    specific_features = args.specific_features if use_specific_features else ['NM_Oulu']

    # Create output directory
    output_dir = create_output_directory()

    # Print execution configuration
    print(f"Running with configuration:")
    print(f"  Models: {model_types}")
    print(f"  Training data: {args.train}")
    print(f"  Test data X: {args.test_x}, Y: {args.test_y}")
    if use_top_n_features:
        print(f"  Using top {top_n_features_count} features")
    if use_specific_features:
        print(f"  Using specific features: {specific_features}")
    print("--------------------------------------------------")

    # Load the training data
    train_data = pd.read_csv(args.train)
    
    # Load the test data and timestamp
    X_test_full = pd.read_csv(args.test_x)
    timeStamp = X_test_full['timeStamp'].copy()

    # Feature and bin columns
    feature_cols = train_data.columns[1:-30]
    bin_cols = train_data.columns[-30:]

    # Compute correlations
    correlation_df = compute_correlations(train_data, feature_cols, bin_cols)

    # Save correlation results
    correlation_path = os.path.join(output_dir, "correlation_results.csv")
    correlation_df.to_csv(correlation_path, index=False)
    
    for model_type in model_types:
        model_dir = create_model_subdirectory(output_dir, model_type)
        r2_scores = []

        model_predictions = pd.DataFrame(timeStamp, columns=['timeStamp'])

        param_info_path = os.path.join(model_dir, "gridsearch_params.txt")
        results_path = os.path.join(model_dir, "results.txt")
        
        with open(param_info_path, 'w') as param_file, open(results_path, 'w') as results_file:
            param_file.write(f"Grid Search Parameters for {model_type.capitalize()}:\n")
            results_file.write(f"Results for {model_type.capitalize()}:\n")

            for i, bin_name in enumerate(bin_cols):
                start_time = time.time()

                correlated_features = correlation_df[correlation_df["Bin Name"] == bin_name]

                if use_top_n_features:
                    selected_features = correlated_features[abs(correlated_features["Correlation"]) > 0.05].nlargest(top_n_features_count, 'Correlation')["Feature Name"]
                    selected_features = selected_features.tolist()
                elif use_specific_features:
                    selected_features = specific_features
                else:
                    selected_features = correlated_features[abs(correlated_features["Correlation"]) > 0.05]["Feature Name"]
                    if len(selected_features) < 10:
                        print(f"Too few features {len(selected_features)} selected, adding more features")
                        top_features = correlated_features.reindex(
                            correlated_features["Correlation"].abs().sort_values(ascending=False).index)["Feature Name"].head(5)
                        selected_features = top_features
                    selected_features = selected_features.tolist()

                print("Selected features are: ", selected_features)

                X_train = train_data[selected_features]
                y_train = train_data[bin_name]

                X_test = X_test_full[selected_features]
                y_test = pd.read_csv(args.test_y)[bin_name]

                r2, mse, etaRMS, y_pred, best_params, mean_train_score, param_grid = run_model_with_grid_search(model_type, X_train, y_train, X_test, y_test)
                
                end_time = time.time()
                elapsed_time = end_time - start_time

                # Save intermediate predictions
                bin_pred_path = os.path.join(model_dir, f"bin_{i}_prediction.csv")
                pd.DataFrame(y_pred, columns=[bin_name]).to_csv(bin_pred_path, index=False)

                result_str = (
                    f"Model: {model_type}, Bin {i}: R2 = {r2:.4f}, MSE = {mse:.4f}, etaRMS = {etaRMS:.4f} "
                    f"Best Params = {best_params}, Mean Training R²: {mean_train_score:.4f}, "
                    f"Time Taken: {elapsed_time:.2f} seconds\n"
                    f"No. of Features: {len(selected_features)}\n"
                )
                print(result_str)
                results_file.write(result_str)

                model_predictions[f'bin_{i}'] = y_pred

                param_file.write(f"\nBin: {bin_name}\nParameter Grid: {param_grid}\nBest Params: {best_params}\n")

                r2_scores.append(r2)

        # Plot R2 scores
        plot_r2_scores(r2_scores, bin_cols, model_type, model_dir)

        # Save model-specific predictions
        model_predictions_path = os.path.join(model_dir, "final_prediction.csv")
        model_predictions.to_csv(model_predictions_path, index=False)

if __name__ == "__main__":
    main()