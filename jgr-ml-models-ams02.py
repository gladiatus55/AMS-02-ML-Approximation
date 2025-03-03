# from sklearn.experimental import enable_halving_search_cv # noqa
# # now you can import normally from model_selection
# from sklearn.model_selection import HalvingGridSearchCV
# from sklearn.pipeline import make_pipeline
import math

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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge

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

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, hidden_dim=128):
        super(TransformerRegressor, self).__init__()
        self.transformer = nn.Transformer(
            d_model=input_dim, 
            nhead=num_heads, 
            num_encoder_layers=num_layers,
            dim_feedforward=hidden_dim
        )
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding sequence length dimension (batch, seq_len, features)
        out = self.transformer(x, x)  # Only using the encoder for simplicity
        out = self.fc(out[:, -1, :])  # Regression output
        return out.squeeze()

def run_transformer_model(X_train, y_train, X_test, y_test):
    # Convert data to tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the model
    input_dim = X_train.shape[1]
    model = TransformerRegressor(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(50):  # Number of epochs
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    y_test_np = np.array(y_test)
    RSS = 0
    etaRMS = 0
    avg_y_test = 0
    avg_y_test_n = 0

    for ij in range(len(y_test_np)):
        tem = ((y_test_np[ij] - y_pred[ij]) * (y_test_np[ij] - y_pred[ij]))
        RSS = RSS + tem
        etaRMS = etaRMS + (tem / (y_test_np[ij] * y_test_np[ij]))
        avg_y_test = avg_y_test + y_test_np[ij]
        avg_y_test_n = avg_y_test_n + 1

    etaRMS = etaRMS / avg_y_test_n
    etaRMS = 100. * math.sqrt(etaRMS)

    print(f"Transformer R² Score on Test Data: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}, etaRMS: {etaRMS:.4f}")
    
    return r2, mse, etaRMS, y_pred, {}, None, None


# Custom Transformer to Extract Embeddings from a Transformer Model
class TransformerEmbeddingExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-uncased', max_length=32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert numerical data to string representations
        text_data = [" ".join(map(str, row)) for row in X]
        # Tokenize the text data
        inputs = self.tokenizer(text_data, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the [CLS] token representation (or mean of all tokens)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # (batch_size, hidden_size)
        return cls_embeddings

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
    elif model_type == 'transformer':
        # r2, mse, y_pred, best_params, mean_train_score, param_grid = run_transformer_model(X_train, y_train, X_test, y_test)
        pipeline = Pipeline([
            ('transformer', TransformerEmbeddingExtractor()),  # Use Transformer to extract embeddings
            ('regressor', Ridge(alpha=1.0))
        ])
        param_grid = None
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
        print('aaaa')
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
    use_top_n_features = False
    top_n_features_count = 1

    use_specific_features = False
    specific_features = ['Phi']

    # Create output directory
    output_dir = create_output_directory()

    # Load the training data
    train_data = pd.read_csv("train.csv")
    
    # Load the test data and timestamp
    X_test_full = pd.read_csv("X_test.csv")
    timeStamp = X_test_full['timeStamp'].copy()

    # Feature and bin columns
    feature_cols = train_data.columns[1:-30]
    bin_cols = train_data.columns[-30:]

    # Compute correlations
    correlation_df = compute_correlations(train_data, feature_cols, bin_cols)

    # Save correlation results
    correlation_path = os.path.join(output_dir, "correlation_results.csv")
    correlation_df.to_csv(correlation_path, index=False)

    model_types = ['xgboost', 'linear','random_forest', 'svr', 'sgdr', 'transformer']	
    
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
                y_test = pd.read_csv("y_test.csv")[bin_name]

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