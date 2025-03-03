import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data = pd.read_csv('train.csv')

X = data.drop(columns=[f'bin_{i}' for i in range(30)])
y = data[[f'bin_{i}' for i in range(30)]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class TransformerRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, num_layers=2, hidden_dim=64):
        super(TransformerRegressionModel, self).__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, src):
        x = self.embedding(src)
        x = self.relu(self.dropout(x))
        x = self.transformer_encoder(x.unsqueeze(1))
        x = x.mean(dim=1)
        x = self.fc_out(x)
        return x

input_dim = X_train_scaled.shape[1]
output_dim = y_train.shape[1]

model = TransformerRegressionModel(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(20):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')

model.eval()
with torch.no_grad():
    total_loss = 0
    y_true = []
    y_pred = []
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item()
        
        y_true.append(y_batch.numpy())
        y_pred.append(outputs.numpy())
    
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    print(f'Average Test Loss: {total_loss / len(test_loader):.4f}')
    print(f'R2 Score: {r2_score:.4f}')