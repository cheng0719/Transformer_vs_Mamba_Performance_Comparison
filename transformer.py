import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import datetime

# Time2Vec 模型
class Time2Vec(nn.Module):
    def __init__(self, in_features, out_features):
        super(Time2Vec, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Parameter(torch.Tensor(in_features, out_features))
        self.b = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.w, 0, 2*np.pi)
        nn.init.uniform_(self.b, 0, 2*np.pi)
    
    def forward(self, x):
        x = x.unsqueeze(-1)  # 增加一個維度，使得可以和權重矩陣進行乘法
        sin_transform = torch.sin(x @ self.w + self.b)
        return sin_transform

# 資料預處理
class StockDataset(Dataset):
    def __init__(self, data_path, seq_length):
        self.seq_length = seq_length
        self.data = pd.read_csv(data_path)
        self.scaler = MinMaxScaler()
        self.data[['high', 'low', 'open', 'close']] = self.scaler.fit_transform(self.data[['high', 'low', 'open', 'close']])
        self.time_encoder = Time2Vec(in_features=1, out_features=4)  # 使用 Time2Vec 進行時間編碼
    
    def __len__(self):
        return len(self.data) - self.seq_length - 1
    
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.seq_length
        input_data = torch.tensor(self.data.iloc[start_idx:end_idx][['high', 'low', 'open', 'close']].values, dtype=torch.float32)
        dates = self.data.iloc[start_idx:end_idx]['date'].values
        date_encodings = self.time_encoder(torch.tensor(dates, dtype=torch.float32))
        input_data = torch.cat([input_data, date_encodings], dim=1)  # 將時間編碼加入到輸入資料中
        target_data = torch.tensor(self.data.iloc[end_idx][['high', 'low', 'open', 'close']].values, dtype=torch.float32)
        return input_data, target_data

# 更新 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, seq_length):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.seq_length = seq_length
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.decoder = nn.Linear(input_size * seq_length, 4)  # 4個輸出特徵：最高價、最低價、開盤價、收盤價
    
    def forward(self, x):
        x = self.transformer_encoder(x.permute(1, 0, 2))  # 將序列維度移到第二維
        x = x.view(-1, self.input_size * self.seq_length)  # 將序列展平
        x = self.decoder(x)
        return x

# 設置資料路徑和超參數
data_path = "your_data_path.csv"
seq_length = 30
input_size = 8  # 特徵數量：最高價、最低價、開盤價、收盤價、時間編碼
batch_size = 64
lr = 0.001
num_epochs = 10

# 載入資料集和模型
dataset = StockDataset(data_path, seq_length)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = TransformerModel(input_size, seq_length)

# 定義損失函數和優化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 訓練模型
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
