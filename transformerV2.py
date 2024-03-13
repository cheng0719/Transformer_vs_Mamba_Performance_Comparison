import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 資料預處理
class StockDataset(Dataset):
    def __init__(self, data_path, seq_length):
        self.seq_length = seq_length
        self.data = pd.read_csv(data_path)
        self.scaler = MinMaxScaler()
        self.data[['high', 'low', 'open', 'close']] = self.scaler.fit_transform(self.data[['high', 'low', 'open', 'close']])
    
    def __len__(self):
        return len(self.data) - self.seq_length - 1
    
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.seq_length
        input_data = torch.tensor(self.data.iloc[start_idx:end_idx][['high', 'low', 'open', 'close']].values, dtype=torch.float32)
        target_data = torch.tensor(self.data.iloc[end_idx][['high', 'low', 'open', 'close']].values, dtype=torch.float32)
        return input_data, target_data

# Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, seq_length):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.seq_length = seq_length
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.decoder = nn.Linear(input_size * seq_length, 4)  # 4個輸出特徵：最高價、最低價、開盤價、收盤價
    
    def forward(self, x):
        x = self.transformer_encoder(x.permute(1, 0, 2))  # 將序列維度移到第二維
        x = x.view(-1, self.input_size * self.seq_length)  # 將序列展平
        x = self.decoder(x)
        return x

# 定義函數以進行模型推論
def predict(model, data):
    model.eval()
    with torch.no_grad():
        inputs, targets = data
        outputs = model(inputs)
        return outputs

# 定義函數以載入 CSV 檔案並進行推論
def predict_from_csv(model, csv_path, seq_length):
    dataset = StockDataset(csv_path, seq_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions = []
    actuals = []
    for idx, (inputs, targets) in enumerate(dataloader):
        if idx < 30:  # 跳過前 30 筆資料
            continue
        outputs = predict(model, (inputs, targets))
        predictions.append(outputs.squeeze().detach().numpy())
        actuals.append(targets.numpy())
    return predictions, actuals, outputs

# 計算模型的準確度
def calculate_accuracy(predicted, actual):
    # use root-mean-square error (RMSE) as the accuracy metric and return the calculated accuracy
    return np.sqrt(np.mean((predicted - actual) ** 2))

def inference(model, data_path, seq_length):
    # 進行推論並計算準確度
    predictions, actuals, outputs = predict_from_csv(model, data_path, seq_length)
    total_accuracy = 0
    for predicted_prices, actual_prices in zip(predictions, actuals):
        accuracy = calculate_accuracy(predicted_prices, actual_prices)
        total_accuracy += accuracy
    average_accuracy = total_accuracy / len(predictions)
    # print(f"Average Accuracy: {average_accuracy}")
    return average_accuracy


# 訓練函數
def train_model(model, train_loader, optimizer, criterion):
    i = 1
    while(1):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_acc = inference(model, training_data_path, seq_length)
        val_acc = inference(model, validation_data_path, seq_length)
        print(f"Epoch {i}, Loss: {total_loss}, Train Acc: {train_acc}, Val Acc: {val_acc}")
        if(i > 10 and val_acc < train_acc):
            break
        i += 1

# 設置資料路徑和超參數
training_data_path = "./tsmc_stock_prices_training_INT.csv"
validation_data_path = "./tsmc_stock_prices_validation_INT.csv"
testing_data_path = "./tsmc_stock_prices_testing_INT.csv"
seq_length = 30
input_size = 4  # 特徵數量：最高價、最低價、開盤價、收盤價
batch_size = 64
lr = 0.001
# num_epochs = 10

# 載入資料集和模型
dataset = StockDataset(training_data_path, seq_length)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = TransformerModel(input_size, seq_length)

# 定義損失函數和優化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 訓練模型
train_model(model, train_loader, optimizer, criterion)


### Inferencing ###
print(inference(model, testing_data_path, seq_length))