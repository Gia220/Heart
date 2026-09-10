# dataset.py
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class HeartDiseaseDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        
        # Separiamo feature e target
        X_raw = self.data.iloc[:, :-1].values
        Y_raw = self.data.iloc[:, -1].values
        
        # Normalizzazione
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X_raw)
        
        # Conversione in tensori
        self.X = torch.tensor(X_norm, dtype=torch.float32)
        self.Y = torch.tensor(Y_raw, dtype=torch.float32).view(-1, 1)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]