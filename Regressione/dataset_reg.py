import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class HeartDiseaseRegressionDataset(Dataset):
    # Ora prende target_col come parametro (di default 'oldpeak')
    def __init__(self, csv_path, target_col='oldpeak'):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.apply(pd.to_numeric, errors='coerce')
        self.df = self.df.dropna()
        
        self.target_col = target_col
        
        self.X = self.df.drop(columns=[self.target_col]).values
        self.y = self.df[self.target_col].values
        
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]