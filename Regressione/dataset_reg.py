import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class HeartDiseaseRegressionDataset(Dataset):
    def __init__(self, csv_path):
        # 1. Carica il dataset
        self.df = pd.read_csv(csv_path)
        
        # --- NOVITÀ: DATA CLEANING AUTOMATICO ---
        # Forza la conversione in numero. Se trova lettere (es. '4he'), le trasforma in NaN
        self.df = self.df.apply(pd.to_numeric, errors='coerce')
        # Elimina tutte le righe che contengono NaN (cioè i record corrotti)
        self.df = self.df.dropna()
        # ----------------------------------------
        
        # 2. Identifichiamo il nuovo target (thalach: frequenza cardiaca massima)
        target_col = 'thalach'
        
        # Le features sono tutte tranne il thalach
        self.X = self.df.drop(columns=[target_col]).values
        self.y = self.df[target_col].values
        
        # 3. Standardizzazione (cruciale per la regressione lineare)
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        
        # 4. Convertiamo in tensori PyTorch (entrambi float32 per la regressione)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]