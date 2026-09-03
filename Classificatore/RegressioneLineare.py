from torch import nn
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, random_split
from Classificatore.dataset import HeartDiseaseDataset
from Classificatore.metrics_report import evaluate_and_save
from Classificatore.train_utils import train_model
import os

class LogisticRegressor(nn.Module):
    def __init__(self, in_features):

        super(LogisticRegressor, self).__init__()
        
        self.linear = nn.Linear(in_features, 1)
        
    def forward(self, x):
        
        logits = self.linear(x)
        return logits


if __name__ == "__main__":

    model_name = 'linear_regressor'

    
    csv_path = "data/heart_johnsmith88_mod.csv"  #path dataset
    dataset = HeartDiseaseDataset(csv_path)

    input_dim = dataset[0][0].shape[0] 
    print(f"Dati caricati! Feature in ingresso: {input_dim}")


    torch.manual_seed(42)
    # train_set 80%         test_set 20%
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    
    print(f"Inizializzazione del LogisticRegressor con {input_dim} feature in ingresso...")
    model = LogisticRegressor(in_features=input_dim)

    #Inizio l'addestramento
    print("Inizio dell'addestramento")
    trained_model = train_model(
        model=model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        name_model=model_name,
        epochs=150, 
        lr=0.07
        
    )
    print("Addestramento completato con successo!")

    #calcolo e salvataggio delle metriche
    evaluate_and_save(trained_model, test_loader, model_name=model_name)

    #salvataggi pesi
    os.makedirs('weight', exist_ok=True)
    percorso_pesi = f'weight/{model_name}.pth'
    torch.save(trained_model.state_dict(), percorso_pesi)
    print(f"Pesi del modello salvati in: {percorso_pesi}")