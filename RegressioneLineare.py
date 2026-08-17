from torch import nn
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from train_utils import train_model
import torch
from torch.utils.data import DataLoader, random_split
from dataset import HeartDiseaseDataset
from metrics_report import evaluate_and_save

class LogisticRegressor(nn.Module):
    def __init__(self, in_features):
        """
        Input:
        in_features: numero di feature in input (es. 13 per il nostro dataset)
        """
        # Richiamiamo il costruttore della superclasse
        super(LogisticRegressor, self).__init__()
        
        # Definiamo la trasformazione lineare: y = Ax + b
        self.linear = nn.Linear(in_features, 1)
        
    def forward(self, x):
        # Calcoliamo e restituiamo i logit
        logits = self.linear(x)
        return logits


if __name__ == "__main__":
    # Definiamo i parametri e carichiamo i dati
    csv_path = "data/heart.csv"  # Assicurati che il path sia corretto
    dataset = HeartDiseaseDataset(csv_path)

    print(f"Dati caricati! Feature in ingresso: {dataset[0][0].shape[0]}")

    model_name = 'linear_regressor'

    # Ricaviamo dinamicamente il numero di feature (es. 13) leggendo il primo campione
    input_dim = dataset[0][0].shape[0] 

    torch.manual_seed(42)
    # Prepariamo lo split e i DataLoader
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Istanziamo il modello base
    print(f"Inizializzazione del LogisticRegressor con {input_dim} feature in ingresso...")
    model = LogisticRegressor(in_features=input_dim)

    # Facciamo partire l'addestramento
    print("Inizio dell'addestramento...")
    trained_model = train_model(
        model=model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        name_model=model_name,
        epochs=150, 
        lr=0.01
        
    )
    print("Addestramento completato con successo!")

    evaluate_and_save(trained_model, test_loader, model_name=model_name)