import sys
import os

# Risale alla cartella padre 'Progetto' e la aggiunge ai percorsi noti
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split
from Tabulari.dataset import HeartDiseaseDataset
from Tabulari.train_utils import train_model
from Tabulari.metrics_report import evaluate_and_save

class HeartDiseaseMLP(nn.Module):
    def __init__(self, in_features, hidden_1=64, hidden_2=32):
        super(HeartDiseaseMLP, self).__init__()
        
        self.model = nn.Sequential(
            # Primo Hidden Layer
            nn.Linear(in_features, hidden_1),
            nn.ReLU(),
            
            # Secondo Hidden Layer
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            
            # Layer di Output
            nn.Linear(hidden_2, 1)
        )
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model_name = 'MLP_h64-32_best'
    csv_path = "data/heart_johnsmith88.csv"  
    
    dataset = HeartDiseaseDataset(csv_path)

    # Riproducibilità
    torch.manual_seed(42)

    # Split 80/20
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = dataset[0][0].shape[0]
    print(f"Dati caricati! Feature in ingresso: {input_dim}")

    print("Inizializzazione del MLP (64-32) con i parametri ottimali...")

    # Istanziamo il modello con la topologia vincente
    mlp_model = HeartDiseaseMLP(in_features=input_dim, hidden_1=64, hidden_2=32)

    print("Inizio dell'addestramento del MLP...")

    # Addestramento con LR e Weight Decay ottimali
    trained_mlp = train_model(
        model=mlp_model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        name_model=model_name,
        epochs=200, 
        lr=0.01,
        weight_decay=0.0001
    )

    print("Addestramento completato!")

    # Calcolo metriche e salvataggio report
    evaluate_and_save(trained_mlp, test_loader, model_name=model_name)

    # Salvataggio dei pesi definitivi
    os.makedirs('weight', exist_ok=True)
    percorso_pesi = f'weight/{model_name}.pth'
    torch.save(trained_mlp.state_dict(), percorso_pesi)
    print(f"Pesi del modello salvati in: {percorso_pesi}")