
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split
from Classificatore.dataset import HeartDiseaseDataset
from Classificatore.train_utils import train_model
from Classificatore.metrics_report import evaluate_and_save
import os


class HeartDiseaseMLP(nn.Module):
    def __init__(self, in_features, hidden_1=64, hidden_2=32):
        """
        Input:
        in_features: numero di feature in input (13)
        hidden_1: unità nel primo livello nascosto
        hidden_2: unità nel secondo livello nascosto
        """
        super(HeartDiseaseMLP, self).__init__()
        
        
        self.model = nn.Sequential(
            # Primo Hidden Layer
            nn.Linear(in_features, hidden_1),
            nn.ReLU(),
            nn.Dropout(p=0.5), # Spegne casualmente il 50% dei neuroni in fase di training
            
            # Secondo Hidden Layer
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            # Layer di Output
            nn.Linear(hidden_2, 1)
        )
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model_name = 'MLP'
    # Definiamo i parametri e carichiamo i dati
    csv_path = "data/heart_johnsmith88.csv"  
    dataset = HeartDiseaseDataset(csv_path)

    torch.manual_seed(42)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    input_dim = dataset[0][0].shape[0]
    print(f"Dati caricati! Feature in ingresso: {input_dim}")


    print(f"Inizializzazione del MLP con {input_dim} feature in ingresso...")

    # Istanziamo il modello
    mlp_model = HeartDiseaseMLP(in_features=input_dim, hidden_1=64, hidden_2=32)

    print("Inizio dell'addestramento del MLP...")


    trained_mlp = train_model(
        model=mlp_model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        name_model=model_name,
        epochs=150, 
        lr=0.005,
        
    )

    print("Addestramento completato!")

    #calcolo metriche e save
    evaluate_and_save(trained_mlp, test_loader, model_name=model_name)


    #salvataggio pesi
    os.makedirs('weight', exist_ok=True)
    percorso_pesi = f'weight/{model_name}.pth'
    torch.save(trained_mlp.state_dict(), percorso_pesi)
    print(f"Pesi del modello salvati in: {percorso_pesi}")