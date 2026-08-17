
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split
from dataset import HeartDiseaseDataset
from train_utils import train_model
from metrics_report import evaluate_and_save


class HeartDiseaseMLP(nn.Module):
    def __init__(self, in_features, hidden_1=64, hidden_2=32):
        """
        Costruisce un classificatore MLP profondo.
        Input:
        in_features: numero di feature in input (13)
        hidden_1: unità nel primo livello nascosto
        hidden_2: unità nel secondo livello nascosto
        """
        super(HeartDiseaseMLP, self).__init__()
        
        # Definiamo la struttura in cascata
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
        # Il dato attraversa in sequenza tutti i layer[cite: 4]
        return self.model(x)

if __name__ == "__main__":

# Definiamo i parametri e carichiamo i dati
    csv_path = "data/heart_johnsmith88.csv"  # Assicurati che il path sia corretto
    dataset = HeartDiseaseDataset(csv_path)

    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Dati caricati! Feature in ingresso: {dataset[0][0].shape[0]}")

    model_name = 'MLP'


    input_dim = dataset[0][0].shape[0]

    print(f"Inizializzazione del MLP con {input_dim} feature in ingresso...")
    # Istanziamo il nostro MLP
    mlp_model = HeartDiseaseMLP(in_features=input_dim, hidden_1=64, hidden_2=32)

    print("Inizio dell'addestramento del MLP...")

    # Richiamiamo la tua funzione aggiornata (assumendo che l'hai importata da train_utils)
    # Sperimentiamo con un learning rate leggermente più basso e 150 epoche
    trained_mlp = train_model(
        model=mlp_model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        name_model=model_name,
        epochs=150, 
        lr=0.005,
        
    )

    print("Addestramento completato!")


    evaluate_and_save(trained_mlp, test_loader, model_name=model_name)