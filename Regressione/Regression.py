import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from dataset_reg import HeartDiseaseRegressionDataset # (Vedi nota sotto sul dataset)
from train_utils_reg import train_regression_model    # (Vedi nota sotto sul training)
from metrics_report_reg import evaluate_regression

class LinearRegressor(nn.Module):
    def __init__(self, in_features):
        """
        Input:
        in_features: 13 (tutte le feature originali TRANNE il thalach, ma includendo 
                     magari il vecchio target della malattia come input!)
        """
        super(LinearRegressor, self).__init__()
        
        # Trasformazione lineare pura
        self.linear = nn.Linear(in_features, 1)
        
    def forward(self, x):
        # NESSUNA FUNZIONE DI ATTIVAZIONE (niente Sigmoide o Softmax).
        # Vogliamo che il modello possa sputare numeri da 70 a 200 (i battiti).
        return self.linear(x)

class MLPRegressor(nn.Module):
    def __init__(self, in_features):
        super(MLPRegressor, self).__init__()
        # Creiamo una vera rete neurale con 2 hidden layer
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    csv_path = "data/heart_johnsmith88_mod.csv"
    
    # Usiamo un dataset modificato che restituisce 'thalach' come y
    dataset = HeartDiseaseRegressionDataset(csv_path)

    input_dim = dataset[0][0].shape[0] 
    print(f"Dati caricati! Feature in ingresso: {input_dim}")

    model_name = 'mlp_regressor_thalach'

    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Inizializzazione del LinearRegressor con {input_dim} feature...")
    #model = LinearRegressor(in_features=input_dim)
    model = MLPRegressor(in_features=input_dim)


    print(f"Inizializzazione del MLPRegressor con {input_dim} feature...")

    print("Inizio dell'addestramento...")
    trained_model = train_regression_model(
        model=model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        name_model=model_name,
        epochs=300,        # Aumentate per dare tempo di convergere
        lr=0.1             # Decuplicato per fare passi più grandi verso il 150
    )
    print("Addestramento completato con successo!")

    evaluate_regression(trained_model, test_loader, model_name=model_name)

    

