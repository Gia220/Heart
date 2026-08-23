import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from dataset_reg import HeartDiseaseRegressionDataset
from train_utils_reg import train_regression_model
from metrics_report_reg import evaluate_regression

class MLPRegressor(nn.Module):
    def __init__(self, in_features):
        super(MLPRegressor, self).__init__()
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
    
    # ---------------------------------------------------------
    # CENTRO DI CONTROLLO: Cambia qui per predire ciò che vuoi!
    target_param = 'oldpeak'
    unit_param = 'mm'
    # ---------------------------------------------------------
    
    dataset = HeartDiseaseRegressionDataset(csv_path, target_col=target_param)

    input_dim = dataset[0][0].shape[0] 
    print(f"Dati caricati! Feature in ingresso: {input_dim}")

    # Il nome del modello cambierà in base a cosa stiamo predicendo
    model_name = f'mlp_regressor_{target_param}'

    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Inizializzazione del MLPRegressor per predire '{target_param}'...")
    model = MLPRegressor(in_features=input_dim)

    print("Inizio dell'addestramento...")
    trained_model = train_regression_model(
        model=model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        name_model=model_name,
        epochs=300, 
        lr=0.005  # Learning Rate ottimizzato per target in piccola scala
    )
    print("Addestramento completato con successo!")

    evaluate_regression(
        trained_model, 
        test_loader, 
        model_name=model_name, 
        target_name=target_param, 
        unit=unit_param
    )