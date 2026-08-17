import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from dataset import HeartDiseaseDataset
from train_utils import train_model
from metrics_report import evaluate_and_save
from RegressioneLineare import LogisticRegressor

if __name__ == "__main__":

    # 1. Carichiamo i dati con la nostra classe originale intatta
    dataset = HeartDiseaseDataset("data/heart_johnsmith88_mod.csv")

    # 2. Estraiamo feature e target per manipolarli nel notebook
    X_raw = dataset.X.numpy()
    Y_raw = dataset.Y

    # 3. Trasformazione Polinomiale
    print("Creazione feature polinomiali in corso...")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_np = poly.fit_transform(X_raw)

    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly_np)

    # 4. Creiamo un nuovo Dataset "al volo" per PyTorch
    X_poly_tensor = torch.tensor(X_poly_scaled, dtype=torch.float32)
    poly_dataset = TensorDataset(X_poly_tensor, Y_raw)

    poly_dim = X_poly_tensor.shape[1]
    print(f"Feature passate da {dataset.X.shape[1]} a {poly_dim}")

    # 5. Split e DataLoaders classici
    torch.manual_seed(42) 
    train_size = int(0.8 * len(poly_dataset))
    test_size = len(poly_dataset) - train_size
    train_dataset, test_dataset = random_split(poly_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 6. Addestramento usando la TUA funzione intatta!


    # --- PRIMO RUN: Senza Weight Decay ---
  
    print("\n--- Addestramento Modello Standard ---")
    poly_model_1 = LogisticRegressor(in_features=poly_dim)
    trained_poly_1 = train_model(poly_model_1, train_loader, test_loader, epochs=150, lr=0.01, name_model="Poly_Reg")
    evaluate_and_save(trained_poly_1, test_loader, model_name="Poly_Reg")

    # --- SECONDO RUN: Con Weight Decay ---
    print("\n--- Addestramento Modello Regolarizzato ---")
    # FONDAMENTALE: Creare una nuova istanza pulita!
    poly_model_2 = LogisticRegressor(in_features=poly_dim) 
    trained_poly_2 = train_model(poly_model_2, train_loader, test_loader, epochs=150, lr=0.01, name_model="Poly_Reg_decay_5", weight_decay=1e-5)
    evaluate_and_save(trained_poly_2, test_loader, model_name="Poly_Reg_decay_5")

