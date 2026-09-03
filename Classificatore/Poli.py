import os
import torch
import joblib
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Assicurati che questi import puntino correttamente ai tuoi file
from Classificatore.dataset import HeartDiseaseDataset
from Classificatore.train_utils import train_model
from Classificatore.metrics_report import evaluate_and_save
from Classificatore.RegressioneLineare import LogisticRegressor

def main():

    dataset = HeartDiseaseDataset("data/heart_johnsmith88_mod.csv")

    #Estraiamo feature e target 
    X_raw = dataset.X.numpy()
    Y_raw = dataset.Y

    #Trasformazione Polinomiale e Normalizzazione
    print("Creazione feature polinomiali in corso...")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_np = poly.fit_transform(X_raw)

    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly_np)

    #Creiamo un nuovo Dataset per PyTorch
    X_poly_tensor = torch.tensor(X_poly_scaled, dtype=torch.float32)
    poly_dataset = TensorDataset(X_poly_tensor, Y_raw)

    poly_dim = X_poly_tensor.shape[1]
    print(f"Feature passate da {dataset.X.shape[1]} a {poly_dim}")

    # Split e DataLoaders classici
    torch.manual_seed(42) 
    train_size = int(0.8 * len(poly_dataset))
    test_size = len(poly_dataset) - train_size
    train_dataset, test_dataset = random_split(poly_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #cartella dei pesi
    os.makedirs('weight', exist_ok=True)

    #PRIMO RUN: Senza Weight Decay
    print("\n--- Addestramento Modello Standard ---")
    poly_model_1 = LogisticRegressor(in_features=poly_dim)
    trained_poly_1 = train_model(poly_model_1, train_loader, test_loader, epochs=250, lr=0.005, name_model="Poly_Reg")
    evaluate_and_save(trained_poly_1, test_loader, model_name="Poly_Reg")
    
    # Salvataggio pesi modello 1
    torch.save(trained_poly_1.state_dict(), 'weight/poly_regressor.pth')
    print("Pesi salvati in: weight/poly_regressor.pth")
    """
    # SECONDO RUN: Con Weight Decay 
    print("\n--- Addestramento Modello Regolarizzato ---")
    poly_model_2 = LogisticRegressor(in_features=poly_dim) 
    trained_poly_2 = train_model(poly_model_2, train_loader, test_loader, epochs=150, lr=0.01, name_model="Poly_Reg_decay_2", weight_decay=1e-2)
    evaluate_and_save(trained_poly_2, test_loader, model_name="Poly_Reg_decay_2")
    
    # Salvataggio pesi modello 2
    torch.save(trained_poly_2.state_dict(), 'weight/poly_regressor_decay.pth')
    print("Pesi salvati in: weight/poly_regressor_decay.pth")"""
    # SALVATAGGIO pesi 
    joblib.dump(poly, 'weight/poly_transform.pkl')
    joblib.dump(scaler, 'weight/poly_scaler.pkl')
    print("Oggetti PolynomialFeatures e StandardScaler salvati correttamente in 'weight/'.")

if __name__ == "__main__":
    main()