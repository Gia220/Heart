import torch
import numpy as np
from torch.utils.data import random_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
import joblib  # Per salvare i modelli di scikit-learn
import pandas as pd
import os
from Classificatore.metrics_report import evaluate_and_save
from torch.utils.data import DataLoader

from Classificatore.dataset import HeartDiseaseDataset

def extract_arrays_from_subset(subset):
    """
    Funzione di utilità per estrarre numpy arrays (X, y) da un PyTorch Subset.
    Necessaria per interfacciare i dati di PyTorch con Scikit-Learn.
    """
    X_list, y_list = [], []
    for x, y in subset:
        X_list.append(x.numpy())
        y_list.append(y.numpy())
    return np.array(X_list), np.array(y_list)


if __name__ == "__main__":
    # 1. Definiamo i parametri e carichiamo i dati (stessa interfaccia PyTorch)
    csv_path = "data/heart_johnsmith88_mod.csv"  # Utilizziamo la baseline da 1000 righe e 13 feature
    dataset = HeartDiseaseDataset(csv_path)

    input_dim = dataset[0][0].shape[0]
    print(f"Dati caricati! Feature in ingresso: {input_dim}")

    model_name = 'random_forest_baseline'

    # 2. Riproducibilità rigorosa: Stesso split di train/test del LogisticRegressor
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. Conversione dei subset PyTorch in tensori Numpy per Scikit-Learn
    print("Estrazione delle feature per Scikit-Learn...")
    X_train, y_train = extract_arrays_from_subset(train_dataset)
    X_test, y_test = extract_arrays_from_subset(test_dataset)

    # Assicuriamoci che i target siano monodimensionali (da (N, 1) a (N,))
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # 4. Inizializzazione del modello Random Forest
    print("Inizializzazione del RandomForestClassifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,        
        max_depth=None,          
        random_state=42,         # Congeliamo la stocasticità degli alberi
        n_jobs=-1,               # Sfrutta tutti i core della CPU
        class_weight='balanced'
    )

    # 5. Addestramento (equivale a train_model, ma istantaneo)
    print("Inizio dell'addestramento...")
    rf_model.fit(X_train, y_train)
    print("Addestramento completato con successo!")

    # 6. Valutazione e Salvataggio (l'equivalente del tuo evaluate_and_save)
    print("\n--- Valutazione sul Test Set ---")
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}\n")
    print("Report Completo:")
    print(classification_report(y_test, y_pred))

    # Salvataggio del modello addestrato (in formato .pkl invece che .pth)
    os.makedirs("saved_models", exist_ok=True)
    save_path = f"saved_models/{model_name}.pkl"
    joblib.dump(rf_model, save_path)
    print(f"Modello salvato in: {save_path}")

    # 7. Bonus Accademico: Estrazione dell'importanza delle feature[cite: 1]
    print("\n--- Feature Importance ---")
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Se il tuo dataset ha feature diverse, adatta la lista feature_names
    if len(feature_names) == input_dim:
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print(importance_df.head(5))

    evaluate_and_save(rf_model, test_loader, model_name="Random_Forest")