import torch
import numpy as np
from torch.utils.data import random_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
import joblib  
import pandas as pd
import os
from torch.utils.data import DataLoader
from Classificatore.dataset import HeartDiseaseDataset

def extract_arrays_from_subset(subset):
    X_list, y_list = [], []
    for x, y in subset:
        X_list.append(x.numpy())
        y_list.append(y.numpy())
    return np.array(X_list), np.array(y_list)

if __name__ == "__main__":

    model_name = 'random_forest'

    csv_path = "data/heart_johnsmith88_mod.csv" 
    dataset = HeartDiseaseDataset(csv_path)

    input_dim = dataset[0][0].shape[0]
    print(f"Dati caricati! Feature in ingresso: {input_dim}")

    torch.manual_seed(42)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print("Estrazione delle feature per Scikit-Learn...")
    X_train, y_train = extract_arrays_from_subset(train_dataset)
    X_test, y_test = extract_arrays_from_subset(test_dataset)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    print("Inizializzazione del RandomForestClassifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,        
        max_depth=None,          
        random_state=42,         
        n_jobs=-1,               
        class_weight='balanced'
    )

    print("Inizio dell'addestramento...")
    rf_model.fit(X_train, y_train)
    print("Addestramento completato con successo!")

    print("\nValutazione sul Test Set ")
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}\n")
    print("Report Completo:")
    print(classification_report(y_test, y_pred))

    print("\n--- Feature Importance ---")
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    if len(feature_names) == input_dim:
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print(importance_df.head(5))

    #salvataggio pesi
    os.makedirs('weight', exist_ok=True)
    save_path = f'weight/{model_name}.pkl'
    joblib.dump(rf_model, save_path)
    print(f"Modello salvato in: {save_path}")