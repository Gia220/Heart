import torch
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_and_save(model, test_loader, model_name="Modello", results_file="results.csv"):
    """
    Valuta il modello sul test set, stampa il report e salva le metriche in un CSV.
    Supporta sia modelli PyTorch che modelli Scikit-Learn.
    """
    all_preds = []
    all_targets = []
    
    # 1. CONTROLLO TIPO DI MODELLO
    if isinstance(model, torch.nn.Module):
        # Percorso PyTorch (Regressione Logistica, MLP, ecc.)
        model.eval()
        with torch.no_grad():
            for X_test, Y_test in test_loader:
                outputs = model(X_test)
                # Applichiamo la sigmoide e sogliamo a 0.5 per la classificazione binaria
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.numpy())
                all_targets.extend(Y_test.numpy())
                
    else:
        # Percorso Scikit-Learn (Random Forest, SVM classica, ecc.)
        for X_test, Y_test in test_loader:
            # I modelli di ML classico richiedono array NumPy e usano .predict()
            preds = model.predict(X_test.numpy())
            
            all_preds.extend(preds)
            all_targets.extend(Y_test.numpy())
            
    # 2. CALCOLO DELLE METRICHE (comune a tutti i modelli)
    # zero_division=0 evita warning se il modello predice sempre la stessa classe
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    # 3. STAMPA DEL REPORT TESTUALE
    print(f"\n--- Report: {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Matrice di Confusione:\n{cm}\n")
    
    # 4. SALVATAGGIO STRUTTURATO NEL CSV
    new_data = pd.DataFrame({
        "Model": [model_name],
        "Accuracy": [acc],
        "Precision": [prec],
        "Recall": [rec],
        "F1_Score": [f1]
    })
    
    # Se il file esiste già, accodiamo i dati (mode='a') senza riscrivere l'header
    if os.path.isfile(results_file):
        new_data.to_csv(results_file, mode='a', header=False, index=False)
    else:
        # Se è la prima volta, creiamo il file con l'header
        new_data.to_csv(results_file, index=False)
        
    print(f"Risultati di {model_name} salvati in {results_file}")