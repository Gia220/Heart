import torch
import numpy as np
import matplotlib.pyplot as plt

def evaluate_regression(model, test_loader, model_name):
    model.eval()
    all_preds = []
    all_targets = []
    
    # Calcolo delle predizioni su tutto il Test Set
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            all_preds.extend(outputs.numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 1. Calcolo Metriche Fisiche
    mae = np.mean(np.abs(all_preds - all_targets))
    mse = np.mean((all_preds - all_targets)**2)
    rmse = np.sqrt(mse)
    
    print(f"\n--- Report Regressione per {model_name} ---")
    print(f"MAE (Errore Assoluto Medio): {mae:.2f} battiti/minuto")
    print(f"MSE (Errore Quadratico Medio): {mse:.2f}")
    print(f"RMSE (Radice Errore Quadratico Medio): {rmse:.2f} battiti/minuto\n")
    
    # 2. Generazione Grafico: Reale vs Predetto
    # Se il modello è perfetto, tutti i punti giacciono sulla linea rossa
    plt.figure(figsize=(8, 6))
    plt.scatter(all_targets, all_preds, alpha=0.6, color='blue', edgecolors='k')
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--', lw=2, label='Predizione Ideale')
    
    plt.xlabel('Thalach Reale (BPM)')
    plt.ylabel('Thalach Predetto (BPM)')
    plt.title(f'Valori Reali vs Predetti ({model_name})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Salva il grafico
    filename = f"{model_name}_scatter.png"
    plt.savefig(filename)
    print(f"Grafico valutativo salvato come: {filename}")