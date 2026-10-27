# metrics_report.py
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, average_precision_score,
                             precision_recall_curve)


def predict_proba(model, loader, device="cpu"):
    """Restituisce (y_true, p_positiva) per modelli PyTorch o scikit-learn."""
    
    ys, ps = [], []
    if isinstance(model, torch.nn.Module):
        model.eval().to(device)
        with torch.no_grad():
            for X, y in loader:
                ps.append(torch.sigmoid(model(X.to(device))).cpu())
                ys.append(y)
        return torch.cat(ys).numpy().ravel(), torch.cat(ps).numpy().ravel()
    for X, y in loader:
        ps.append(model.predict_proba(X.numpy())[:, 1])
        ys.append(y.numpy().ravel())
    return np.concatenate(ys), np.concatenate(ps)


def find_threshold(y_true, probs, min_recall=None):
    """
    Sceglie la soglia sul VALIDATION.
    - min_recall=None  -> massimizza l'F1
    - min_recall=0.8   -> massima precision con recall >= 0.8
    """
    prec, rec, thr = precision_recall_curve(y_true, probs)
    prec, rec = prec[:-1], rec[:-1]
    if min_recall is None:
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        return float(thr[np.argmax(f1)])
    ok = rec >= min_recall
    if not ok.any():
        return float(thr[np.argmax(rec)])
    return float(thr[np.where(ok)[0][np.argmax(prec[ok])]])


def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "F1_Score": f1_score(y_true, preds, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, probs),
        "PR_AUC": average_precision_score(y_true, probs),
    }, confusion_matrix(y_true, preds)


def evaluate_and_save(model, loader, model_name="Modello", threshold=0.5, results_file="results.csv", device="cpu"):

    """ stampa il report e accoda a results.csv."""
    y_true, probs = predict_proba(model, loader, device)
    m, cm = compute_metrics(y_true, probs, threshold)

    print(f"\n--- Report: {model_name} (soglia {threshold:.3f}) ---")
    for k, v in m.items():
        print(f"{k:<10} {v:.4f}")
    print(f"Matrice di confusione:\n{cm}\n")

    row = pd.DataFrame([{"Model": model_name, **m}])
    row.to_csv(results_file, mode="a", header=not os.path.isfile(results_file), index=False)
    print(f"Risultati salvati in {results_file}")
    return m