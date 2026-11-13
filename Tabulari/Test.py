import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from dataset import set_seed, load_splits, make_loaders
from metrics_report import predict_proba, find_threshold, evaluate_and_save
from DeepMlp import DeepMLPClassifier


def find_checkpoint(weights_dir, model_name):
    """Cerca automaticamente il file dei pesi (.pt o .pth) del modello salvato."""
    search_dirs = [weights_dir, "weights", "weight", "checkpoints", "models", "."]
    patterns = [
        f"{model_name}.pt",
        f"{model_name}.pth",
        f"{model_name}_best.pt",
        f"{model_name}_best.pth",
        f"*{model_name}*.pt",
        f"*{model_name}*.pth",
    ]
    for folder in search_dirs:
        if not folder or not os.path.isdir(folder):
            continue
        for pat in patterns:
            matches = sorted(glob.glob(os.path.join(folder, pat)))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        f"Nessun file pesi trovato per '{model_name}' in '{weights_dir}' (né nelle cartelle standard)."
    )


def load_trained_mlp(ckpt_path, in_features, hidden_units, n_hidden, device):
    """Carica i pesi salvati (sia state_dict che intero modello nn.Module)."""
    model = DeepMLPClassifier(in_features=in_features, hidden_units=hidden_units, n_hidden=n_hidden)
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


def compute_binary_metrics(y_true, probs, threshold, model_name, n_features):
    preds = (probs >= threshold).astype(int)
    return {
        "Modello": model_name,
        "Feature_In": n_features,
        "Soglia_Val": threshold,
        "ROC_AUC": roc_auc_score(y_true, probs),
        "PR_AUC": average_precision_score(y_true, probs),
        "F1_Score": f1_score(y_true, preds, zero_division=0),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "Accuracy": accuracy_score(y_true, preds),
    }


def main():
    parser = argparse.ArgumentParser(description="Valutazione finale su Test Set dei 4 migliori DeepMLP")
    parser.add_argument("--csv", default="../data/dataset_balanced.csv")
    parser.add_argument("--weights_dir", default="../weight")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_recall", type=float, default=None)
    parser.add_argument("--output_csv", default="tabular_test_comparison.csv")
    args = parser.parse_args()

    csv_dataset = args.csv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device di valutazione: {device}\n")

    # Configurazione esatta dei 4 modelli migliori selezionati
    models_to_test = [
        {
            "name": "DeepMLP_h128x1_Dataset2b",
            "n_hidden": 1,
            "k_features": None,
            "csv": csv_dataset,
        },
        {
            "name": "DeepMLP_h128x1_k15_Dataset2b",
            "n_hidden": 1,
            "k_features": 15,
            "csv": csv_dataset,
        },
        {
            "name": "DeepMLP_h128x2_Dataset2b",
            "n_hidden": 2,
            "k_features": None,
            "csv": csv_dataset,
        },
        {
            "name": "DeepMLP_h128x2_k15_Dataset2b",
            "n_hidden": 2,
            "k_features": 15,
            "csv": csv_dataset,
        },
    ]

    summary_rows = []
    probs_dict = {}
    targets_dict = {}

    for cfg in models_to_test:
        name = cfg["name"]
        k_feat = cfg["k_features"]
        n_hid = cfg["n_hidden"]

        print(f"--- Valutazione: {name} ---")
        set_seed(args.seed)
        splits = load_splits(cfg["csv"], seed=args.seed)

        if k_feat is not None:
            selector = SelectKBest(score_func=f_classif, k=k_feat)
            splits["X_train"] = selector.fit_transform(splits["X_train"], splits["y_train"])
            splits["X_val"] = selector.transform(splits["X_val"])
            splits["X_test"] = selector.transform(splits["X_test"])

        _, val_loader, test_loader = make_loaders(splits, batch_size=args.batch)
        input_dim = splits["X_train"].shape[1]

        ckpt_path = find_checkpoint(args.weights_dir, name)
        print(f"[+] Pesi caricati da: {ckpt_path} (input_dim={input_dim})")

        model = load_trained_mlp(
            ckpt_path=ckpt_path,
            in_features=input_dim,
            hidden_units=128,
            n_hidden=n_hid,
            device=device,
        )

        # Soglia ottimale ricalcolata sul validation set esattamente come in DeepMlp.py
        y_val, p_val = predict_proba(model, val_loader, device)
        thr = find_threshold(y_val, p_val, min_recall=args.min_recall)

        # Valutazione finale sul Test Set
        y_test, p_test = predict_proba(model, test_loader, device)
        evaluate_and_save(model, test_loader, model_name=name, threshold=thr, device=device)

        summary_rows.append(compute_binary_metrics(y_test, p_test, thr, name, input_dim))
        probs_dict[name] = p_test
        targets_dict[name] = y_test

    # Se tutti e 4 i modelli sono stati valutati sullo stesso identico Test Set, calcola anche l'Ensemble
    first_target = next(iter(targets_dict.values()))
    if all(np.array_equal(first_target, t) for t in targets_dict.values()):
        p_ens = np.mean(list(probs_dict.values()), axis=0)
        mean_thr = float(np.mean([r["Soglia_Val"] for r in summary_rows]))
        summary_rows.append(
            compute_binary_metrics(first_target, p_ens, mean_thr, "Ensemble_4_DeepMLP", -1)
        )

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 95)
    print("RIEPILOGO COMPARATIVO TEST SET - MODELLI TABULARI (DeepMLP)")
    print("=" * 95)
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    df_summary.to_csv(args.output_csv, index=False, float_format="%.4f")
    print(f"\n[+] Tabella comparativa salvata in: {args.output_csv}")


if __name__ == "__main__":
    main()