import os


import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from dataset_ptbxl import prepare_ptbxl_5class_data, PTBXLDataset, SUPERCLASSES
from transformers import ECGTransformer
from CNN import PTBXLImageDataset

torch.backends.cudnn.enabled = False


def load_transformer(weights_path, device):
    model = ECGTransformer(
        input_channels=12, d_model=128, nhead=8, num_layers=4, num_classes=5
    )
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_resnet18(weights_path, device):
    state_dict = torch.load(weights_path, map_location=device)
    resnet = models.resnet18(weights=None)
    num_ftrs = resnet.fc.in_features

    # Riconosce automaticamente se i pesi salvati includono il Dropout prima di fc
    if "fc.1.weight" in state_dict:
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, 5),
        )
    else:
        resnet.fc = nn.Linear(num_ftrs, 5)

    resnet.load_state_dict(state_dict)
    resnet.to(device)
    resnet.eval()
    return resnet


def get_predictions(model, loader, device, is_cnn=False):
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in loader:
            if is_cnn:
                inputs = inputs.contiguous().to(device)
            else:
                inputs = inputs.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    return np.concatenate(all_targets, axis=0), np.concatenate(all_probs, axis=0)


def compute_full_report(y_true, probs, model_name, threshold=0.5):
    preds = (probs >= threshold).astype(int)

    row = {
        "Modello": model_name,
        "ROC_AUC_Macro": roc_auc_score(y_true, probs, average="macro"),
        "PR_AUC_Macro": average_precision_score(y_true, probs, average="macro"),
        "F1_Macro": f1_score(y_true, preds, average="macro", zero_division=0),
        "Precision_Macro": precision_score(y_true, preds, average="macro", zero_division=0),
        "Recall_Macro": recall_score(y_true, preds, average="macro", zero_division=0),
        "Accuracy_Exact": accuracy_score(y_true, preds),
    }

    roc_per_class = roc_auc_score(y_true, probs, average=None)
    pr_per_class = average_precision_score(y_true, probs, average=None)

    for idx, cls_name in enumerate(SUPERCLASSES):
        row[f"ROC_{cls_name}"] = roc_per_class[idx]
        row[f"PR_{cls_name}"] = pr_per_class[idx]

    return row


def main():
    parser = argparse.ArgumentParser(description="Valutazione finale su Test Set PTB-XL (Fold 10)")
    parser.add_argument("--data_dir", type=str, default="data/ptb-xl", help="Cartella radice PTB-XL")
    parser.add_argument(
        "--transformer_weights",
        type=str,
        default="weight/transformer_ecg_best_filter.pth",
        help="Percorso pesi Transformer 1D",
    )
    parser.add_argument(
        "--cnn_weights",
        type=str,
        default="runs/resnet_ptbxl_reg/resnet18_images_best.pth",
        help="Percorso pesi ResNet-18 2D",
    )
    parser.add_argument("--image_folder", type=str, default="images_512_4x3", help="Sottocartella immagini 4x3")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per inferenza")
    parser.add_argument("--no_filter", action="store_true", help="Disattiva il filtro sul segnale 1D del Transformer")
    parser.add_argument("--output_csv", type=str, default="test_results_comparison.csv", help="File CSV di output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device di valutazione: {device}")

    # 1. Caricamento Test Set ufficiale (strat_fold == 10)
    _, _, test_df = prepare_ptbxl_5class_data(args.data_dir)
    print(f"Campioni nel Test Set (Fold 10): {len(test_df)}\n")

    # 2. DataLoader per Transformer 1D (segnali numerici) e CNN 2D (immagini 4x3)
    apply_filter = not args.no_filter
    test_dataset_1d = PTBXLDataset(test_df, args.data_dir, sampling_rate=100, apply_filter=apply_filter)
    test_dataset_2d = PTBXLImageDataset(test_df, args.data_dir, image_folder=args.image_folder)

    loader_1d = DataLoader(test_dataset_1d, batch_size=args.batch_size, shuffle=False, num_workers=0)
    loader_2d = DataLoader(test_dataset_2d, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 3. Inferenza Transformer 1D
    print(f"[1/2] Valutazione Transformer 1D (filtro={apply_filter}) da: {args.transformer_weights}")
    transformer = load_transformer(args.transformer_weights, device)
    y_true_1d, probs_transformer = get_predictions(transformer, loader_1d, device, is_cnn=False)

    # 4. Inferenza ResNet-18 2D
    print(f"[2/2] Valutazione ResNet-18 2D da: {args.cnn_weights}")
    resnet = load_resnet18(args.cnn_weights, device)
    y_true_2d, probs_cnn = get_predictions(resnet, loader_2d, device, is_cnn=True)

    # Verifica di allineamento dei campioni tra i due DataLoader
    assert np.array_equal(y_true_1d, y_true_2d), "Errore: disallineamento tra i target 1D e 2D!"

    # 5. Calcolo Ensemble (Soft Voting 50% Transformer + 50% ResNet-18)
    probs_ensemble = 0.5 * probs_transformer + 0.5 * probs_cnn

    # 6. Generazione report comparativo
    results = [
        compute_full_report(y_true_1d, probs_transformer, "Transformer 1D"),
        compute_full_report(y_true_1d, probs_cnn, "ResNet-18 2D"),
        compute_full_report(y_true_1d, probs_ensemble, "Ensemble (1D + 2D)"),
    ]
    df_results = pd.DataFrame(results)

    print("\n" + "=" * 85)
    print("RISULTATI TEST SET (FOLD 10) - METRICHE MACRO GLOBALI")
    print("=" * 85)
    macro_cols = [
        "Modello",
        "ROC_AUC_Macro",
        "PR_AUC_Macro",
        "F1_Macro",
        "Precision_Macro",
        "Recall_Macro",
        "Accuracy_Exact",
    ]
    print(df_results[macro_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 85)
    print("DETTAGLIO PER SINGOLA SUPER-CLASSE - ROC-AUC")
    print("=" * 85)
    roc_cols = ["Modello"] + [f"ROC_{c}" for c in SUPERCLASSES]
    print(df_results[roc_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 85)
    print("DETTAGLIO PER SINGOLA SUPER-CLASSE - PR-AUC")
    print("=" * 85)
    pr_cols = ["Modello"] + [f"PR_{c}" for c in SUPERCLASSES]
    print(df_results[pr_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    df_results.to_csv(args.output_csv, index=False, float_format="%.4f")
    print(f"\n[+] Tabella completa salvata in: {args.output_csv}")


if __name__ == "__main__":
    main()