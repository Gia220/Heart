# RegLog.py
import argparse
import torch
from torch import nn

from dataset import set_seed, load_splits, make_loaders
from train_utils import train_model
from metrics_report import predict_proba, find_threshold, evaluate_and_save


class LogisticRegressor(nn.Module):
    def __init__(self, in_features):
        super(LogisticRegressor, self).__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        logits = self.linear(x)
        return logits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/heart_disease_health_indicators.csv")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--no_pos_weight", action="store_true")
    ap.add_argument("--min_recall", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name_model", default="name_model")
    args = ap.parse_args()

    model_name = args.name_model

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Split stratificato train/val/test, scaler fittato solo sul train
    splits = load_splits(args.csv, seed=args.seed)
    train_loader, val_loader, test_loader = make_loaders(splits, batch_size=args.batch)

    input_dim = splits["X_train"].shape[1]
    print(f"Dati caricati! Feature in ingresso: {input_dim}")

    print(f"Inizializzazione del LogisticRegressor con {input_dim} feature in ingresso...")
    model = LogisticRegressor(in_features=input_dim)

    print("Inizio dell'addestramento")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        name_model=model_name,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        pos_weight=None if args.no_pos_weight else splits["pos_weight"],
        patience=args.patience,
        device=device,
    )
    print("Addestramento completato con successo!")

    # Soglia scelta sul validation, test usato una sola volta
    y_val, p_val = predict_proba(trained_model, val_loader, device)
    thr = find_threshold(y_val, p_val, min_recall=args.min_recall)
    print(f"Soglia scelta sul validation: {thr:.3f}")

    evaluate_and_save(trained_model, test_loader, model_name=model_name,threshold=thr, device=device)
    # I pesi della migliore epoca sono già salvati in weight/{model_name}.pth da train_model
