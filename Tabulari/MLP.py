import argparse
import torch
import torch.nn as nn

from dataset import set_seed, load_splits, make_loaders
from train_utils import train_model
from metrics_report import predict_proba, find_threshold, evaluate_and_save


class HeartDiseaseMLP(nn.Module):
    def __init__(self, in_features, hidden_1=64, hidden_2=32):
        super(HeartDiseaseMLP, self).__init__()

        self.model = nn.Sequential(
            # Primo Hidden Layer
            nn.Linear(in_features, hidden_1),
            nn.ReLU(),

            # Secondo Hidden Layer
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),

            # Layer di Output
            nn.Linear(hidden_2, 1)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/heart_disease_health_indicators.csv")
    ap.add_argument("--h1", type=int, default=64)
    ap.add_argument("--h2", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--no_pos_weight", action="store_true")
    ap.add_argument("--min_recall", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name_model", default="name_model")

    args = ap.parse_args()

    model_name = args.name_model


    # Riproducibilità
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Split stratificato train/val/test, scaler fittato solo sul train
    splits = load_splits(args.csv, seed=args.seed)
    train_loader, val_loader, test_loader = make_loaders(splits, batch_size=args.batch)

    input_dim = splits["X_train"].shape[1]
    print(f"Dati caricati! Feature in ingresso: {input_dim}")

    print(f"Inizializzazione del MLP ({args.h1}-{args.h2})...")
    mlp_model = HeartDiseaseMLP(in_features=input_dim, hidden_1=args.h1, hidden_2=args.h2)

    print("Inizio dell'addestramento del MLP...")
    trained_mlp = train_model(
        model=mlp_model,
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
    print("Addestramento completato!")

    # Soglia scelta sul validation, test usato una sola volta
    y_val, p_val = predict_proba(trained_mlp, val_loader, device)
    thr = find_threshold(y_val, p_val, min_recall=args.min_recall)
    print(f"Soglia scelta sul validation: {thr:.3f}")

    evaluate_and_save(trained_mlp, test_loader, model_name=model_name, threshold=thr, device=device)
