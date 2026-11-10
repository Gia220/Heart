import argparse
import torch
import torch.nn as nn

from dataset import set_seed, load_splits, make_loaders
from train_utils import train_model
from metrics_report import predict_proba, find_threshold, evaluate_and_save

EMB_COLS = {"GenHlth": 6, "Age": 14, "Education": 7, "Income": 9, "Diabetes": 3}


class DeepMLPClassifier(nn.Module):


    def __init__(self, in_features, hidden_units, n_hidden=2, out_classes=1):
        super(DeepMLPClassifier, self).__init__()
        layers, prev = [], in_features
        for _ in range(n_hidden):
            layers += [nn.Linear(prev, hidden_units), nn.ReLU()]
            prev = hidden_units
        layers.append(nn.Linear(prev, out_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/heart_disease_health_indicators.csv")
    ap.add_argument("--name_model", default=None, help="nome della run (default: automatico)")
    ap.add_argument("--hidden_units", type=int, default=128, help="neuroni per ogni livello nascosto")
    ap.add_argument("--n_hidden", type=int, default=2, help="livelli nascosti")
    ap.add_argument("--optim", choices=["adamw", "sgd"], default="adamw",help="sgd = come nel lab (SGD con momentum); richiede la patch a train_utils.py")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-3 con adamw, 0.01 con sgd")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--no_pos_weight", action="store_true")
    ap.add_argument("--min_recall", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_features", type=int, default=None, help="Numero di feature top da mantenere")
    args = ap.parse_args()

    lr = args.lr if args.lr is not None else (0.01 if args.optim == "sgd" else 1e-3)
    model_name = args.name_model or (f"DeepMLP_h{args.hidden_units}x{args.n_hidden}_k{args.k_features if args.k_features else 'ALL'}")

    # Riproducibilita'
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Split stratificato train/val/test, scaler fittato solo sul train
    splits = load_splits(args.csv, seed=args.seed)

    #N di feature ridotto
    if args.k_features is not None:
        print(f"Selezione delle {args.k_features} feature più importanti in corso...")
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Usa il test ANOVA (f_classif) per valutare l'importanza
        selector = SelectKBest(score_func=f_classif, k=args.k_features)
        
        # FIT RIGOROSO SOLO SUL TRAIN: impara quali sono le colonne migliori
        splits["X_train"] = selector.fit_transform(splits["X_train"], splits["y_train"])
        
        # APPLICA LA MASCHERA A VAL E TEST (senza guardare le loro etichette)
        splits["X_val"] = selector.transform(splits["X_val"])
        splits["X_test"] = selector.transform(splits["X_test"])
        
        # Stampa gli indici delle colonne salvate (fondamentale per la tua tesi)
        colonne_scelte = selector.get_support(indices=True)
        print(f"[!] Feature originarie mantenute (indici): {colonne_scelte}")

    train_loader, val_loader, test_loader = make_loaders(splits, batch_size=args.batch)

    input_dim = splits["X_train"].shape[1]
    print(f"Dati caricati! Feature in ingresso: {input_dim}")


    model = DeepMLPClassifier(input_dim, args.hidden_units, args.n_hidden)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{model_name}: {args.n_hidden} livelli nascosti x {args.hidden_units} neuroni, {n_par:,} parametri")

    # Argomenti extra solo se si chiede SGD: con AdamW (default) funziona con qualsiasi train_utils.py
    extra = {"optimizer_name": "sgd", "momentum": args.momentum} if args.optim == "sgd" else {}

    print("Inizio dell'addestramento del Deep MLP...")
    trained = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        name_model=model_name,
        epochs=args.epochs,
        lr=lr,
        weight_decay=args.wd,
        pos_weight=None if args.no_pos_weight else splits["pos_weight"],
        patience=args.patience,
        device=device,
        **extra,
    )
    print("Addestramento completato!")

    # Soglia scelta sul validation, test usato una sola volta
    y_val, p_val = predict_proba(trained, val_loader, device)
    thr = find_threshold(y_val, p_val, min_recall=args.min_recall)
    print(f"Soglia scelta sul validation: {thr:.3f}")

    evaluate_and_save(trained, test_loader, model_name=model_name, threshold=thr, device=device)
