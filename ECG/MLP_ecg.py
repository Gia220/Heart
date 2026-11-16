import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_ptbxl import prepare_ptbxl_5class_data, PTBXLDataset, compute_metrics

torch.backends.cudnn.enabled = False

class MLP(nn.Module):
    """
    MLP su segnale ECG concatenato (12 * 1000 = 12.000 feature).
    Architettura: Input (12000) -> Hidden (hidden_units) -> Output (5)
    """

    def __init__(self, in_features=12000, hidden_units=256, dropout=0.4, num_classes=5):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                         # [Batch, 12, 1000] -> [Batch, 12000]
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden_units), # Unico hidden layer
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes), # Layer di uscita (5 super-classi)
        )

    def forward(self, x):
        return self.net(x)

def train_model(
    data_dir,
    epochs=30,
    batch_size=64,
    lr=1e-4,
    hidden_units=256,
    dropout=0.4,
    log_dir="runs/mlp_ptbxl",
    run_id="mlp_flat",
    apply_filter=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione su device: {device} | Filtro attivo: {apply_filter}")

    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard log dir: {log_dir}")

    train_df, val_df, _ = prepare_ptbxl_5class_data(data_dir)
    train_loader = DataLoader(
        PTBXLDataset(train_df, data_dir, apply_filter=apply_filter),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        PTBXLDataset(val_df, data_dir, apply_filter=apply_filter),
        batch_size=batch_size,
        shuffle=False,
    )

    model = MLP(
        in_features=12 * 1000,
        hidden_units=hidden_units, 
        dropout=dropout,
        num_classes=5,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri totali ECGFlattenedMLP: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    patience = 7
    epochs_no_improve = 0

    os.makedirs("weight", exist_ok=True)
    save_path = os.path.join("weight", f"mlp_ecg_best_{run_id}.pth")

    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        writer.add_scalar("Loss/Train", train_loss, epoch)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        m = compute_metrics(all_targets, all_preds, threshold=0.5)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoca {epoch+1:3d}/{epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"PR-AUC {m['PR_AUC']:.4f} | ROC-AUC {m['ROC_AUC']:.4f} | lr {current_lr:.1e}"
        )

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("PR_AUC/val", m["PR_AUC"], epoch)
        writer.add_scalar("Precision/val", m["Precision"], epoch)
        writer.add_scalar("Recall/val", m["Recall"], epoch)
        writer.add_scalar("F1_Score/val", m["F1_Score"], epoch)
        writer.add_scalar("Accuracy/val", m["Accuracy"], epoch)
        writer.add_scalar("ROC_AUC/val", m["ROC_AUC"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  [+] Nuovo modello top salvato! (Val Loss scesa a {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[!] Early Stopping attivato all'epoca {epoch+1}.")
                break

    writer.close()
    print(f"\nAddestramento completato. Il modello migliore è in: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training MLP Flattened su PTB-XL")
    parser.add_argument("--data_dir", type=str, required=True, help="Percorso della cartella PTB-XL")
    parser.add_argument("--epochs", type=int, default=30, help="Numero di epoche")
    parser.add_argument("--batch_size", type=int, default=64, help="Dimensione del batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=256, help="Neuroni primo layer nascosto")
    parser.add_argument("--dropout", type=float, default=0.4, help="Probabilità di Dropout")
    parser.add_argument("--log_dir", type=str, default=None, help="Cartella log TensorBoard")
    parser.add_argument("--run_id", type=str, default="flat", help="ID della run per salvare i pesi")
    parser.add_argument("--no_filter", action="store_true", help="Disattiva il filtro passa-banda")

    args = parser.parse_args()
    apply_filter = not args.no_filter
    log_dir = args.log_dir or f"runs/MLP_flat_ep{args.epochs}_bs{args.batch_size}_lr{args.lr}_{args.run_id}"

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_units=args.hidden_units,
        dropout=args.dropout,
        log_dir=log_dir,
        run_id=args.run_id,
        apply_filter=apply_filter,
    )