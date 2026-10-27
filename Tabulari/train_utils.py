import copy
import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score

from metrics_report import predict_proba, compute_metrics, find_threshold


def unique_run_name(base, logs_dir="logs"):
    """Se logs/<base> esiste già, restituisce <base>_1, <base>_2, ..."""
    name, i = base, 1
    while os.path.exists(os.path.join(logs_dir, name)):
        name = f"{base}_{i}"
        i += 1
    return name


def train_model(model, train_loader, val_loader, name_model, epochs=100, lr=1e-3,
                weight_decay=1e-4, pos_weight=None, patience=15, device="cpu",
                weights_dir="weight"):
    """
    Allena con AdamW, scheduler e early stopping sulla PR-AUC di validation.
    """
    model.to(device)

    pw = torch.tensor([pos_weight], device=device) if pos_weight else None

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    run_name = unique_run_name(name_model)
    writer = SummaryWriter(f"logs/{run_name}")
    os.makedirs(weights_dir, exist_ok=True)

    #per salvare i pesi della migliore epoca e per l'early stop
    best_ap, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(epochs):
        # train 
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        #  validation 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                val_loss += criterion(model(X.to(device)), y.to(device)).item()
        val_loss /= len(val_loader)

        y_val, p_val = predict_proba(model, val_loader, device)
        ap = average_precision_score(y_val, p_val)
        m, _ = compute_metrics(y_val, p_val, threshold=0.5)
        scheduler.step(ap)

        print(f"Epoca {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | " f"PR-AUC {ap:.4f} | ROC-AUC {m['ROC_AUC']:.4f} | lr {optimizer.param_groups[0]['lr']:.1e}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("PR_AUC/val", ap, epoch)
        writer.add_scalar("Precision/val", m["Precision"], epoch)
        writer.add_scalar("Recall/val", m["Recall"], epoch)
        writer.add_scalar("F1_Score/val", m["F1_Score"], epoch)
        writer.add_scalar("Accuracy/val", m["Accuracy"], epoch)               
        writer.add_scalar("ROC_AUC/val", m["ROC_AUC"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)



        #  checkpoint + early stopping 
        if ap > best_ap:
            best_ap, bad_epochs = ap, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, f"{weights_dir}/{run_name}.pth")
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                print(f"Early stopping all'epoca {epoch} (miglior PR-AUC val: {best_ap:.4f})")
                break

    writer.close()
    model.load_state_dict(best_state)
    return model