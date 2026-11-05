import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             multilabel_confusion_matrix, roc_auc_score, average_precision_score)
from dataset_ptbxl import prepare_ptbxl_5class_data, PTBXLDataset

# Disabilita cuDNN per compatibilità con vecchie GPU K80
torch.backends.cudnn.enabled = False

class ECGTransformer(nn.Module):
    def __init__(self, input_channels=12, d_model=128, nhead=8, num_layers=4, num_classes=5):
        super(ECGTransformer, self).__init__()
        
        # Comprime i 1000 timestep a 250 ed espande i canali a d_model
        self.conv_stem = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=d_model, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # Positional Encoding (Parametro apprendibile)
        self.pos_encoder = nn.Parameter(torch.randn(1, 250, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classificatore Globale
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.conv_stem(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out


def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, average='macro', zero_division=0),
        "Recall": recall_score(y_true, preds, average='macro', zero_division=0),
        "F1_Score": f1_score(y_true, preds, average='macro', zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, probs, average='macro'),
        "PR_AUC": average_precision_score(y_true, probs, average='macro'),
    }



def train_model(data_dir, epochs=30, batch_size=64, lr=1e-4, log_dir="runs/transformer_2", run_id="run0"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione su device: {device}")

    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard log dir: {log_dir}")

    train_df, val_df, _ = prepare_ptbxl_5class_data(data_dir)
    train_loader = DataLoader(PTBXLDataset(train_df, data_dir), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PTBXLDataset(val_df, data_dir), batch_size=batch_size, shuffle=False)

    model = ECGTransformer().to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 1. SCHEDULER: Dimezza il learning rate se la val_loss non scende per 3 epoche di fila
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Variabile per tracciare la migliore validation loss
    best_val_loss = float('inf')
    
    os.makedirs("Weight", exist_ok=True)
    save_path = os.path.join("Weight", f"transformer_ecg_best_{run_id}.pth")

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
        writer.add_scalar('Loss/Train', train_loss, epoch)
        
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
        
        # Calcolo Metriche
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        m = compute_metrics(all_targets, all_preds, threshold=0.5)

        # 2. AGGIORNAMENTO SCHEDULER: Passiamo la val_loss per fargli valutare se ridurre il LR
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoca {epoch+1:3d}/{epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | " 
              f"PR-AUC {m['PR_AUC']:.4f} | ROC-AUC {m['ROC_AUC']:.4f} | lr {current_lr:.1e}")
        
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("PR_AUC/val", m['PR_AUC'], epoch)
        writer.add_scalar("Precision/val", m["Precision"], epoch)
        writer.add_scalar("Recall/val", m["Recall"], epoch)
        writer.add_scalar("F1_Score/val", m["F1_Score"], epoch)
        writer.add_scalar("Accuracy/val", m["Accuracy"], epoch)               
        writer.add_scalar("ROC_AUC/val", m["ROC_AUC"], epoch)
        writer.add_scalar("LR", current_lr, epoch)
        
        # 3. MODEL CHECKPOINTING: Salva i pesi SOLO se la Validation Loss è la migliore vista finora
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  [+] Nuovo modello top salvato! (Val Loss scesa a {best_val_loss:.4f})")

    writer.close()
    print(f"\nAddestramento completato. Il modello migliore è in: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Transformer su PTB-XL")
    parser.add_argument("--data_dir", type=str, required=True, help="Percorso della cartella PTB-XL")
    parser.add_argument("--epochs", type=int, default=30, help="Numero di epoche")
    parser.add_argument("--batch_size", type=int, default=64, help="Dimensione del batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    run_id = "baseline_Wander"

    # La cartella log conterrà il timestamp per non sovrascrivere mai i vecchi grafici
    log_dir = f"runs/Transformer_ep{args.epochs}_bs{args.batch_size}_lr{args.lr}_{run_id}"
    
    train_model(
        data_dir=args.data_dir, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        lr=args.lr,
        log_dir=log_dir,
        run_id=run_id
    )