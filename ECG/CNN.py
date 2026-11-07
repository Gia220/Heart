import os
# 1. BLOCCO ASSOLUTO DI CUDNN A LIVELLO DI SISTEMA (Prima di importare torch!)
os.environ["TORCH_USE_CUDNN"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import time
import argparse
import torch

# 2. BLOCCO DI CUDNN A LIVELLO PYTORCH
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from PIL import Image
import torchvision.transforms as transforms

# Importa dal file del dataset la funzione per i metadati
from dataset_ptbxl import prepare_ptbxl_5class_data

# ==========================================
# DEFINIZIONE DEL DATASET VISIVO
# ==========================================
class PTBXLImageDataset(Dataset):
    def __init__(self, df, data_dir, image_folder="images_512_4x3"):
        self.df = df
        self.data_dir = data_dir
        self.image_folder = image_folder
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = f"{row.name:05d}.jpg" 
        img_path = os.path.join(self.data_dir, self.image_folder, img_name)
        
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        
        labels = row.label_vector.astype(np.float32)
        
        return image_tensor, torch.tensor(labels)

# ==========================================
# FUNZIONI DI METRICA
# ==========================================
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

# ==========================================
# MAIN TRAINING LOOP
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='ResNet-18 PTB-XL Immagini 512x512')
    parser.add_argument('--data_dir', type=str, default="data/ptb-xl", help='Percorso cartella PTB-XL')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='./runs/resnet_ptbxl_images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware in uso: {device}")

    print(f"Caricamento metadati da: {args.data_dir}")
    train_df, val_df, _ = prepare_ptbxl_5class_data(args.data_dir)
    
    train_dataset = PTBXLImageDataset(train_df, args.data_dir, image_folder="images_512_4x3")
    val_dataset = PTBXLImageDataset(val_df, args.data_dir, image_folder="images_512_4x3")

    # 3. NUM_WORKERS A ZERO ASSOLUTO: previene la corruzione della memoria GPU sui cluster
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    for param in resnet.parameters():
        param.requires_grad = False
        
    for param in resnet.layer4.parameters():
        param.requires_grad = True

    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 5)
    resnet = resnet.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam([
        {'params': resnet.layer4.parameters(), 'lr': 1e-4},
        {'params': resnet.fc.parameters(), 'lr': 1e-3}
    ])
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(args.output_dir, f"logs_{run_id}"))
    
    best_val_loss = float('inf')

    print("\nInizio addestramento ResNet-18 sulle immagini 512x512...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        resnet.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.contiguous().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        resnet.eval()
        val_loss = 0.0
        all_targets, all_probs = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.contiguous().to(device)
                labels = labels.to(device)
                
                outputs = resnet(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                probs = torch.sigmoid(outputs)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(epoch_val_loss)
        
        all_targets_np = np.concatenate(all_targets, axis=0)
        all_probs_np = np.concatenate(all_probs, axis=0)
        m = compute_metrics(all_targets_np, all_probs_np, threshold=0.5)
        
        current_lr = optimizer.param_groups[0]['lr']
        end_time = time.time()
        
        print(f"Epoca {epoch+1:2d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | PR-AUC: {m['PR_AUC']:.4f} | ROC-AUC: {m['ROC_AUC']:.4f} | LR: {current_lr:.1e} | Tempo: {(end_time - start_time):.0f}s")
        
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/Val', epoch_val_loss, epoch)
        writer.add_scalar("PR_AUC/Val", m['PR_AUC'], epoch)
        writer.add_scalar("ROC_AUC/Val", m["ROC_AUC"], epoch)
        writer.add_scalar("Precision/Val", m["Precision"], epoch)
        writer.add_scalar("Recall/Val", m["Recall"], epoch)
        writer.add_scalar("F1_Score/Val", m["F1_Score"], epoch)
        writer.add_scalar("Accuracy/Val", m["Accuracy"], epoch)               
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model_path = os.path.join(args.output_dir, 'resnet18_images_best.pth')
            torch.save(resnet.state_dict(), model_path)
            print(f"  [+] Nuovo miglior modello salvato!")

    writer.close()

if __name__ == '__main__':
    main()