import os
# Blocco cuDNN per i problemi del cluster a 512x512
os.environ["TORCH_USE_CUDNN"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
from dataset_ptbxl import compute_metrics

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from dataset_ptbxl import prepare_ptbxl_5class_data

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


def main():
    parser = argparse.ArgumentParser(description='ResNet-18 PTB-XL - Fine Tuning Regolarizzato')
    parser.add_argument('--data_dir', type=str, default="data/ptb-xl", help='Percorso cartella PTB-XL')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='./runs/resnet_ptbxl_images')
    parser.add_argument("--run_id", type=str, default="3", help="ID della run per salvare i pesi")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware in uso: {device}")

    train_df, val_df, _ = prepare_ptbxl_5class_data(args.data_dir)
    
    train_dataset = PTBXLImageDataset(train_df, args.data_dir, image_folder="images_512_4x3")
    val_dataset = PTBXLImageDataset(val_df, args.data_dir, image_folder="images_512_4x3")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Congela tutta la rete
    for param in resnet.parameters():
        param.requires_grad = False
        
    # Sblocca SOLO il secondo (ultimo) blocco di layer4 invece di tutto layer4
    for param in resnet.layer4[1].parameters():
        param.requires_grad = True

    # Aggiungi Dropout prima del layer lineare per ridurre la co-dipendenza tra neuroni
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 5)
    )
    resnet = resnet.to(device)

    criterion = nn.BCEWithLogitsLoss()

    # ADAMW con Weight Decay a 1e-3 come suggerito
    optimizer = optim.AdamW([
        {'params': resnet.layer4[1].parameters(), 'lr': 2e-5},
        {'params': resnet.fc.parameters(), 'lr': 3e-4}
    ], weight_decay=1e-2)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    writer = SummaryWriter(log_dir=args.output_dir)
    
    # Setup Early Stopping
    best_val_loss = float('inf')
    patience = 3
    epochs_no_improve = 0

    print("\nInizio addestramento ResNet-18 Regolarizzata...")
    epochs = args.epochs
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- TRAINING ---
        resnet.train()
        resnet.bn1.eval()
        resnet.layer1.eval()
        resnet.layer2.eval()
        resnet.layer3.eval()
        resnet.layer4[0].eval()
        
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
            
        train_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        
        # --- VALIDATION ---
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
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoca {epoch+1:3d}/{epochs} | train loss {train_loss:.4f} | val loss {epoch_val_loss:.4f} | " 
              f"PR-AUC {m['PR_AUC']:.4f} | ROC-AUC {m['ROC_AUC']:.4f} | lr {current_lr:.1e}")
        writer.add_scalar("Loss/val", epoch_val_loss, epoch)
        writer.add_scalar("PR_AUC/val", m['PR_AUC'], epoch)
        writer.add_scalar("Precision/val", m["Precision"], epoch)
        writer.add_scalar("Recall/val", m["Recall"], epoch)
        writer.add_scalar("F1_Score/val", m["F1_Score"], epoch)
        writer.add_scalar("Accuracy/val", m["Accuracy"], epoch)               
        writer.add_scalar("ROC_AUC/val", m["ROC_AUC"], epoch)
        writer.add_scalar("LR", current_lr, epoch)        
  
        
        # Logica di salvataggio 
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            model_path = os.path.join(args.output_dir, 'resnet18_images_best.pth')
            save_path = os.path.join("Weight", f"CNN_{args.run_id}.pth")
            torch.save(resnet.state_dict(), save_path)

            print(f"  [+] Miglior modello salvato! (Loss scesa a {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  [-] Nessun miglioramento per {epochs_no_improve} epoche.")
            if epochs_no_improve >= patience:
                print(f"\n[!] EARLY STOPPING attivato all'epoca {epoch+1}.")
                break

    writer.close()

if __name__ == '__main__':
    main()