import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import kagglehub

def main():
    parser = argparse.ArgumentParser(description='ResNet-18 ECG Fine-Tuning')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64, consigliato per GPU cluster)')
    parser.add_argument('--epochs', type=int, default=15, help='Numero di epoche (default: 15)')
    parser.add_argument('--output_dir', type=str, default='./runs/ecg_resnet', help='Cartella per TensorBoard e pesi')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware in uso: {device}")

    #Download tramite Kaggle API 
    print("Download dataset da 100k immagini in corso (Kaggle)...")
    base_path = kagglehub.dataset_download("mhasnain1806/ecg-images")
    
    data_dir = None
    for root, dirs, files in os.walk(base_path):
        if 'MI_segment' in dirs and 'Normal_segment' in dirs:
            data_dir = root
            break
            
    if data_dir is None:
        raise FileNotFoundError("Errore: Impossibile trovare le cartelle delle classi.")

    print(f"Dati pronti in: {data_dir}")

    #  Pipeline Dati
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_for_train = datasets.ImageFolder(root=data_dir, transform=train_transform)
    dataset_for_test = datasets.ImageFolder(root=data_dir, transform=test_transform)

    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset_for_train))
    test_size = len(dataset_for_train) - train_size
    train_indices, test_indices = random_split(range(len(dataset_for_train)), [train_size, test_size])

    train_dataset = Subset(dataset_for_train, train_indices.indices)
    test_dataset = Subset(dataset_for_test, test_indices.indices)

    # Nota: num_workers=4 velocizza il caricamento dati sul cluster
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    #  Inizializzazione Modello Binario
    resnet = models.resnet18(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    for param in resnet.layer4.parameters():
        param.requires_grad = True

    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 2)
    resnet = resnet.to(device)

    pesi_classi = torch.tensor([1.0, 4.22]).to(device)
    criterion = nn.CrossEntropyLoss(weight=pesi_classi)

    optimizer = optim.Adam([
        {'params': resnet.layer4.parameters(), 'lr': 1e-4},
        {'params': resnet.fc.parameters(), 'lr': 1e-3}
    ])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    # Training Loop
    print("\nInizio addestramento sul Cluster...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train 
        resnet.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Test 
        resnet.eval()
        val_loss, correct_test, total_test = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = resnet(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        epoch_test_loss = val_loss / total_test
        epoch_test_acc = correct_test / total_test

        scheduler.step(epoch_test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        writer.add_scalar('Learning_Rate/Layer4', current_lr, epoch)
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/Test', epoch_test_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/Test', epoch_test_acc, epoch)

        end_time = time.time()
        print(f"Epoca [{epoch+1}/{args.epochs}] | Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.4f} | LR: {current_lr} | Tempo: {end_time - start_time:.0f}s")

    # Salvataggio Pesi
    model_path = os.path.join(args.output_dir, 'resnet18_ecg_finetuned.pth')
    torch.save(resnet.state_dict(), model_path)
    print(f"\nAddestramento completato. Pesi salvati in: {model_path}")
    writer.close()

if __name__ == '__main__':
    main()