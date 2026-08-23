import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
import kagglehub

# ==========================================
# 1. SETUP E DIRECTORY
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware in uso: {device}")

# Creiamo una cartella separata per i log di questa rete
os.makedirs('risultati_ecg_custom/logs', exist_ok=True)
writer = SummaryWriter('risultati_ecg_custom/logs')

# ==========================================
# 2. DOWNLOAD E PREPARAZIONE DATASET
# ==========================================
print("Verifica dataset da 100k immagini (Kaggle)...")
data_dir = kagglehub.dataset_download("mhasnain1806/ecg-images")
print(f"Dati pronti in: {data_dir}")

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

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# ==========================================
# 3. DEFINIZIONE ARCHITETTURA CUSTOM
# ==========================================
class CustomECGNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomECGNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==========================================
# 4. INIZIALIZZAZIONE E TRAINING LOOP
# ==========================================
model = CustomECGNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 15

print("\nInizio addestramento Custom CNN sul Cluster...")
for epoch in range(epochs):
    start_time = time.time()
    
    # --- TRAINING ---
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / total_train
    epoch_train_acc = correct_train / total_train
    
    # --- TESTING ---
    model.eval()
    val_loss, correct_test, total_test = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    epoch_test_loss = val_loss / total_test
    epoch_test_acc = correct_test / total_test
    
    end_time = time.time()
    
    # LOGGING E TENSORBOARD
    print(f"Epoca [{epoch+1}/{epochs}] | Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.4f} | LR: 0.0005 | Tempo: {end_time - start_time:.0f}s")
    writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
    writer.add_scalar('Loss/Test', epoch_test_loss, epoch)
    writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
    writer.add_scalar('Accuracy/Test', epoch_test_acc, epoch)
    writer.add_scalar('Learning_Rate/Base', 0.0005, epoch)

print("\nAddestramento concluso! Salvataggio pesi...")
torch.save(model.state_dict(), 'risultati_ecg_custom/custom_ecg_net_final.pth')
writer.close()
print("Modello salvato in risultati_ecg_custom/custom_ecg_net_final.pth")