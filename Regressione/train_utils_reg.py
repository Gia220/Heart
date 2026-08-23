import torch
from torch import nn
from torch import optim
import copy

def train_regression_model(model, train_loader, test_loader, name_model, epochs=150, lr=0.01):
    # Usiamo l'Errore Quadratico Medio (MSE) invece della CrossEntropy
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        # --- FASE DI TRAIN ---
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # --- FASE DI TEST ---
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                
        test_loss = test_loss / len(test_loader.dataset)
        
        # Salviamo i pesi se l'errore è sceso
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
        # Stampa i log ogni 10 epoche per non inondare il terminale
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train MSE: {train_loss:.4f} - Test MSE: {test_loss:.4f}")
            
    print(f"Miglior Test MSE raggiunto: {best_loss:.4f}")
    
    # Carichiamo i pesi dell'epoca migliore
    model.load_state_dict(best_model_wts)
    return model