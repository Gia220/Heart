from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import torch
import datetime

def train_model(model, train_loader, test_loader, name_model, epochs=100, lr=0.01,weight_decay=0.0):
    
    # Definiamo la loss function stabile applicata direttamente ai logit
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Definiamo l'ottimizzatore (Stochastic Gradient Descent)[cite: 5]
    optimizer = SGD(model.parameters(), lr=lr, weight_decay= weight_decay)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Inizializziamo TensorBoard per tracciare le metriche
    run_log_dir = f"logs/{name_model}_{timestamp}"
    writer = SummaryWriter(run_log_dir)
    
    for epoch in range(epochs):
        
        # Mettiamo il modello in modalità training[cite: 2]
        model.train() 
        epoch_loss = 0.0
        
        for X_batch, Y_batch in train_loader:
            
            # Azzeriamo i gradienti accumulati nell'iterazione precedente[cite: 2]
            optimizer.zero_grad() 
            
            # Forward pass: calcoliamo le predizioni
            outputs = model(X_batch)
            
            # Calcoliamo il valore della loss[cite: 2]
            loss = criterion(outputs, Y_batch) 
            
            # Backward pass: calcoliamo il gradiente della loss rispetto ai parametri[cite: 2]
            loss.backward() 
            
            # Aggiorniamo i pesi[cite: 2]
            optimizer.step() 
            
            epoch_loss += loss.item()

        # Loggiamo la media della loss di addestramento su TensorBoard[cite: 2]
        writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)
        
        # --- Fase di Test / Valutazione ---
        # Mettiamo il modello in modalità valutazione[cite: 2]
        model.eval() 
        
        # Disabilitiamo i gradienti per risparmiare memoria durante il test[cite: 2]
        with torch.no_grad(): 
            test_loss = 0.0
            all_preds = []
            all_targets = []
            
            for X_test, Y_test in test_loader:
                test_outputs = model(X_test)
                
                # 1. Calcoliamo la loss di test per questo batch e la sommiamo[cite: 2]
                batch_loss = criterion(test_outputs, Y_test)
                test_loss += batch_loss.item()
                
                # 2. Calcoliamo le predizioni per l'accuracy
                probs = torch.sigmoid(test_outputs)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.numpy())
                all_targets.extend(Y_test.numpy())
                
            # Calcoliamo le metriche medie per l'intera epoca
            epoch_test_loss = test_loss / len(test_loader)
            acc = accuracy_score(all_targets, all_preds)
            
            # Tracciamo entrambe le metriche su TensorBoard[cite: 2, 4]
            writer.add_scalar('Loss/test', epoch_test_loss, epoch)
            writer.add_scalar('Accuracy/test', acc, epoch)
            
    writer.close()
    return model
