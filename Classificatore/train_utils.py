from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import torch
import datetime

def train_model(model, train_loader, test_loader, name_model, epochs=100, lr=0.01, weight_decay=0.0):
    
    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    
    #Stochastic Gradient Descent
    optimizer = SGD(model.parameters(), lr=lr, weight_decay= weight_decay)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")       #time per i log

    run_log_dir = f"logs/{name_model}_{timestamp}"                      #cartella log
    writer = SummaryWriter(run_log_dir)

    #inzio training
    for epoch in range(epochs):
        
        model.train() 

        epoch_loss = 0.0
        
        for X_batch, Y_batch in train_loader:
            
            optimizer.zero_grad()                   #azzeriamo i gradienti
            
            outputs = model(X_batch)                #calcolo predizioni
            
            loss = criterion(outputs, Y_batch)      # Calcoliamo il valore della loss
            
            loss.backward() 
            
            optimizer.step()                        # Aggiornamento pesi
            
            epoch_loss += loss.item()

        writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)          # Log la media della loss di addestramento su TensorBoard

        
        #Fase di Test
        model.eval() 
        
        with torch.no_grad(): 
            test_loss = 0.0
            all_preds = []
            all_targets = []
            
            for X_test, Y_test in test_loader:
                test_outputs = model(X_test)
                
                #Calcoliamo la loss di test per questo batch e la sommiamo
                batch_loss = criterion(test_outputs, Y_test)
                test_loss += batch_loss.item()
                
                #Calcoliamo le predizioni per l'accuracy
                probs = torch.sigmoid(test_outputs)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.numpy())
                all_targets.extend(Y_test.numpy())
                
            #metriche medie per l'intera epoca
            epoch_test_loss = test_loss / len(test_loader)
            acc = accuracy_score(all_targets, all_preds)
            
            # Log entrambe le metriche su TensorBoard
            writer.add_scalar('Loss/test', epoch_test_loss, epoch)
            writer.add_scalar('Accuracy/test', acc, epoch)
            
    writer.close()
    return model
