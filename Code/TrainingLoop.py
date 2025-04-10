import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time



def TrainingLoop(classifier, epochs, trainLoader, valLoader, filepath, device):

    startTime = time.time()


    lossFunction = nn.MSELoss()
    optimiser = Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimiser, 'min', factor=0.5, patience=3, verbose=True)

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [],
        'val_rmse': [],
        'train_mape': [],
        'val_mape': []
    }

    # epochs = epochsMax
    patience = 10
    best_val_loss = 99999999.0
    epochs_no_improve = 0
    for epoch in range(epochs):
        classifier.train()
        epoch_loss = 0
        total_rmse = 0
        total_mape = 0
        num_batches = 0
        total_samples = 0

        train_losses, train_rmses, train_mapes = [], [], []


        for batch in trainLoader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)


            optimiser.zero_grad()  
            predictions = classifier(x_batch).flatten()  

            loss = lossFunction(predictions, y_batch.flatten()) 
            loss.backward() 
            optimiser.step()
            
            batchsize = y_batch.size(0)
            epoch_loss += loss.item() * batchsize  # Scale loss by batch size
            total_samples += batchsize 

            predictions_numpy = predictions.detach().cpu().numpy().reshape(-1, 1)
            y_batch_numpy = y_batch.detach().cpu().numpy().reshape(-1, 1)


            predictions_original = predictions_numpy.flatten()
            y_batch_original = y_batch_numpy.flatten()

            rmse = np.sqrt(np.mean((predictions_original - y_batch_original) ** 2))
            
            mape = np.mean(np.abs((predictions_original - y_batch_original) / (y_batch_original + 1))) * 100

            total_rmse += rmse
            total_mape += mape
            num_batches += 1

        # Validation
        classifier.eval()
        val_loss = 0
        val_rmse = 0
        val_mape = 0
        val_batches = 0
        with torch.no_grad():
            for batch in valLoader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                val_predictions = classifier(x_batch).flatten()
                loss = lossFunction(val_predictions, y_batch.flatten())
                val_loss += loss.item()

                val_predictions_numpy = val_predictions.cpu().numpy().reshape(-1, 1)
                y_val_numpy = y_batch.cpu().numpy().reshape(-1, 1)

                val_predictions_original = val_predictions_numpy.flatten()
                y_val_original = y_val_numpy.flatten()

                val_rmse += np.sqrt(np.mean((val_predictions_original - y_val_original) ** 2))
                val_mape += np.mean(np.abs((val_predictions_original - y_val_original) / (y_val_original + 1))) * 100 

                val_batches += 1

        avg_rmse = total_rmse / num_batches
        avg_mape = total_mape / num_batches

        avg_val_rmse = val_rmse / val_batches
        avg_val_loss = val_loss / val_batches
        avg_val_mape = val_mape / val_batches

        
        history["train_loss"].append(epoch_loss/total_samples)
        history["train_rmse"].append(avg_rmse)
        history["train_mape"].append(avg_mape)

        history["val_loss"].append(avg_val_loss)
        history["val_rmse"].append(avg_val_rmse)
        history["val_mape"].append(avg_val_mape)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/total_samples:.6f}, Train RMSE: {avg_rmse:.4f}, Train MAPE: {avg_mape:.2f}% | "
            f"Val Loss: {avg_val_loss:.6f}, Val RMSE: {avg_val_rmse:.4f}, Val MAPE: {avg_val_mape:.2f}%")
        
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(classifier.state_dict(), filepath + 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                epochs = epoch 
                classifier.load_state_dict(torch.load(filepath + 'best_model.pth'))
                break


    endTime = time.time()

    return history, epochs, endTime - startTime
