import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


def TestingLoop(classifier, epochsRan, testloader, filepath, device, dataSplit, scaler_x):

    lossFunction = nn.MSELoss()
    epochs = epochsRan
    xtest = dataSplit
    
    test_loss = 0
    test_rmse = 0
    test_mape = 0
    test_batches = 0
    predictions = []
    cloud_cover = []
    with torch.no_grad():
        for batch in testloader:
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            outputs = classifier(x_batch)
            predictions.extend(outputs.squeeze().tolist())

            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            test_predictions = classifier(x_batch).flatten()
            loss = lossFunction(test_predictions, y_batch.flatten())
            test_loss += loss.item()

            test_predictions_numpy = test_predictions.cpu().numpy().reshape(-1, 1)
            y_test_numpy = y_batch.cpu().numpy().reshape(-1, 1)

            test_predictions_original = test_predictions_numpy.flatten()
            y_test_original = y_test_numpy.flatten()

            test_rmse += np.sqrt(np.mean((test_predictions_original - y_test_original) ** 2))
            test_mape += np.mean(np.abs((test_predictions_original - y_test_original) / (y_test_original + 1))) * 100 

            test_batches += 1


    avg_test_rmse = test_rmse / test_batches
    avg_test_loss = test_loss / test_batches
    avg_test_mape = test_mape / test_batches


    print("----\n\n\n")
    print(f"Test Loss: {avg_test_loss:.6f}, Test RMSE: {avg_test_rmse:.4f}, Test MAPE: {avg_test_mape:.2f}%")



    print("\n\n\n----")

    with open(filepath + "TestMetrics.txt", "w") as f:
        f.write(f"Test Loss: {avg_test_loss:.6f}\n")
        f.write(f"Test RMSE: {avg_test_rmse:.4f}\n")
        f.write(f"Test MAPE: {avg_test_mape:.2f}%\n")
        f.write(f"Test epoch: {epochs:.2f}%\n")
    print("Metrics saved")


    for i in range(0, len(xtest)):
            
            temp = (scaler_x.inverse_transform(np.array(xtest[i][4]).reshape(1, -1)))
            cloud_cover.append(temp[0][4])

    return cloud_cover, predictions