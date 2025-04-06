from transformers import PatchTSTConfig
from tsfm_public.toolkit.dataset import ForecastDFDataset

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Standard
import os

# Third Party
from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
)
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# First Party
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

from transformers import set_seed

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW

set_seed(2023)
device = "cpu"

if __name__ == '__main__':
    # The ECL data is available from https://github.com/zhouhaoyi/Informer2020?tab=readme-ov-file#data
    dataset_path = "C:\\Users\\demaw\\OneDrive\\Documents\\University\\Year3\\Project\\ProjectSolarPV\\Code\\Transformer\\ECL.csv"
    timestamp_column = "time"
    id_columns = []

    context_length = 3
    forecast_horizon = 1
    patch_length = 1
    num_workers = 2  # Reduce this if you have low number of CPU cores
    batch_size = 64  # Adjust according to GPU memory


    df_weather = pd.read_csv("Data\OpenMeteoData.csv")
    df_radiance = pd.read_csv("Data\PVGISdata.csv")

    df_radiance = df_radiance[1944:]

    df_weather['time'] = pd.to_datetime(df_weather['time'])

    df_radiance.drop(columns=['time'], inplace=True)

    df_weather.insert(0, 'ID', range(1, len(df_weather) + 1))
    df_radiance.insert(0, 'ID', range(1, len(df_radiance) + 1))

    df_merged = pd.merge(df_weather, df_radiance, on='ID')

    df_merged.drop(columns=['ID'], inplace=True)
    #df_merged.drop(columns=['time'], inplace=True)

    df_merged.dropna(inplace=True)

    df_np = df_merged[:].to_numpy()

    df_merged[timestamp_column] = df_merged[timestamp_column].astype(np.int64) / 1e9

    data = df_merged[:]


    forecast_columns = list(data.columns[1:])

    observable_columns_list = []

    observable_columns_list = [column for column in forecast_columns if column != "P"]
    # target_columns = [data.columns[0]]

    # get split
    num_train = int(len(data) * 0.7)
    num_test = int(len(data) * 0.2)
    num_valid = len(data) - num_train - num_test
    border1s = [
        0,
        num_train - context_length,
        len(data) - num_test - context_length,
    ]
    border2s = [num_train, num_train + num_valid, len(data)]

    train_start_index = border1s[0]  # None indicates beginning of dataset
    train_end_index = border2s[0]

    # we shift the start of the evaluation period back by context length so that
    # the first evaluation timestamp is immediately following the training data
    valid_start_index = border1s[1]
    valid_end_index = border2s[1]

    test_start_index = border1s[2]
    test_end_index = border2s[2]

    train_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=train_start_index,
        end_index=train_end_index,
    )
    valid_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=valid_start_index,
        end_index=valid_end_index,
    )
    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )

    time_series_preprocessor = TimeSeriesPreprocessor(
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=forecast_columns[7],
        observable_columns=observable_columns_list,
        scaling=True,
    )
    time_series_preprocessor = time_series_preprocessor.train(train_data)


    train_dataset = ForecastDFDataset(
        time_series_preprocessor.preprocess(train_data),
        id_columns=id_columns,
        timestamp_column="time",
        target_columns=[forecast_columns[7]],
        observable_columns=observable_columns_list,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )
    valid_dataset = ForecastDFDataset(
        time_series_preprocessor.preprocess(valid_data),
        id_columns=id_columns,
        timestamp_column="time",
        target_columns=[forecast_columns[7]],
        observable_columns=observable_columns_list,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )
    test_dataset = ForecastDFDataset(
        time_series_preprocessor.preprocess(test_data),
        id_columns=id_columns,
        timestamp_column="time",
        target_columns=[forecast_columns[7]],
        observable_columns=observable_columns_list,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )


    config = PatchTSTConfig(
        num_input_channels=len(forecast_columns),
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_length,
        prediction_length=forecast_horizon,
        random_mask_ratio=0.4,
        d_model=128,
        num_attention_heads=16,
        num_hidden_layers=3,
        ffn_dim=256,
        dropout=0.2,
        head_dropout=0.2,
        pooling_type=None,
        channel_attention=False,
        scaling="std",
        loss="mse",
        pre_norm=True,
        norm_type="batchnorm",
    )

    model = PatchTSTForPrediction(config).to(device)

    lossFunction = nn.MSELoss()

    optimiser = Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)



    # training_args = TrainingArguments(
    #     output_dir="./checkpoint/patchtst/electricity/pretrain/output/",
    #     overwrite_output_dir=True,
    #     # learning_rate=0.001,
    #     num_train_epochs=100,
    #     do_eval=True,
    #     evaluation_strategy="epoch",
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     dataloader_num_workers=num_workers,
    #     save_strategy="epoch",
    #     logging_strategy="epoch",
    #     save_total_limit=3,
    #     logging_dir="./checkpoint/patchtst/electricity/pretrain/logs/",  # Make sure to specify a logging directory
    #     load_best_model_at_end=True,  # Load the best model when training ends
    #     metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
    #     greater_is_better=False,  # For loss
    #     label_names=["future_values"],
    # )

    # # Create the early stopping callback
    # early_stopping_callback = EarlyStoppingCallback(
    #     early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
    #     early_stopping_threshold=0.0001,  # Minimum improvement required to consider as improvement
    # )

    # define trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=valid_dataset,
    #     callbacks=[early_stopping_callback],
    #     # compute_metrics=compute_metrics,
    # )

    # # pretrain
    # trainer.train()

    history = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'train_rmse': [],
    'val_rmse': [],
    'train_mape': [],
    'val_mape': []

            }

    epochs = 25
    for epoch in range(epochs):
        model.train()  
        epoch_loss = 0
        total_rmse = 0
        total_mape = 0
        num_batches = 0

        train_losses, train_rmses, train_mapes = [], [], []

        for batch in train_dataloader:
            inputs = batch['past_values'].to(device)
            targets = batch['future_values'].to(device)

            # output = model(inputs)
            # print(output)
            # print(output.__dict__)

            # print(inputs.shape)
            # print(targets.shape)

            optimiser.zero_grad()  
            predictions = model(inputs).prediction_outputs.flatten()  

            loss = lossFunction(predictions, targets.flatten()) 
            loss.backward() 
            optimiser.step()
            
            epoch_loss += loss.item()

            # # Convert predictions and targets back to original scale
            predictions_numpy = predictions.detach().cpu().numpy().reshape(-1, 1)
            y_batch_numpy = targets.detach().cpu().numpy().reshape(-1, 1)

            # predictions_original = scaler_y.inverse_transform(predictions_numpy).flatten()
            # y_batch_original = scaler_y.inverse_transform(y_batch_numpy).flatten()

            predictions_original = predictions_numpy.flatten()
            y_batch_original = y_batch_numpy.flatten()

            # Compute RMSE and MAPE
            rmse = np.sqrt(np.mean((predictions_original - y_batch_original) ** 2))
            
            # Avoid division by very small numbers
            mape = np.mean(np.abs((predictions_original - y_batch_original) / (y_batch_original + 1))) * 100

            total_rmse += rmse
            total_mape += mape
            num_batches += 1
        



        model.eval()
        val_loss = 0
        val_rmse = 0
        val_mape = 0
        val_batches = 0

        with torch.no_grad():  
            for batch in val_dataloader:

                val_inputs = batch['past_values'].to(device)
                val_targets = batch['future_values'].to(device)
                # X_val_batch = X_val_tensor[i:i + batch_size]
                # y_val_batch = y_val_tensor[i:i + batch_size]

                val_predictions = model(val_inputs).prediction_outputs.flatten()
                loss = lossFunction(val_predictions, val_targets.flatten())
                val_loss += loss.item()

                val_predictions_numpy = val_predictions.cpu().numpy().reshape(-1, 1)
                y_val_numpy = val_targets.cpu().numpy().reshape(-1, 1)

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

        history["train_loss"].append(epoch_loss/len(train_dataloader.dataset))
        history["train_rmse"].append(avg_rmse)
        history["train_mape"].append(avg_mape)

        history["val_loss"].append(avg_val_loss)
        history["val_rmse"].append(avg_val_rmse)
        history["val_mape"].append(avg_val_mape)



        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/len(train_dataloader.dataset):.6f}, Train RMSE: {avg_rmse:.4f}, Train MAPE: {avg_mape:.2f}% | "
            f"Val Loss: {avg_val_loss:.6f}, Val RMSE: {avg_val_rmse:.4f}, Val MAPE: {avg_val_mape:.2f}%")
        
    epochs_range = range(1, epochs + 1)

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history["val_rmse"], label="Val RMSE")
    plt.plot(epochs_range, history["train_rmse"], label="Train RMSE")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Training vs Validation RMSE")

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history["val_mape"], label="Val MAPE")
    plt.plot(epochs_range, history["train_mape"], label="Train MAPE")
    plt.xlabel("Epochs")
    plt.ylabel("MAPE")
    plt.legend()
    plt.title("Training vs Validation MAPE")

    plt.show()
