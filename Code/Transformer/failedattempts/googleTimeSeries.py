import timesfm

# Loading the timesfm-2.0 checkpoint:
# For PAX


# For Torch
model = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="cpu",
          per_core_batch_size=32,
          horizon_len=1,
          num_layers=50,
          use_positional_embedding=False,
          context_len=2048,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
  )


import pandas as pd
import numpy as np
from collections import defaultdict




df = pd.read_csv('https://datasets-nixtla.s3.amazonaws.com/EPF_FR_BE.csv')
df['ds'] = pd.to_datetime(df['ds'])



# Data pipelining
def get_batched_data_fn(
    batch_size: int = 128, 
    context_len: int = 120, 
    horizon_len: int = 24,
):
  examples = defaultdict(list)

  num_examples = 0
  for country in ("FR", "BE"):
    sub_df = df[df["unique_id"] == country]
    for start in range(0, len(sub_df) - (context_len + horizon_len), horizon_len):
      num_examples += 1
      examples["country"].append(country)
      examples["inputs"].append(sub_df["y"][start:(context_end := start + context_len)].tolist())
      examples["gen_forecast"].append(sub_df["gen_forecast"][start:context_end + horizon_len].tolist())
      examples["week_day"].append(sub_df["week_day"][start:context_end + horizon_len].tolist())
      examples["outputs"].append(sub_df["y"][context_end:(context_end + horizon_len)].tolist())
  
  def data_fn():
    for i in range(1 + (num_examples - 1) // batch_size):
      yield {k: v[(i * batch_size) : ((i + 1) * batch_size)] for k, v in examples.items()}
  
  return data_fn



# Define metrics
def mse(y_pred, y_true):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  return np.mean(np.square(y_pred - y_true), axis=1, keepdims=True)

def mae(y_pred, y_true):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  return np.mean(np.abs(y_pred - y_true), axis=1, keepdims=True)





import time

# Benchmark
batch_size = 128
context_len = 120
horizon_len = 24
input_data = get_batched_data_fn(batch_size = 128)
metrics = defaultdict(list)


for i, example in enumerate(input_data()):
  raw_forecast, _ = model.forecast(
      inputs=example["inputs"], freq=[0] * len(example["inputs"])
  )
  start_time = time.time()
  # Forecast with covariates
  # Output: new forecast, forecast by the xreg
  cov_forecast, ols_forecast = model.forecast_with_covariates(  
      inputs=example["inputs"],
      dynamic_numerical_covariates={
          "gen_forecast": example["gen_forecast"],
      },
      dynamic_categorical_covariates={
          "week_day": example["week_day"],
      },
      static_numerical_covariates={},
      static_categorical_covariates={
          "country": example["country"]
      },
      freq=[0] * len(example["inputs"]),
      xreg_mode="xreg + timesfm",              # default
      ridge=0.0,
      force_on_cpu=False,
      normalize_xreg_target_per_input=True,    # default
  )
  print(
      f"\rFinished batch {i} linear in {time.time() - start_time} seconds",
      end="",
  )
  metrics["eval_mae_timesfm"].extend(
      mae(raw_forecast[:, :horizon_len], example["outputs"])
  )
  metrics["eval_mae_xreg_timesfm"].extend(mae(cov_forecast, example["outputs"]))
  metrics["eval_mae_xreg"].extend(mae(ols_forecast, example["outputs"]))
  metrics["eval_mse_timesfm"].extend(
      mse(raw_forecast[:, :horizon_len], example["outputs"])
  )
  metrics["eval_mse_xreg_timesfm"].extend(mse(cov_forecast, example["outputs"]))
  metrics["eval_mse_xreg"].extend(mse(ols_forecast, example["outputs"]))

print()

for k, v in metrics.items():
  print(f"{k}: {np.mean(v)}")

