from huggingface_hub import hf_hub_download
import torch
from transformers import InformerModel, InformerConfig

# Download the dataset (this part remains the same as you need the data)
file = hf_hub_download(
    repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

# Create a default configuration for the Informer model
config = InformerConfig()

# Initialize the InformerModel from the configuration (no pretrained weights)
model = InformerModel(config)

# Now you can use the model for training with your data
outputs = model(
    past_values=batch["past_values"],
    past_time_features=batch["past_time_features"],
    past_observed_mask=batch["past_observed_mask"],
    static_categorical_features=batch["static_categorical_features"],
    static_real_features=batch["static_real_features"],
    future_values=batch["future_values"],
    future_time_features=batch["future_time_features"],
)
last_hidden_state = outputs.last_hidden_state

print(f"Model initialized without pretrained weights: {model}")