import matplotlib.pyplot as plt
import numpy as np

# Your calculated values
avg_test_rmse = 1.2345
avg_test_loss = 0.005678
avg_test_mape = 5.67

# Prepare the data for plotting
metrics = ['RMSE', 'Loss', 'MAPE (%)']
values = [avg_test_rmse, avg_test_loss, avg_test_mape]

# Create a simple bar plot
plt.figure(figsize=(6, 4))  # Adjust figure size as needed
plt.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])

# Add labels and title
plt.ylabel('Value')
plt.title('Test Performance Metrics')

# Add the numerical values on top of the bars
for i, v in enumerate(values):
    plt.text(i, v + 0.05, f'{v:.4f}' if metrics[i] != 'MAPE (%)' else f'{v:.2f}', ha='center', va='bottom')

# Adjust y-axis limits for better visualization
plt.ylim(0, max(values) * 1.2)

# Remove spines for a cleaner look (optional)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Save the plot as an image
plt.savefig('test_metrics.png')  # You can choose other formats like .jpg, .pdf, etc.

# Optionally, display the plot
plt.show()

print("----\n\n\n")
print(f"Test Loss: {avg_test_loss:.6f}, Test RMSE: {avg_test_rmse:.4f}, Test MAPE: {avg_test_mape:.2f}%")
print("\nTest metrics saved as 'test_metrics.png'")