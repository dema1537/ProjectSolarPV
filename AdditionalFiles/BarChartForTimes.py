import matplotlib.pyplot as plt
import numpy as np


categories = ['CNN', 'CCN-LSTM', 'LSTM', 'Simple Transformer', 'Forecast Transformer']
values = [1750.992, 3310.210 , 1808.551, 25143.434, 50427.230]

colors = ['blue', 'green', 'red', 'purple', 'orange']
plt.figure(figsize=(9, 4))

plt.bar(categories, values, color=colors)

plt.xlabel('Models', fontweight='bold')
plt.ylabel('Time training (seconds)', fontweight='bold')
plt.title('Time to train each model')

plt.tight_layout()
plt.savefig('AdditionalFiles/TrainingTimesGraph.png')



categories = ['CNN', 'LSTM', 'CNN-LSTM', 'Simple Transformer', 'Forecast Transformer']
values = [14.030, 14.591 , 4.212, 92.911, 119.423]

colors = ['blue', 'green', 'red', 'purple', 'orange']
plt.figure(figsize=(9, 4))

plt.bar(categories, values, color=colors)

plt.xlabel('Models', fontweight='bold')
plt.ylabel('Time training (seconds)', fontweight='bold')
plt.title('Inference time for each model')

plt.tight_layout()
plt.savefig("AdditionalFiles/InferenceTimesGraph.png")