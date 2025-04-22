import matplotlib.pyplot as plt
import numpy as np


categories = ['CNN', 'CNN-LSTM', 'LSTM', 'Simple Transformer', 'Full Transformer']
values = [1750.992, 3310.210 , 1808.551, 25143.434, 50427.230]

colors = ['blue', 'green', 'red', 'purple', 'orange']
plt.figure(figsize=(9, 4))

plt.bar(categories, values, color=colors)

plt.xlabel('Models')
plt.ylabel('Time training (seconds)')
plt.title('Time to train each model')

plt.tight_layout()
plt.savefig('AdditionalFiles/TrainingTimesGraph.png')



categories = ['CNN', 'CNN-LSTM', 'LSTM', 'Simple Transformer', 'Full Transformer']
values = [14.030, 14.591 , 4.212, 92.911, 119.423]

colors = ['blue', 'green', 'red', 'purple', 'orange']
plt.figure(figsize=(9, 4))

plt.bar(categories, values, color=colors)

plt.xlabel('Models')
plt.ylabel('Time training (seconds)')
plt.title('Inference time for each model')

plt.tight_layout()
plt.savefig("AdditionalFiles/InferenceTimesGraph.png")