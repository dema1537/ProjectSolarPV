import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import os




def Evalutaion(predictions, cloud_cover, scaler_y, ytest, history, filepath, title):


    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)    
        print(f"Directory '{filepath}' created successfully.")

    test_predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))

    fig, ax2 = plt.subplots()
    ax1 = ax2.twinx()

    ax1.bar(range(len(cloud_cover)), cloud_cover, label='Cloud Cover', alpha=0.5, width=1.0)
    ax1.set_ylabel("Cloud Cover (%)")


    ax2.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual', color='blue')
    ax2.plot(test_predictions, label='Predicted', color='orange')
    ax2.set_ylabel("PV", color='blue') 
    ax2.tick_params(axis='y', labelcolor='blue')

    ax2.legend(loc='upper left')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    ax1.set_xlabel("Time Steps")

    plt.title("Actual and Predicted Values for " + title)

    # ax1.set_xlim(75, 150)
    plt.savefig(filepath + title + '.png')
    #plt.show()


