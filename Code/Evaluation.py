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
    plt.savefig(filepath + "testgraph" + '.png')
    #plt.show()

    # fig, ax2 = plt.subplots()
    # ax1 = ax2.twinx()

    # ax1.bar(range(len(cloud_cover)), cloud_cover, label='Cloud Cover', alpha=0.5, width=1.0) # Added alpha for better visibility
    # ax1.set_ylabel("Cloud Cover (%)")



    # ax2.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual', color='blue')
    # ax2.plot(test_predictions, label='Predicted', color='orange')
    # ax2.set_ylabel("PV", color='blue') # Label the second y-axis
    # ax2.tick_params(axis='y', labelcolor='blue')

    # ax2.legend(loc='upper left')

    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # ax1.set_xlabel("Time Steps")

    # plt.title("Actual and Predicted Values on Sunny days")


    # ax1.set_xlim(790, 840)
    # plt.savefig(filepath + '2DaySunny.png')
    # #plt.show()

    # fig, ax2 = plt.subplots()
    # ax1 = ax2.twinx()

    # ax1.bar(range(len(cloud_cover)), cloud_cover, label='Cloud Cover', alpha=0.5, width=1.0) # Added alpha for better visibility
    # ax1.set_ylabel("Cloud Cover (%)")



    # ax2.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual', color='blue')
    # ax2.plot(test_predictions, label='Predicted', color='orange')
    # ax2.set_ylabel("PV", color='blue') # Label the second y-axis
    # ax2.tick_params(axis='y', labelcolor='blue')

    # ax2.legend(loc='upper left')

    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # ax1.set_xlabel("Time Steps")

    # plt.title("Actual and Predicted Values on Full Cloud Cover")

    # ax1.set_xlim(2040, 2110)
    # ax2.set_ylim(-1000, 22500)
    # plt.savefig(filepath + '3DayFullCloud.png')
    # #plt.show()




    # epochs_range = range(1, len(history["val_loss"]) + 1)
    # plt.figure(figsize=(12, 4))

    # plt.subplot(1, 3, 1)
    # plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    # plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.title("Training vs Validation Loss")

    # plt.subplot(1, 3, 2)
    # plt.plot(epochs_range, history["val_rmse"], label="Val RMSE")
    # plt.plot(epochs_range, history["train_rmse"], label="Train RMSE")
    # plt.xlabel("Epochs")
    # plt.ylabel("RMSE")
    # plt.legend()
    # plt.title("Training vs Validation RMSE")

    # plt.subplot(1, 3, 3)
    # plt.plot(epochs_range, history["val_mape"], label="Val MAPE")
    # plt.plot(epochs_range, history["train_mape"], label="Train MAPE")
    # plt.xlabel("Epochs")
    # plt.ylabel("MAPE")
    # plt.legend()
    # plt.title("Training vs Validation MAPE")


    # plt.savefig(filepath + 'TrainingValidationGraphs.png')
    #plt.show()
    #plt.show()


# def Evalutaion(predictions, cloud_cover, scaler_y, ytest, history, filepath, title):

#     test_predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))

#     fig, ax2 = plt.subplots()
#     ax1 = ax2.twinx()

#     ax1.bar(range(len(cloud_cover)), cloud_cover, label='Cloud Cover', alpha=0.5, width=1.0)
#     ax1.set_ylabel("Cloud Cover (%)")


#     ax2.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual', color='blue')
#     ax2.plot(test_predictions, label='Predicted', color='orange')
#     ax2.set_ylabel("PV", color='blue') 
#     ax2.tick_params(axis='y', labelcolor='blue')

#     ax2.legend(loc='upper left')

#     lines, labels = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines + lines2, labels + labels2, loc='upper left')

#     ax1.set_xlabel("Time Steps")

#     plt.title("Actual and Predicted Values on Mixed Cloud Cover")

#     ax1.set_xlim(75, 150)
#     plt.savefig(filepath + '3DayMixedClouds.png')
#     #plt.show()

#     fig, ax2 = plt.subplots()
#     ax1 = ax2.twinx()

#     ax1.bar(range(len(cloud_cover)), cloud_cover, label='Cloud Cover', alpha=0.5, width=1.0) # Added alpha for better visibility
#     ax1.set_ylabel("Cloud Cover (%)")



#     ax2.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual', color='blue')
#     ax2.plot(test_predictions, label='Predicted', color='orange')
#     ax2.set_ylabel("PV", color='blue') # Label the second y-axis
#     ax2.tick_params(axis='y', labelcolor='blue')

#     ax2.legend(loc='upper left')

#     lines, labels = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines + lines2, labels + labels2, loc='upper left')

#     ax1.set_xlabel("Time Steps")

#     plt.title("Actual and Predicted Values on Sunny days")


#     ax1.set_xlim(790, 840)
#     plt.savefig(filepath + '2DaySunny.png')
#     #plt.show()

#     fig, ax2 = plt.subplots()
#     ax1 = ax2.twinx()

#     ax1.bar(range(len(cloud_cover)), cloud_cover, label='Cloud Cover', alpha=0.5, width=1.0) # Added alpha for better visibility
#     ax1.set_ylabel("Cloud Cover (%)")



#     ax2.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual', color='blue')
#     ax2.plot(test_predictions, label='Predicted', color='orange')
#     ax2.set_ylabel("PV", color='blue') # Label the second y-axis
#     ax2.tick_params(axis='y', labelcolor='blue')

#     ax2.legend(loc='upper left')

#     lines, labels = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines + lines2, labels + labels2, loc='upper left')

#     ax1.set_xlabel("Time Steps")

#     plt.title("Actual and Predicted Values on Full Cloud Cover")

#     ax1.set_xlim(2040, 2110)
#     ax2.set_ylim(-1000, 22500)
#     plt.savefig(filepath + '3DayFullCloud.png')
#     #plt.show()




#     epochs_range = range(1, len(history["val_loss"]) + 1)
#     plt.figure(figsize=(12, 4))

#     plt.subplot(1, 3, 1)
#     plt.plot(epochs_range, history["val_loss"], label="Val Loss")
#     plt.plot(epochs_range, history["train_loss"], label="Train Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.title("Training vs Validation Loss")

#     plt.subplot(1, 3, 2)
#     plt.plot(epochs_range, history["val_rmse"], label="Val RMSE")
#     plt.plot(epochs_range, history["train_rmse"], label="Train RMSE")
#     plt.xlabel("Epochs")
#     plt.ylabel("RMSE")
#     plt.legend()
#     plt.title("Training vs Validation RMSE")

#     plt.subplot(1, 3, 3)
#     plt.plot(epochs_range, history["val_mape"], label="Val MAPE")
#     plt.plot(epochs_range, history["train_mape"], label="Train MAPE")
#     plt.xlabel("Epochs")
#     plt.ylabel("MAPE")
#     plt.legend()
#     plt.title("Training vs Validation MAPE")


#     plt.savefig(filepath + 'TrainingValidationGraphs.png')
#     #plt.show()
#     #plt.show()



