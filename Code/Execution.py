#training files:

#1 Code/CNN/CNNmodel.py
#2 Code/CNNLSTM/CNNLSTMModel.py
#3 Code/LSTM/LSTMmodel.py
#4 Code/Transformer/ForecastingModel.py
#5 Code/Transformer/TransformerModel.py

#TL files:

#1 Code/CNN/transferLearningCNN.py
#2 Code/CNNLSTM/transferLearningCNNLSTM.py
#3 Code/LSTM/transferLearningLSTM.py
#4 Code/Transformer/transferLearningForecasting.py
#5 Code/Transformer/transferLearningTransformer.py


import os
import subprocess



if __name__ == "__main__":



    #CNN
    try:
        result = subprocess.run(['python', "Code/CNN/CNNModel.py"], capture_output=True, text=True, check=True)
        print(result.stdout)  # Print the output of the executed script
    except subprocess.CalledProcessError as e:
        print(f"Error executing CNN: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: Python interpreter not found.")
    print("-" * 20)

    try:
        result = subprocess.run(['python', "Code/CNN/transferLearningCNN.py"], capture_output=True, text=True, check=True)
        print(result.stdout)  # Print the output of the executed script
    except subprocess.CalledProcessError as e:
        print(f"Error executing tlCNN: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: Python interpreter not found.")
    print("-" * 20)





    #CNNLSTM
    try:
        result = subprocess.run(['python', "Code/CNNLSTM/CNNLSTMModel.py"], capture_output=True, text=True, check=True)
        print(result.stdout)  # Print the output of the executed script
    except subprocess.CalledProcessError as e:
        print(f"Error executing CNNLSTM: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: Python interpreter not found.")
    print("-" * 20)

    try:
        result = subprocess.run(['python', "Code/CNNLSTM/transferLearningCNNLSTM.py"], capture_output=True, text=True, check=True)
        print(result.stdout)  # Print the output of the executed script
    except subprocess.CalledProcessError as e:
        print(f"Error executing tlCNNLSTM: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: Python interpreter not found.")
    print("-" * 20)


    #LSTM
    try:
        result = subprocess.run(['python', "Code/LSTM/LSTMmodel.py"], capture_output=True, text=True, check=True)
        print(result.stdout)  # Print the output of the executed script
    except subprocess.CalledProcessError as e:
        print(f"Error executing LSTM: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: Python interpreter not found.")
    print("-" * 20)

    try:
        result = subprocess.run(['python', "Code/LSTM/transferLearningLSTM.py"], capture_output=True, text=True, check=True)
        print(result.stdout)  # Print the output of the executed script
    except subprocess.CalledProcessError as e:
        print(f"Error executing tlLSTM: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: Python interpreter not found.")
    print("-" * 20)


    #Forecast
    try:
        result = subprocess.run(['python', "Code/Transformer/ForecastingModel.py"], capture_output=True, text=True, check=True)
        print(result.stdout)  # Print the output of the executed script
    except subprocess.CalledProcessError as e:
        print(f"Error executing FORECASTING: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: Python interpreter not found.")
    print("-" * 20)

    try:
        result = subprocess.run(['python', "Code/Transformer/transferLearningForecasting.py"], capture_output=True, text=True, check=True)
        print(result.stdout)  # Print the output of the executed script
    except subprocess.CalledProcessError as e:
        print(f"Error executing tlFORECASTING: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: Python interpreter not found.")
    print("-" * 20)


    #Transformer
    try:
        result = subprocess.run(['python', "Code/Transformer/TransformerModel.py"], capture_output=True, text=True, check=True)
        print(result.stdout)  # Print the output of the executed script
    except subprocess.CalledProcessError as e:
        print(f"Error executing TRANSFORMER: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: Python interpreter not found.")
    print("-" * 20)

    try:
        result = subprocess.run(['python', "Code/Transformer/transferLearningTransformer.py"], capture_output=True, text=True, check=True)
        print(result.stdout)  # Print the output of the executed script
    except subprocess.CalledProcessError as e:
        print(f"Error executing tlTRANSFORMER: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: Python interpreter not found.")
    print("-" * 20)





    print("Finished executing all Python files.")