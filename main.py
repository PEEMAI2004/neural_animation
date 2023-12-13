from neural_network import *
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyecharts import options as opts
from pyecharts.charts import HeatMap

def mse(y, y_pred):
    return np.mean(np.power(y - y_pred, 2))


def mse_prime(y, y_pred):
    return 2 * (y_pred - y) / np.size(y)


data = pd.read_csv("./train.csv").values[:100]
np.random.shuffle(data)

X = data[:, 1:].reshape(-1, 784, 1) / 255
Y = data[:, 0].reshape(-1, 1, 1)
print(np.unique(Y))

st.title('My Neural Network')
st.subheader('Hello')

# User input for number of epochs and hidden layers
epochs = st.slider('Number of Epochs 10 power by', 2, 4, 3)
epochs = 10 ** epochs
# User input for learning rate
learning_rate = st.slider('Learning Rate 10 power by', -6, 0, -2)
learning_rate = 10 ** learning_rate
# User input for size of image
# image_size = st.slider('Image Size', 1, 28, 28)
image_size = 28

# User input for choice of activation function
activation = st.selectbox('Activation Function', ['Tanh', 'ReLU', 'Sigmoid'])
if activation == 'Tanh':
    network = [Dense(image_size ** 2, 10), Tanh(), Dense(10, 10), Tanh()]
elif activation == 'Sigmoid':
    network = [Dense(image_size ** 2, 10), Sigmoid(), Dense(10, 10), Sigmoid()]
else:
    network = [Dense(image_size ** 2, 10), ReLU(), Dense(10, 10), ReLU()]

# Add trend button to start
if st.button('Start'):
    # write error to .csv file
    with open('error.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Error'])  # Write header row
        for epoch in range(epochs):
            err = 0
            for x, y in zip(X, Y):
                output = x
                for layer in network:
                    output = layer.forward(output)

                y_true = np.eye(10)[y].T.reshape(-1, 1)
                err += mse(y_true, output)
                grad = mse_prime(y_true, output)

                for layer in reversed(network):
                    grad = layer.backward(grad, learning_rate)

            err /= len(X)
            print(f"epoch {epoch} error = {err}")
            # write error to .csv file with epoch number
            writer.writerow([epoch, err])
        # close .csv file
        csvfile.close()   

    # Visualize accuracy and loss using matplotlib
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pd.read_csv('error.csv')['Error'], 'r', label='Error Rate')
    ax.set_title('Error Rate')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error')
    ax.legend()

    # Show loss graph
    st.pyplot(fig)

    # Calculate accuracy of model
    accuracy = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)
        accuracy += (np.argmax(output) == y).mean()
    accuracy /= len(X)
    st.write('Accuracy: ', round(accuracy * 100, 2), '%')

    # Calculate loss of model
    loss = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)
        y_true = np.eye(10)[y].T.reshape(-1, 1)
        loss += mse(y_true, output)
    loss /= len(X)
    st.write('Loss: ', round(loss, 2))

    # # Show table of weights and biases
    # col1, col2 = st.columns([10,1])
    # col1.header('Weights')
    # col1.write(network[0].weight)
    # col2.header('Biases')
    # col2.write(network[0].bias)

    # Show Biases as a barchart using matplotlib.pyplot
    st.title('Biases')
    data = network[0].bias
    data = data.reshape(10, 1)  # Split 10 into 10 1x1 matrices
    data = data.tolist()
    data = [i[0] for i in data]  # Convert to 1D list

    # Show Biases as a barchart using matplotlib.pyplot
    fig, ax = plt.subplots(figsize=(10, 10))
    # set title of each barchart
    ax.set_title('Biases from input layer to first layer')
    bars = ax.bar([str(i) for i in range(10)], data)

    # Add labels to each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom')

    st.pyplot(fig)
    

    # Show Weights as a heatmaps using matplotlib.pyplot
    st.title('Heatmap of Weights')
    data = network[0].weight
    data = data.reshape(10, 28, 28)  # Split 7840 into 10 28x28 matrices
    data = data.tolist()

    for i in range(10):
        heatmap_data = pd.DataFrame(data[i])
        heatmap_data.columns = [str(i) for i in range(28)]  # Set column names as string numbers
        heatmap_data.index = [str(i) for i in range(28)]  # Set index names as string numbers
        heatmap_data_list = heatmap_data.values.tolist()
        heatmap_data_list = [[i, j, heatmap_data_list[i][j]] for i in range(28) for j in range(28)]  # Adjust the range to 28

        # Show Weights as a heatmaps using matplotlib.pyplot
        fig, ax = plt.subplots(figsize=(10, 10))
        # set title of each heatmap
        ax.set_title(f'Heatmap of Weights from first layer to node {i} in second layer')
        ax = sns.heatmap(heatmap_data, cmap='coolwarm')
        st.pyplot(fig)    


    print("-" * 30 + "after trained" + "-" * 30)
    for x, y in list(zip(X, Y))[:20]:
        output = x
        for layer in network:
            output = layer.forward(output)

        print(f"actual y = {y}")
        print(f"prediction = {np.argmax(output)}")
        print("-" * 50)
