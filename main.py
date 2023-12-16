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
import time

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

# User input for number of hidden layers
num_layers = st.slider('Number of Hidden Layers', 1, 9, 1) + 1

image_size = 28
# User input for number of nodes
# nodes = st.slider('Number of Nodes', 10, 16, 10)
nodes = 10

# User input for choice of activation function
activation = st.selectbox('Activation Function', ['Tanh', 'ReLU', 'Sigmoid'])
if activation == 'Tanh':
    network = [Dense(image_size ** 2, nodes), Tanh()]
    for _ in range(num_layers - 1):
        network.append(Dense(nodes, nodes))
        network.append(Tanh())
elif activation == 'Sigmoid':
    network = [Dense(image_size ** 2, nodes), Sigmoid()]
    for _ in range(num_layers - 1):
        network.append(Dense(nodes, nodes))
        network.append(Sigmoid())
else:
    network = [Dense(image_size ** 2, nodes), ReLU()]
    for _ in range(num_layers - 1):
        network.append(Dense(nodes, nodes))
        network.append(ReLU())

# Add trend button to start
if st.button('Start'):
    start_time = time.time()
    # write error to .csv file
    with open('error.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Error'])  # Write header row
        # Train model
        for epoch in range(epochs):# loop through each epoch
            err = 0
            for x, y in zip(X, Y): # loop through each image and label
                output = x
                # forward propagation
                for layer in network: # loop through each layer
                    output = layer.forward(output)
                # calculate error
                y_true = np.eye(nodes)[y].T.reshape(-1, 1)
                err += mse(y_true, output) 
                grad = mse_prime(y_true, output) 
                # back propagation
                for layer in reversed(network): # loop through each layer in reverse order
                    grad = layer.backward(grad, learning_rate)
            # calculate error rate
            err /= len(X)
            print(f"epoch {epoch} error = {err*100}%")
            # write error to .csv file with epoch number
            writer.writerow([epoch, err*100])
        # close .csv file
        csvfile.close()   

    # Visualize accuracy and loss using matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pd.read_csv('error.csv')['Error'], 'r', label='Error Rate')
    ax.set_title('Error Rate')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error percentage')
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
    st.write('Avg. Accuracy: ', round(accuracy * 100, 2), '%')

    # Calculate loss of model
    loss = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)
        y_true = np.eye(nodes)[y].T.reshape(-1, 1)
        loss += mse(y_true, output)
    loss /= len(X)
    st.write('Loss: ', round(loss, 2))

    # # Show table of weights and biases
    # col1, col2 = st.columns([10,1])
    # col1.header('Weights')
    # col1.write(network[0].weight)
    # col2.header('Biases')
    # col2.write(network[0].bias)

    # Show Weights as a heatmaps using matplotlib.pyplot from layer to node
    # loop through every layer
    for i, layer in enumerate(network):
        if hasattr(layer, 'weight'):
            st.subheader('Heatmap of Weights from layer {} to layer {}'.format(int(i/2), int(i/2+1)))
            if i == 0: # If first layer
                data = layer.weight
                data = data.reshape(nodes, image_size, image_size)  # Split data into nodes * image_size * image_size matrices
                data = data.tolist()
                # create heatmap for 1st layer to 2nd layer
                fig, axs = plt.subplots(2, 5, figsize=(20, 10))
                for j in range(nodes):
                    heatmap_data = pd.DataFrame(data[j])
                    heatmap_data.columns = [str(k) for k in range(28)]  # Set column names as string numbers
                    heatmap_data.index = [str(k) for k in range(28)]  # Set index names as string numbers
                    heatmap_data_list = heatmap_data.values.tolist()
                    heatmap_data_list = [[k, l, heatmap_data_list[k][l]] for k in range(28) for l in range(28)]  # Adjust the range to 28

                    # Show Weights as square heatmaps using matplotlib.pyplot
                    ax = axs[j % 2, j // 2]
                    ax.set_title(f'layer {i} to node {j} in layer {i+1}')
                    ax.set_aspect('equal')  # Set aspect ratio to make the heatmap square
                    ax = sns.heatmap(heatmap_data, cmap='coolwarm', ax=ax, cbar=False)  # Hide color palette
                # Add shared colorbar on RHS
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Define cbar_ax
                fig.colorbar(ax.collections[0], cax=cbar_ax)

                st.pyplot(fig)
                
            else: # If not first layer, show heat map size of nodes * nodes
                data = layer.weight
                data = data.reshape(nodes, nodes)
                data = data.tolist()
                # create heatmap for Nnd layer to (N+1)rd layer
                heatmap_data = pd.DataFrame(data)
                heatmap_data.columns = [str(k) for k in range(nodes)]
                heatmap_data.index = [str(k) for k in range(nodes)]
                heatmap_data_list = heatmap_data.values.tolist()
                heatmap_data_list = [[k, l, heatmap_data_list[k][l]] for k in range(nodes) for l in range(nodes)]

                # Show Weights as a heatmaps using matplotlib.pyplot
                fig, ax = plt.subplots(figsize=(10, 10))
                # set title of each heatmap
                ax.set_title(f'Heatmap of Weights from layer {int(i/2)} to layer {int(i/2+1)}')
                ax = sns.heatmap(heatmap_data, cmap='coolwarm')
                st.pyplot(fig)

            # Show Biases as a barchart using matplotlib.pyplot
            st.subheader('Biases from layer {} to layer {}'.format(int(i/2), int(i/2+1)))
            data = network[i].bias
            data = data.reshape(nodes, 1)  # Split 10 into 10 1x1 matrices
            data = data.tolist()
            data = [j[0] for j in data]  # Convert to 1D list

            # Show Biases as a barchart using matplotlib.pyplot
            fig, ax = plt.subplots(figsize=(10, 5))
            # set title of each barchart
            ax.set_title('Biases from layer {} to layer {}'.format(int(i/2), int(i/2+1)))
            bars = ax.bar([str(j) for j in range(nodes)], data)

            # Add labels to each bar
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom')

            st.pyplot(fig)


        print(i)


    print("-" * 30 + "after trained" + "-" * 30)
    for x, y in list(zip(X, Y))[:20]:
        output = x
        for layer in network:
            output = layer.forward(output)

        print(f"actual y = {y}")
        print(f"prediction = {np.argmax(output)}")
        print("-" * 50)

    # Show time taken to train
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")