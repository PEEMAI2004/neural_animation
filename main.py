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
import networkx as nx
import matplotlib.pyplot as plt

def draw_neural_net(layer_sizes):
    '''
    Draw a neural network cartoon using matplotlib and networkx.
    
    :param layer_sizes: list of layer sizes, including input and output dimensionality
    '''
    node_spacing= 1
    # Create a new figure
    G = nx.DiGraph()
    for i in range(len(layer_sizes) - 1):       # loop through each layer
        for j in range(layer_sizes[i]):         # loop through each node in the layer
            for k in range(layer_sizes[i+1]):   # loop through each node in the next layer
                if i == len(layer_sizes) - 2:   # check if it's the last layer
                    G.add_edge((i,j), (i+1,k), color='g')  # add edge from node in layer i to node in layer i+1 with green color (hidden to output)
                elif i == 0:                    # check if it's the first layer
                    G.add_edge((i,j), (i+1,k), color='r')  # add edge from node in layer i to node in layer i+1 with red color (input to hidden)
                else:
                    G.add_edge((i,j), (i+1,k), color='b')  # add edge from node in layer i to node in layer i+1 with blue color (hidden to hidden)

    pos = {} # position of each node
    for i, layer_size in enumerate(layer_sizes): # loop through each layer
        layer_height = (layer_size - 1) / 2.0 # calculate the height of the layer
        for j in range(layer_size): # loop through each node in the layer
            pos[(i, j)] = [i, layer_height - j * node_spacing] # set position of each node based on layer and index of node in layer

    nx.draw(G, pos, with_labels=False, arrows=False, node_size=200, node_color='black', edge_color=[G[u][v]['color'] for u,v in G.edges()]) # draw the neural network with black node color
    plt.title('Neural Network Graph')  # set the title of the plot
    # plt.show() # display the plot of the neural network graph
    st.set_option('deprecation.showPyplotGlobalUse', False) # disable warning
    st.pyplot() # display the plot of the neural network graph 
    st.write('This is a graph of the neural network, the red edges are the edges from the input layer to the hidden layer, the blue edges are the edges from the hidden layer to the hidden layer, and the green edges are the edges from the hidden layer to the output layer. ')

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
st.write('''
## What is Neural Network ?

A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. In essence, neural networks are used to approximate functions that can depend on a large number of inputs and are generally unknown.

Neural networks are a subset of machine learning and are at the heart of deep learning algorithms. They are called "neural" because they are designed to mimic neurons in the human brain. A neuron takes inputs, does some processing, and produces one output. Similarly, a neural network takes a set of inputs, processes them through hidden layers using weights that are adjusted during training, and outputs a prediction representing the combined input signal.

The neural network in this app is being used to recognize handwritten digits, a classic problem in machine learning. The network is trained on a dataset of handwritten digits and their corresponding labels, and it learns to map the input images to the correct digit.
''')
st.subheader('This is a neural network that We made from scratch using Python and NumPy, Let\'s train a neural network to recognize handwritten digits!')

# User input for number of epochs and hidden layers
st.write('Select the number of epochs and hidden layers, More eppochs means more accurate but slower')
epochs = st.slider('Number of Epochs 10 power by', 2, 4, 3)
epochs = 10 ** epochs

# User input for learning rate
st.write('Select the learning rate, More learning rate means faster but less accurate')
learning_rate = st.slider('Learning Rate 10 power by', -6, 0, -2)
learning_rate = 10 ** learning_rate

# User input for number of hidden layers
st.write('Select the number of hidden layers, More hidden layers means more accurate but slower')
num_layers = st.slider('Number of Hidden Layers', 1, 9, 1) + 1

# fix image size and number of nodes in each hidden layer
image_size = 28
nodes = 10

# User input for choice of activation function
st.write('Select the activation function, Tanh is the default')
st.write('''
## Activation Functions

1. **Tanh (Hyperbolic Tangent):**
   - Outputs values between -1 and 1.
   - Symmetric around the origin.
   - Smooth and differentiable.
   - Often used in hidden layers of neural networks.

2. **ReLU (Rectified Linear Unit):**
   - Outputs the input directly if it is positive, otherwise outputs 0.
   - Computationally efficient and helps alleviate the vanishing gradient problem.
   - Introduces sparsity in the network by zeroing out negative values.
   - Commonly used in hidden layers of deep neural networks.

3. **Sigmoid:**
   - Outputs values between 0 and 1.
   - Maps the input to a probability-like output.
   - Useful for binary classification problems.
   - Suffers from the vanishing gradient problem for very large or small inputs.
   - Often used in the output layer for binary classification tasks.
''')


activation = st.selectbox('Select Activation Function', ['Tanh', 'ReLU', 'Sigmoid'])
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

    # Call the function with your layer sizes
    hidden_layers = [nodes] * (num_layers - 1)
    draw_neural_net([14] + hidden_layers + [1])

    # Visualize error using matplotlib
    st.write('error rate was calculate by mean squared error (MSE) and the error rate is the average of all the MSE') 
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pd.read_csv('error.csv')['Error'], 'r', label='Error Rate')
    ax.set_title('Error Rate')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error percentage')
    ax.legend()

    # Show error rate graph
    st.pyplot(fig)

    # Calculate accuracy of model
    st.write('Accuracy is the percentage of correct predictions')
    st.write('Accuracy = (number of correct predictions) / (total number of predictions)')
    accuracy = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)
        accuracy += (np.argmax(output) == y).mean()
    accuracy /= len(X)
    st.write('Avg. Accuracy: ', round(accuracy * 100, 2), '%')

    # Calculate loss of model
    st.write('Loss is the average of all the mean squared error (MSE)')
    st.write('Loss = (sum of all the MSE) / (total number of predictions)')
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
                st.text('There is a total of {} nodes in layer {}. There is a total of {} nodes in layer {}'.format(image_size ** 2, int(i/2), nodes, int(i/2+1)))
                st.text('So, there is a total of {} weights from layer {} to layer {}, \nbecause every nodes are connect together'.format(image_size ** 2 * nodes, int(i/2), int(i/2+1)))
                st.text('Weights are use for adjusting the output of each node in the next layer')

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
                st.text('There is a total of {} nodes in layer {}. There is a total of {} nodes in layer {}'.format(nodes, int(i/2), nodes, int(i/2+1)))
                st.text('So, there is a total of {} weights from layer {} to layer {}, \nbecause every nodes are connect together'.format(nodes ** 2, int(i/2), int(i/2+1)))
                st.text('Weights are use for adjusting the output of each node in the next layer')

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
                fig, ax = plt.subplots(figsize=(10, 5))
                # set title of each heatmap
                ax.set_title(f'Heatmap of Weights from layer {int(i/2)} to layer {int(i/2+1)}')
                ax.set_aspect('equal')  # Set aspect ratio to make the heatmap square
                ax = sns.heatmap(heatmap_data, cmap='coolwarm')
                st.pyplot(fig)

            # Show Biases as a barchart using matplotlib.pyplot
            st.subheader('Biases from layer {} to layer {}'.format(int(i/2), int(i/2+1)))
            st.text('There is a total of {} biases from layer {} to layer {}'.format(nodes, int(i/2), int(i/2+1)))
            st.text('Biases are use for adjusting the output of each node in the next layer')
            data = network[i].bias
            data = data.reshape(nodes, 1)  # Split 10 into 10 1x1 matrices
            data = data.tolist()
            data = [j[0] for j in data]  # Convert to 1D list

            # Show Biases as a barchart using matplotlib.pyplot
            fig, ax = plt.subplots(figsize=(10, 5))
            # set title of each barchart
            ax.set_title('Biases from layer {} to layer {}'.format(int(i/2), int(i/2+1)))
            bars = ax.bar([str(j) for j in range(nodes)], data)

            # Color positive values blue and negative values red
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    bar.set_color('blue')
                else:
                    bar.set_color('red')

                # Add labels to each bar
                ax.annotate(f'{round(height, 2)}', xy=(bar.get_x() + bar.get_width() / 2, height),
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