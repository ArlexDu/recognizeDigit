import CNN
from CNN import Network
from CNN import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer,ReLU
import matplotlib.pyplot as plt
import numpy as np
# load data from MINST
training_data, validation_data, test_data = CNN.load_data_shared()
mini_batch_size = 10
load_file = None#'F:\\Projects\\Python\\recognizeDigit\\train\\cnn_network.json'
# initial network
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size,load_file)
# start training
evaluation_cost, evaluation_accuracy,training_cost, training_accuracy = net.SGD(training_data, 30, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)

tx = len(training_cost)
x = np.arange(0,tx,1)
# plot cost chart
plt.figure(1)
plt.title('The cost of training')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.plot(x, training_cost,label='train data')
plt.plot(x, evaluation_cost,label='evaluation data')
plt.legend(loc='upper right')
plt.show()

# plot accuracy chart
plt.figure(2)
plt.title('The accuracy of training')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(x, training_accuracy,label='train data')
plt.plot(x, evaluation_accuracy,label='evaluation data')
plt.legend(loc='lower right')
plt.show()
# save the weights and biases
net.save("cnn_network.json")