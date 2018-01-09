import mnist_loader
import FNN
import matplotlib.pyplot as plt
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = FNN.Network([784,40,10])

evaluation_cost, evaluation_accuracy,training_cost, training_accuracy = net.SGD(training_data,60,10,0.1,lmbda=5.0,evaluation_data=validation_data,monitor_evaluation_accuracy=True,monitor_evaluation_cost=True,monitor_training_accuracy=True,monitor_training_cost=True,early_stopping_n=10)

tx = len(training_cost)
x = np.arange(0,tx,1)

plt.figure(1)
plt.title('The cost of training')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.plot(x, training_cost,label='train data')
plt.plot(x, evaluation_cost,label='evaluation data')
plt.legend(loc='upper right')
plt.show()

plt.figure(2)
plt.title('The accuracy of training')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(x, training_accuracy,label='train data')
plt.plot(x, evaluation_accuracy,label='evaluation data')
plt.legend(loc='lower right')
plt.show()

# net.save("fnn_network.json")