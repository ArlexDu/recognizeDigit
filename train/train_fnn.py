import mnist_loader
import FNN

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = FNN.Network([784,40,10])

net.SGD(training_data,60,10,0.1,lmbda=5.0,evaluation_data=validation_data,monitor_evaluation_accuracy=True,monitor_evaluation_cost=True,monitor_training_accuracy=True,monitor_training_cost=True,early_stopping_n=10)

net.save("fnn_network.json")