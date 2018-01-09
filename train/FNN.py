"""FNN.py 
~~~~~~~~~~~~~~
implementing the stochastic gradient descent learning algorithm for a 
feedforward neural network.Improvements include the addition of the 
cross-entropy cost function,log-likelihood cost function, regularization, and 
better initialization of network weights and softmax.
"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np


#### Define the quadratic and cross-entropy cost functions

class LogLikeCost(object):

    @staticmethod
    def fn(a, y):
        # return the cost
        return np.sum(np.nan_to_num(-y*np.log(a)))

    @staticmethod
    def delta(z, a, y):
        # return the cost derivative
        return (a-y)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        # return the cost
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        # return the cost derivative
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes,file=None):
        # The list ``sizes`` contains the number of neurons in the respective
        # layers of the network.
        self.num_layers = len(sizes)
        self.sizes = sizes
        if file!=None:
            self.loadParams(file)
        else:
            self.default_weight_initializer()
        self.cost= CrossEntropyCost

    def default_weight_initializer(self):
        # Initialize each weight using a Gaussian distribution with mean 0
        # and standard deviation 1
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        # Return the output of the network if `a` is input.
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a)+b)
        b = self.biases[-1]
        w = self.weights[-1]
        a = softmax(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n = 0):

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # early stopping functionality:
        best_accuracy=0
        no_accuracy_change=0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            print("Epoch %s training complete" % j)

            # the training set cost
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            # the training set accuracy
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy/n)
                print("Accuracy on training data: %f" % (accuracy/n))
            # the evaluation set cost
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            # the evaluation set accuracy
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy/n_data)
                print("Accuracy on evaluation data: %f" % (accuracy/n_data))

            # Early stopping:
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                    print("Early-stopping: Best so far {}".format(best_accuracy/n_data))
                else:
                    no_accuracy_change += 1

                if (no_accuracy_change == early_stopping_n):
                    print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

        return evaluation_cost, evaluation_accuracy,training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        # Update the network's weights and biases by applying gradient
        # descent using backpropagation to a single mini batch.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # Return a tuple ``(nabla_b, nabla_w)`` representing the
        # gradient for the cost function C_x.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # because the last layer is softmax layer, so we need softmax activation function.
        w = self.weights[-1]
        b = self.biases[-1]
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)
        # backward pass
        delta = LogLikeCost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
            # cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # '**' - to the power of.
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def loadParams(self,file):
        with open(file, 'r') as load_file:
            load_dict = json.load(load_file)
        self.weights = [np.array(w) for w in load_dict['weights']]
        self.biases = [np.array(b) for b in load_dict['biases']]

    def predict(self,data):
        data = np.reshape(data, (784, 1))
        return self.feedforward(data)
#### Miscellaneous functions
def vectorized_result(j):
    # Return a 10-dimensional unit vector with a 1.0 in the j'th position
    # and zeroes elsewhere.
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    # The sigmoid function.
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # Derivative of the sigmoid function.
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    # The softmax function.
    return np.exp(z)/np.sum(np.exp(z))

def softmax_prime(z):
    # Derivative of the softmax function.
    return softmax(z)*(1-softmax(z))