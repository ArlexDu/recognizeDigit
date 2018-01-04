import json
import os
from django.conf import settings
import numpy as np
class fnn_network(object):

    weights = []
    biases = []

    @staticmethod
    def loadParams():
        params = os.path.join(settings.TRAIN_ROOT,'fnn_network.json')
        with open(params,'r') as load_file:
            load_dict = json.load(load_file)
        fnn_network.weights = [np.array(w) for w in load_dict['weights']]
        fnn_network.biases = [np.array(b) for b in load_dict['biases']]

    @staticmethod
    def getParams():
        return fnn_network.weights,fnn_network.biases

    @staticmethod
    def feedforward(data):
        """Return the output of the network if ``a`` is input."""
        # print('data shape is %s' % data.shape)
        data = np.reshape(data,(784,1))
        for b, w in zip(fnn_network.biases, fnn_network.weights):
            # print('biases shape is %s,%s' % b.shape)
            # print('weight shape is %s,%s' % w.shape)
            # print('data shape is %s,%s' % data.shape)
            data = fnn_network.sigmoid(np.dot(w, data) + b)
        return data

    @staticmethod
    def sigmoid(z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))