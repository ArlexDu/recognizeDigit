import json
import os
from django.conf import settings
import numpy as np
class fnn_network(object):

    weights=[]
    biases=[]

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