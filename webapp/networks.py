class fnn_network(object):

    weights=[]
    biais=[]

    @staticmethod
    def setWieghts():
        fnn_network.weights = [1,2,3]

    @staticmethod
    def getWieghts():
        return fnn_network.weights