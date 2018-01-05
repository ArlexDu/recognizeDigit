# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal.pool import pool_2d

w = theano.shared(
    np.asarray(
        np.random.rand(3,1,5,5),
        dtype=theano.config.floatX),
    borrow=True)
b = theano.shared(
    np.asarray(
        np.random.rand(3,),
        dtype=theano.config.floatX),
    borrow=True)
params = []
param = [w, b]
params.append(param)
params.append(param)
print(w)
print(b)
print("w")
print(w.get_value())
print("b")
print(b.get_value())
print("params")
print(params)
print([params[0][0].get_value(),params[0][1].get_value()])