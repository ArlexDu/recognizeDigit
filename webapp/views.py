from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from webapp.networks import fnn_network,cnn_network,ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer,ReLU
from webapp.process import *
import os
import time
import base64
import numpy as np

# Create your views here.
mini_batch_size = 1

cnn = cnn_network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

def index(request):
    return render(request,'index.html',{})

def upload(request):
    data = request.POST.get('file')
    path = settings.MEDIA_ROOT
    type = data.split(';')[0].split('/')[1]
    fileName = os.path.join(path,str(int(time.time()*1000))+'.'+type)
    fileData = base64.b64decode(data.split(',')[1])
    file = open(fileName,'wb')
    file.write(fileData)
    file.close()
    array_pictures = process(fileName)
    results = []
    for pic in array_pictures:
        # result_array = fnn_network.feedforward(pic)
        result_array = cnn.predict(pic)
        # print(result_array)
        result = np.argmax(result_array)
        results.append(result)
    return HttpResponse(results)