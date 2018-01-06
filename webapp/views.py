from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from webapp.process import *
import os
import json
import time
import base64
import numpy as np
import train.network2
import train.network3
from train.network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer,ReLU

mini_batch_size = 1
load_cnn = os.path.join(settings.TRAIN_ROOT,'cnn_network.json')
cnn = train.network3.Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size,load_cnn)
load_fnn = os.path.join(settings.TRAIN_ROOT,'fnn_network.json')
fnn = train.network2.Network([784,40,10],load_fnn)

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
    result_fnn = 0
    result_cnn = 0
    for pic in array_pictures:
        rfnn = np.argmax(fnn.predict(pic))
        # rcnn = cnn.predict(pic)[0]
        # print(result_array)
        #result = np.argmax(result_array)
        result_fnn = result_fnn*10+rfnn
        result_cnn = result_cnn*10+rcnn
    result = {'fnn':str(result_fnn),"cnn":str(result_cnn)}
    #print(result)
    return HttpResponse(json.dumps(result))