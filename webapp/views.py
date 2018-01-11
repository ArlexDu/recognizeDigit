from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from webapp.process import *
import os
import json
import time
import base64
import numpy as np
import train.FNN
import train.CNN
from train.CNN import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer,ReLU

mini_batch_size = 1
load_cnn = os.path.join(settings.TRAIN_ROOT,'cnn_network.json')
cnn = train.CNN.Network([
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
fnn = train.FNN.Network([784, 40, 10], load_fnn)

def index(request):
    return render(request,'index.html',{})

# predict function
def upload(request):
    data = request.POST.get('file')
    # save data as a picture
    path = settings.MEDIA_ROOT
    type = data.split(';')[0].split('/')[1]
    fileName = os.path.join(path,str(int(time.time()*1000))+'.'+type)
    fileData = base64.b64decode(data.split(',')[1])
    file = open(fileName,'wb')
    file.write(fileData)
    file.close()
    # split picture into single digit picture
    array_pictures = process(fileName)
    result_fnn = []
    result_cnn = []
    # predict digit from pictures
    for pic in array_pictures:
        rfnn = np.argmax(fnn.predict(pic))
        rcnn = cnn.predict(pic)[0]
        # print(result_array)
        #result = np.argmax(result_array)
        result_fnn.append(rfnn)
        result_cnn.append(rcnn)
    rfnn = ''
    rcnn = ''
    for f,c in zip(result_fnn,result_cnn):
        rfnn = rfnn + str(f)
        rcnn = rcnn + str(c)
    result = {'fnn':rfnn,"cnn":rcnn}
    #print(result)
    return HttpResponse(json.dumps(result))