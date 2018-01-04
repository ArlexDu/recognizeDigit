from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from webapp.networks import fnn_network
from webapp.process import *
import os
import time
import base64
import numpy as np

# Create your views here.

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
    array_picture = process(fileName)
    result = np.argmax(fnn_network.feedforward(array_picture))
    print('result is ')
    print(result)
    return HttpResponse(result)