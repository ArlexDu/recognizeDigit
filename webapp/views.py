from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from webapp.networks import fnn_network
from webapp.process import *
import os
import time
import base64

# Create your views here.

def index(request):
    weights,biases  =fnn_network.getParams()
    print(biases)
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
    return HttpResponse(data)