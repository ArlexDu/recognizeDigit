from django.apps import AppConfig
from webapp.networks import fnn_network

class WebappConfig(AppConfig):
    name = 'webapp'
    def ready(self):
        fnn_network.setWieghts()
