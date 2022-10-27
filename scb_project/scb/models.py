from django.db import models

# Create your models here.
class allmodel(models.Model):
    Filename = models.CharField(max_length=50)
    Modelname = models.CharField(max_length=50)
    Input = models.CharField(max_length=50)
    Output = models.CharField(max_length=50)