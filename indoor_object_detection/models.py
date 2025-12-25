
from django.db import models
from django.utils import timezone



class user(models.Model):
    name=models.CharField(max_length=150)
    phone_number=models.CharField(max_length=120)
    email=models.CharField(max_length=120)
    password=models.CharField(max_length=120)

class feedback(models.Model):
    username=models.CharField(max_length=150)
    feedbacks=models.CharField(max_length=150)

class fileupload(models.Model):
    username=models.CharField(max_length=150)
    file=models.FileField(max_length=150)
    result=models.CharField(max_length=150)
    created_at = models.DateTimeField(auto_now_add=True)

class UserActivity(models.Model):
    user = models.ForeignKey(user, on_delete=models.CASCADE)
    username = models.CharField(max_length=150)
    object_search = models.CharField(max_length=100)
    time = models.DateTimeField(auto_now_add=True)
    created_at = models.DateTimeField(auto_now_add=True)
