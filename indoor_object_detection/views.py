from django.http import HttpResponse
from django.shortcuts import render
from django.shortcuts import redirect
# FILE UPLOAD AND VIEW
from  django.core.files.storage import FileSystemStorage
# SESSION
from django.conf import settings
from .models import *
import os
import subprocess
import sys
from ML import speech_to_text

active_processes = {}




def first(request):
    return render(request, 'index.html')

def index(request):
    return render(request, 'index.html')


def reg(request):
    return render(request, 'registration.html')



def addreg(request):
    if request.method=="POST":
        name=request.POST.get('name')
        phone=request.POST.get('phone_number')
        email=request.POST.get('email')
        password=request.POST.get('password')
         
        sa=user(name=name,phone_number=phone,email=email,password=password)
        sa.save()

    return render(request,'index.html',{'message':"Successfully Registered"})


def login(request):
    return render(request,'login.html')

def addlogin(request):
    email=request.POST.get('email')
    password=request.POST.get('password')
    if email=='admin@gmail.com' and password=='admin':
        request.session['details']='admin'
        return render(request,'index.html')
    elif user.objects.filter(email=email,password=password).exists():
        userdetails=user.objects.get(email=email,password=password)
        request.session['uid']=userdetails.id
        request.session['uname']=userdetails.name
        return render(request,'index.html')
    else:
        return render(request,'login.html')

def logout(request):
    session_keys=list(request.session.keys())
    for key in session_keys:
        del request.session[key]
    return redirect(index)

# Admin Views
def view_users(request):
    if request.session.get('details') != 'admin':
        return redirect('login')
    users = user.objects.all()
    return render(request, 'view_users.html', {'users': users})

def view_results(request):
    if request.session.get('details') != 'admin':
        return redirect('login')
    results = fileupload.objects.all()
    return render(request, 'view_results.html', {'results': results})

# User Views
def upload_file(request):
    if not request.session.get('uid'):
        return redirect('login')

    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        user_id = request.session.get('uid')
        username = user.objects.get(id=user_id).name

        # Save file to fileupload model with blank result
        file_upload = fileupload(
            username=username,
            file=uploaded_file,
            result=''  # Keep result blank as requested
        )
        file_upload.save()

        # Redirect to user results page with success message
        return render(request, 'upload.html', {
            'message': 'File uploaded successfully! Object detection will be processed.'
        })

    return render(request, 'upload.html')

def speech_recognition(request):
    object_search = speech_to_text.predict()
    request.session["object_search"] = object_search
    return render(request,'navigation.html')

def test_navigation(request):
    return render(request,'navigation.html')

def navigate(request):
    if not request.session.get('uid'):
        return redirect('login')
    uid = request.session['uid']
    if uid in active_processes and active_processes[uid].poll() is None:
        # Process already running, perhaps redirect to detecting
        return render(request, 'detecting.html')
    else:

        object_search = request.session.get("object_search", "")
        # Save user activity
        username = user.objects.get(id=uid).name
        UserActivity.objects.create(user_id=uid, username=username, object_search=object_search)
        object_det_cmd = [
            sys.executable, 'object_detect_v7/detect.py',
            '--weights', 'object_detect_v7/yolov7x.pt',
            '--source', '0',
            '--view-img',
            '--object-search', object_search
        ]
        process = subprocess.Popen(object_det_cmd)
        active_processes[uid] = process
        return render(request, 'detecting.html')

def stop_detection(request):
    if request.session.get('uid'):
        uid = request.session['uid']
        if uid in active_processes:
            process = active_processes[uid]
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            del active_processes[uid]
    return redirect('navigation')

def view_user_activities(request):
    if request.session.get('details') != 'admin':
        return redirect('login')
    activities = UserActivity.objects.select_related('user').all().order_by('-time')
    return render(request, 'view_user_activities.html', {'activities': activities})

def view_user_results(request):
    if not request.session.get('uid'):
        return redirect('login')
    user_id = request.session.get('uid')
    results = fileupload.objects.filter(username=user.objects.get(id=user_id).name)
    return render(request, 'user_results.html', {'results': results})
