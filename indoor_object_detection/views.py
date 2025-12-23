from django.http import HttpResponse
from django.shortcuts import render
from django.shortcuts import redirect
# FILE UPLOAD AND VIEW
from  django.core.files.storage import FileSystemStorage
# SESSION
from django.conf import settings
from .models import *




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

def view_user_results(request):
    if not request.session.get('uid'):
        return redirect('login')
    user_id = request.session.get('uid')
    results = fileupload.objects.filter(username=user.objects.get(id=user_id).name)
    return render(request, 'user_results.html', {'results': results})
