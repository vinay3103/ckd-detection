from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from .forms import UserSignupForm, UserLoginForm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from .model import train_model, predict  # Import your functions
from django.http import JsonResponse
import os
from django.conf import settings


# Define paths (should match what you used in training)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'voting_classifier.pkl')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'models', 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(settings.BASE_DIR, 'models', 'label_encoder.pkl')
DATA_PATH = os.path.join(settings.BASE_DIR, 'data', 'ckd_data.csv')



def signup_view(request):
    if request.method == 'POST':
        form = UserSignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            phone_number = form.cleaned_data.get('phone_number')
            messages.success(request, "Signup successful. Please log in.")
            return redirect('login')
    else:
        form = UserSignupForm()
    return render(request, 'signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = UserLoginForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, "Invalid credentials")
    else:
        form = UserLoginForm()
    return render(request, 'login.html', {'form': form})

def home_view(request):
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'home.html')

def about_view(request):
    return render(request, 'about.html')

def train_model_view(request):
    # This view would trigger model training when accessed (e.g., via a specific URL)
    voting_clf, scaler, label_encoder = train_model()
    return JsonResponse({"message": "Model trained successfully."})

def predict_view(request):
    if request.method == 'POST':
        data = {
            'age': float(request.POST['age']),
            'bp': float(request.POST['bp']),  
            'sg': float(request.POST['sg']),  
            'al': float(request.POST['al']),
            'su': float(request.POST['su']),
            'rbc': float(request.POST['rbc']),
            'pc': float(request.POST['pc']),
            'pcc': float(request.POST['pcc']),
            'ba': float(request.POST['ba']),
            'bgr': float(request.POST['bgr']),
            'bu': float(request.POST['bu']),
            'sc': float(request.POST['sc']),
            'sod': float(request.POST['sod']),
            'pot': float(request.POST['pot']),
            'hemo': float(request.POST['hemo']),
            'pcv': float(request.POST['pcv']),
            'wc': float(request.POST['wc']),
            'rc': float(request.POST['rc']),
            'htn': int(request.POST['htn']),
            'dm': int(request.POST['dm']),
            'cad': int(request.POST['cad']),
            'appet': int(request.POST['appet']),
            'pe': int(request.POST['pe']),
            'ane': int(request.POST['ane'])
        }
        if data['age'] < 0:
            return JsonResponse({'error': "Age cannot be negative"}, status=400)
        prediction = predict(data)
        return JsonResponse({'prediction': int(prediction[0])})
    else:
        return render(request, 'predict.html')  # Load the input form template

def load_model():
    return joblib.load('ckd_model.pkl')  # Load trained model
def predict_disease(request):
    if request.method == 'POST':
        try:
            data = request.POST.dict()
            input_data = pd.DataFrame([data])

            voting_clf = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            label_encoder = joblib.load(LABEL_ENCODER_PATH)

            input_data_scaled = scaler.transform(input_data)
            prediction = voting_clf.predict(input_data_scaled)
            predicted_class = label_encoder.inverse_transform(prediction)

            return JsonResponse({
                'status': 'success',
                'prediction': predicted_class[0]
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    else:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid request method. Please use POST.'
        })
