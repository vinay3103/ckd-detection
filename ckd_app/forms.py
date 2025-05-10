from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

class UserSignupForm(UserCreationForm):
    phone_number = forms.CharField(max_length=15, required=True, help_text='Required')

    class Meta:
        model = User
        fields = ('username', 'phone_number', 'password1', 'password2')

class UserLoginForm(AuthenticationForm):
    class Meta:
        model = User
        fields = ('username', 'password')

class PredictionForm(forms.Form):
    input_data = forms.CharField(max_length=100, help_text="Provide input data for prediction.")

