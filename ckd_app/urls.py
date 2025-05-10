from django.urls import path
from . import views

urlpatterns = [
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('home/', views.home_view, name='home'),
    path('about/', views.about_view, name='about'),
    path('predict/', views.predict_view, name='predict'),
    path('train-model/', views.train_model_view, name='train_model'),
    path('predict/', views.predict_view, name='predict'),
]
