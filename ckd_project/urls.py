from django.contrib import admin
from django.urls import path, include
from ckd_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.signup_view, name='root'),  # Redirect root URL to the signup page
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('home/', views.home_view, name='home'),
    path('about/', views.about_view, name='about'),
    path('predict/', views.predict_view, name='predict'),
    path('train-model/', views.train_model_view, name='train_model'),
]
