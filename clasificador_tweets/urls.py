from django.urls import path
from clasificador_tweets import views

urlpatterns = [
    path('', views.home, name="home"),
    path('clasificador/', views.clasificador, name="clasificador"),
]