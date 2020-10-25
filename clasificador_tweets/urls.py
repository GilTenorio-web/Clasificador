from django.urls import path
from clasificador_tweets import views

urlpatterns = [
    path('', views.clasificador, name="clasificador")
]