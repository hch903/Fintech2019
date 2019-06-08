from django.urls import path,include
from . import views

urlpatterns = [
    path('',views.score.as_view(),name = "score"),
]
