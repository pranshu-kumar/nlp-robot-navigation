from django.urls import path
from . import views

urlpatterns = [
    path('', views.translate_instruction, name='translator'),
    path('robottranslation/', views.show_translation, name='robottranslation')
]