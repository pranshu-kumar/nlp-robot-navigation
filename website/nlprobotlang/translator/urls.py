from django.urls import path
from .views import translate_instruction

urlpatterns = [
    path('', translate_instruction, name='translator')
]