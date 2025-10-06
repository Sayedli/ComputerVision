from __future__ import annotations

from django.urls import path
from .views import index, api_recognize, reload_model

urlpatterns = [
    path('', index, name='index'),
    path('api/recognize', api_recognize, name='api_recognize'),
    path('reload', reload_model, name='reload_model'),
]

