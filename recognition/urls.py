from __future__ import annotations

from django.urls import path
from .views import index, api_recognize, reload_model, manage, manage_encode, manage_train

urlpatterns = [
    path('', index, name='index'),
    path('api/recognize', api_recognize, name='api_recognize'),
    path('reload', reload_model, name='reload_model'),
    path('manage', manage, name='manage'),
    path('manage/encode', manage_encode, name='manage_encode'),
    path('manage/train', manage_train, name='manage_train'),
]
