from django.urls import path
from analyzer.views import analyze

urlpatterns = [
    path('analyze/', analyze, name='analyze'),
]
