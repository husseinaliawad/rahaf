# faq_app/urls.py  
from django.urls import path  
from . import views  

urlpatterns = [  
    path('', views.home_view, name='home'),      # Home view at root URL (optional but recommended)  
    path('add/', views.add_view, name='add'),    # URL for adding (e.g., add a question)  
    path('search/', views.search_view, name='search'),  # URL for searching  
]