
from django.urls import path 
from . import views
urlpatterns = [
    path('',views.home,name="home"),
    path('self/page',views.arrgen,name="arrgen"),
    path('self/',views.self,name="self"),
    path('covid/',views.covid,name="covid"),
    path('video/',views.video,name="video"),
    path('model/',views.model,name="model"),
    # path('model1/',views.model1,name="model1"),
   
]
