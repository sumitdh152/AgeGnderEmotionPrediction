
from django.contrib import admin
from django.urls import path
from face_detect.views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home,name='about'),
    path('login', Login,name='login'),
    path('logout', Logout,name='logout'),
    path("face_detect",face_detect),
    path('signup', signup,name='signup'),
    path('about', about,name='home'),
    path('contact', contact,name='contact'),
]+static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
