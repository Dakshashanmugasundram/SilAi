from django.shortcuts import render, redirect
from django.http import JsonResponse

from allauth.socialaccount.providers.google.views import GoogleOAuth2Adapter
from allauth.socialaccount.providers.oauth2.client import OAuth2Client
from dj_rest_auth.registration.views import SocialLoginView

from django.http import JsonResponse
from django.views.decorators.http import require_GET
from .models import CustomUserModel

class GoogleLogin(SocialLoginView): # if you want to use Authorization Code Grant, use this
    adapter_class = GoogleOAuth2Adapter
    callback_url = "http://localhost:3000/"
    client_class = OAuth2Client

def email_confirmation(request, key):
    return redirect(f"http://localhost:3000/dj-rest-auth/registration/account-confirm-email/{key}")

def reset_password_confirm(request, uid, token):
    return redirect(f"http://localhost:3000/reset/password/confirm/{uid}/{token}")

@require_GET
def check_email(request):
    email = request.GET.get("email", "").strip()
    exists = CustomUserModel.objects.filter(email=email).exists()
    return JsonResponse({"exists": exists})