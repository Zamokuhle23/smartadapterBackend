from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

from .models import LearnerProfile
from .serializers import LearnerProfileSerializer, RegisterSerializer


class RegisterView(APIView):
    """POST {username, email?, password, level?, form_level?} -> JWT pair."""

    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        refresh = RefreshToken.for_user(user)
        return Response(
            {
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "username": user.username,
            }
        )



class LearnerProfileView(APIView):
    """GET/PATCH the current student's personalization profile."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        profile = LearnerProfile.for_user(request.user)
        return Response(LearnerProfileSerializer(profile).data)

    def patch(self, request):
        profile = LearnerProfile.for_user(request.user)
        serializer = LearnerProfileSerializer(profile, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

