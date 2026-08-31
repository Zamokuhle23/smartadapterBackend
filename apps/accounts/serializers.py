from rest_framework import serializers

from .models import LearnerProfile, User


class LearnerProfileSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    level = serializers.CharField(source="user.level", read_only=True)
    form_level = serializers.IntegerField(source="user.form_level", read_only=True)

    class Meta:
        model = LearnerProfile
        fields = (
            "username",
            "level",
            "form_level",
            "preferred_language",
            "learning_style",
            "pace",
            "diagnostic_complete",
            "daily_goal_minutes",
        )
        read_only_fields = ("username", "level", "form_level")


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        model = User
        fields = ("username", "email", "password", "level", "form_level", "school_name")

    def create(self, validated_data):
        password = validated_data.pop("password")
        user = User(**validated_data)
        user.set_password(password)
        user.save()
        LearnerProfile.objects.create(user=user)
        return user
