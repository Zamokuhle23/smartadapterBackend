"""
JWT authentication for WebSocket connections.

The Android client cannot use Django session cookies, so it connects as
ws://host/ws/chat/<id>/?token=<access>. This middleware validates the token
with SimpleJWT and attaches the resolved user to the scope.
"""

from urllib.parse import parse_qs

from channels.db import database_sync_to_async
from channels.middleware import BaseMiddleware
from django.contrib.auth.models import AnonymousUser
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import AccessToken


@database_sync_to_async
def _get_user(user_id):
    from apps.accounts.models import User

    try:
        return User.objects.get(pk=user_id)
    except User.DoesNotExist:
        return AnonymousUser()


class JwtAuthMiddleware(BaseMiddleware):
    async def __call__(self, scope, receive, send):
        query = parse_qs(scope.get("query_string", b"").decode())
        token = (query.get("token") or [None])[0]
        scope["user"] = AnonymousUser()
        if token:
            try:
                validated = AccessToken(token)
                scope["user"] = await _get_user(validated["user_id"])
            except TokenError:
                pass  # stays anonymous; consumer closes with 4401
        return await super().__call__(scope, receive, send)
