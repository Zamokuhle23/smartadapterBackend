"""
ASGI config for config project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

application = get_asgi_application()

# WebSocket support via Django Channels.
# Custom middleware: JWT passed as ?token= query param (mobile clients).
from apps.tutoring.auth import JwtAuthMiddleware  # noqa: E402
from channels.routing import ProtocolTypeRouter, URLRouter  # noqa: E402
from channels.security.websocket import AllowedHostsOriginValidator  # noqa: E402

import apps.tutoring.routing  # noqa: E402

application = ProtocolTypeRouter(
    {
        "http": application,
        "websocket": AllowedHostsOriginValidator(
            JwtAuthMiddleware(URLRouter(apps.tutoring.routing.websocket_urlpatterns))
        ),
    }
)

