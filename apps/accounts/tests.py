from django.test import TestCase
from rest_framework.test import APIClient

from apps.accounts.models import User
from apps.syllabus.models import Enrollment, Subject, Syllabus


class AuthAndWorkspaceTests(TestCase):
    def setUp(self):
        syllabus = Syllabus.objects.create(level="EGCSE", name="EGCSE T", version="1.0")
        self.subject = Subject.objects.create(syllabus=syllabus, code="6880", name="Mathematics")
        self.client = APIClient()

    def test_register_returns_tokens_and_creates_profile(self):
        response = self.client.post(
            "/api/auth/register/",
            {"username": "newkid", "password": "StrongPass99", "level": "EGCSE"},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("access", body)
        self.assertTrue(User.objects.filter(username="newkid").exists())
        self.assertIsNotNone(User.objects.get(username="newkid").profile)

    def test_enroll_and_workspace_roundtrip(self):
        self.client.post(
            "/api/auth/register/",
            {"username": "w", "password": "StrongPass99"},
            format="json",
        )
        tokens = self.client.post(
            "/api/auth/token/", {"username": "w", "password": "StrongPass99"}, format="json"
        ).json()
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {tokens['access']}")

        enrolled = self.client.post("/api/my-subjects/", {"subject_id": self.subject.id}, format="json")
        self.assertEqual(enrolled.status_code, 201)

        listing = self.client.get("/api/my-subjects/").json()
        self.assertEqual(len(listing["results"]), 1)
        self.assertEqual(listing["results"][0]["subject"]["code"], "6880")

        workspace = self.client.get(f"/api/workspace/{self.subject.id}/").json()
        self.assertEqual(workspace["subject"]["name"], "Mathematics")
        self.assertIsNone(workspace["latest_session_id"])

    def test_workspace_unknown_subject(self):
        self.client.post(
            "/api/auth/register/", {"username": "x", "password": "StrongPass99"}, format="json"
        )
        tokens = self.client.post(
            "/api/auth/token/", {"username": "x", "password": "StrongPass99"}, format="json"
        ).json()
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {tokens['access']}")
        response = self.client.get("/api/workspace/999999/")
        self.assertEqual(response.status_code, 400)

    def test_tiered_subject_requires_tier(self):
        # Make the subject tiered, then enrolling without a tier must be rejected.
        self.subject.tiers_available = ["core", "extended"]
        self.subject.save()
        self.client.post(
            "/api/auth/register/", {"username": "t", "password": "StrongPass99"}, format="json"
        )
        tokens = self.client.post(
            "/api/auth/token/", {"username": "t", "password": "StrongPass99"}, format="json"
        ).json()
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {tokens['access']}")

        without = self.client.post("/api/my-subjects/", {"subject_id": self.subject.id}, format="json")
        self.assertEqual(without.status_code, 400)

        good = self.client.post(
            "/api/my-subjects/",
            {"subject_id": self.subject.id, "tier": "core"},
            format="json",
        )
        self.assertEqual(good.status_code, 201)
        self.assertEqual(good.json()["tier"], "core")

        bad = self.client.post(
            "/api/my-subjects/",
            {"subject_id": self.subject.id, "tier": "ancient"},
            format="json",
        )
        self.assertEqual(bad.status_code, 400)

