from django.contrib.auth import get_user_model
from rest_framework import status
from rest_framework.test import APIClient, APITestCase

from .models import Subject, Syllabus


class SyllabusFilterTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        User = get_user_model()
        cls.user = User.objects.create_user(username="tester", password="pass12345")
        cls.jc = Syllabus.objects.create(
            level="JC",
            name="Junior Certificate",
            status=Syllabus.Status.PUBLISHED,
        )
        cls.egcse = Syllabus.objects.create(
            level="EGCSE",
            name="Eswatini General Certificate",
            status=Syllabus.Status.PUBLISHED,
        )
        Subject.objects.create(syllabus=cls.jc, code="309", name="Mathematics (JC)")
        Subject.objects.create(syllabus=cls.egcse, code="6880", name="Mathematics (EGCSE)")
        Subject.objects.create(syllabus=cls.egcse, code="6884", name="Biology")

    def setUp(self):
        self.client.force_authenticate(self.user)

    def test_syllabi_level_filter(self):
        r = self.client.get("/api/syllabi/", {"level": "EGCSE"})
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        results = r.json()["results"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["level"], "EGCSE")
        self.assertEqual(results[0]["id"], self.egcse.id)

    def test_syllabi_level_filter_jc(self):
        r = self.client.get("/api/syllabi/", {"level": "JC"})
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        results = r.json()["results"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], self.jc.id)

    def test_subjects_syllabus_filter(self):
        r = self.client.get("/api/subjects/", {"syllabus": self.egcse.id})
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        codes = {s["code"] for s in r.json()["results"]}
        self.assertEqual(codes, {"6880", "6884"})