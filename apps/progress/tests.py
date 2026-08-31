from django.test import TestCase

from apps.progress.services.bkt import P_INIT, update_mastery


class BKTTests(TestCase):
    def test_first_correct_increases_mastery(self):
        self.assertGreater(update_mastery(None, True), P_INIT)

    def test_first_wrong_decreases_mastery(self):
        self.assertLess(update_mastery(None, False), P_INIT)

    def test_monotonic_path(self):
        m0 = update_mastery(None, True)
        m1 = update_mastery(m0, True)
        m2 = update_mastery(m1, False)
        self.assertGreater(m1, m0)
        self.assertLess(m2, m1)

    def test_stays_bounded(self):
        m = P_INIT
        for _ in range(50):
            m = update_mastery(m, True)
        self.assertLessEqual(m, 1.0)
        for _ in range(50):
            m = update_mastery(m, False)
        self.assertGreaterEqual(m, 0.0)
