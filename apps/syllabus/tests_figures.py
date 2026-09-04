
import shutil
import tempfile

from django.core.files.base import ContentFile
from django.test import TestCase, override_settings

from apps.syllabus.services.figure_keys import (
    build_figure_key,
    parse_session,
)


def _synthetic_pdf(with_code=True):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    # Thin-line triangle (every path has ~zero area: the old area filter
    # dropped all of these).
    page.draw_line(pymupdf.Point(100, 500), pymupdf.Point(200, 500))
    page.draw_line(pymupdf.Point(200, 500), pymupdf.Point(100, 600))
    page.draw_line(pymupdf.Point(100, 600), pymupdf.Point(100, 500))
    page.insert_text(pymupdf.Point(90, 495), "A")
    page.insert_text(pymupdf.Point(205, 505), "B")
    page.insert_text(pymupdf.Point(90, 615), "C")
    # Examiner box built from 4 separate rules (passes primitive counts,
    # must die on the furniture keyword instead).
    page.draw_line(pymupdf.Point(350, 500), pymupdf.Point(500, 500))
    page.draw_line(pymupdf.Point(500, 500), pymupdf.Point(500, 600))
    page.draw_line(pymupdf.Point(500, 600), pymupdf.Point(350, 600))
    page.draw_line(pymupdf.Point(350, 600), pymupdf.Point(350, 500))
    page.insert_text(pymupdf.Point(360, 540), "For Examiners Use")
    if with_code:
        page.insert_text(pymupdf.Point(72, 72), "MATHEMATICS 6880/02")
    return doc.tobytes()


class FigureKeyTests(TestCase):
    def test_full_key(self):
        self.assertEqual(
            build_figure_key("igcse", "0580", 2024, "MJ", 1, 12, 2),
            "IGCSE-0580-2024-MJ-P1-p12-f2")

    def test_missing_segments_omitted(self):
        self.assertEqual(
            build_figure_key("egcse", "6880", 2022, "", None, 3, 1),
            "EGCSE-6880-2022-p3-f1")

    def test_parse_session_cambridge(self):
        self.assertEqual(parse_session("0580_m24_qp_12"), "MJ")
        self.assertEqual(parse_session("0620_s23_ms_21"), "ON")
        self.assertEqual(parse_session("0620_w17_qp_33"), "FM")

    def test_parse_session_unknown(self):
        self.assertEqual(
            parse_session("EGCSE Mathematics 2022 Question Paper 1"), "")
        self.assertEqual(parse_session("M/J 2024 Paper 1"), "MJ")


class FigureExtractionTests(TestCase):
    def test_thin_line_diagram_found_examiner_box_dropped(self):
        import os

        from apps.syllabus.services.ingestion import extract_figures

        with tempfile.NamedTemporaryFile(suffix=".pdf",
                                         delete=False) as fh:
            fh.write(_synthetic_pdf())
            path = fh.name
        try:
            figs = extract_figures(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(figs), 1)
        self.assertEqual(figs[0]["page"], 1)
        self.assertEqual(figs[0]["kind"], "vector")

    def test_cover_verification(self):
        import os

        import pymupdf

        from apps.syllabus.services.ingestion import verify_cover_subject

        def pdf_with(text):
            doc = pymupdf.open()
            page = doc.new_page(width=595, height=842)
            page.insert_text(pymupdf.Point(72, 72), text)
            with tempfile.NamedTemporaryFile(suffix=".pdf",
                                             delete=False) as fh:
                fh.write(doc.tobytes())
                return fh.name

        good = pdf_with("MATHEMATICS 6880/02 Structured Questions")
        bad = pdf_with("AGRICULTURE 6882/01 Multiple Choice")
        empty = pdf_with(" ")
        try:
            self.assertTrue(verify_cover_subject(good, {"6880"}))
            self.assertFalse(verify_cover_subject(bad, {"6880"}))
            self.assertTrue(verify_cover_subject(empty, {"6880"}))
        finally:
            for p in (good, bad, empty):
                os.unlink(p)


class FigureStoreTests(TestCase):
    def test_keys_stable_across_reingest(self):
        from apps.rag.models import DocumentFigure
        from apps.syllabus.models import Subject, Syllabus, SyllabusDocument
        from apps.syllabus.services.ingestion import _store_document_figures

        tmp = tempfile.mkdtemp()
        try:
            with override_settings(MEDIA_ROOT=tmp):
                syl = Syllabus.objects.create(level="EGCSE", name="E",
                                              version="1")
                subj = Subject.objects.create(syllabus=syl, code="6880",
                                              name="Mathematics")
                doc = SyllabusDocument(
                    syllabus=syl, subject=subj, title="Maths P2 2024",
                    source="egcse", year=2024, paper_number=2)
                doc.file.save("t.pdf", ContentFile(_synthetic_pdf()),
                              save=True)
                n1 = _store_document_figures(doc)
                keys1 = sorted(DocumentFigure.objects.filter(
                    document=doc).values_list("stable_key", flat=True))
                n2 = _store_document_figures(doc)
                keys2 = sorted(DocumentFigure.objects.filter(
                    document=doc).values_list("stable_key", flat=True))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertGreater(n1, 0)
        self.assertEqual(n1, n2)
        self.assertEqual(keys1, keys2)
        self.assertTrue(
            all(k.startswith("EGCSE-6880-2024-P2-p1-f") for k in keys1))
