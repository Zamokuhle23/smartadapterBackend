"""
Document ingestion pipeline: extract text -> chunk -> embed -> index.

Runs synchronously when Celery is not configured (CELERY_TASK_ALWAYS_EAGER or
no Redis), and as a background task otherwise.
"""

import os

from django.conf import settings

from apps.rag.models import DocumentChunk
from apps.rag.services.embeddings import get_embedder


def extract_text(path: str) -> str:
    """Full text of a document (any supported format)."""
    return "\n".join(extract_pages(path))


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Backwards-compatible plain-text chunker (no page info)."""
    return [t for _, t in chunk_pages([text], size=size, overlap=overlap)]


def extract_pages(path: str) -> list[str]:
    """
    Extract text per page. PDF pages come back in order as separate strings so the
    chunker can record which PDF page each chunk came from (for figure mapping).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt" or ext == ".md":
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return [fh.read()]
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install 'pypdf' to ingest PDF files") from exc
        reader = PdfReader(path)
        return [page.extract_text() or "" for page in reader.pages]
    if ext == ".docx":
        try:
            import docx  # python-docx
        except ImportError as exc:
            raise RuntimeError("Install 'python-docx' to ingest DOCX files") from exc
        document = docx.Document(path)
        return ["\n".join(p.text for p in document.paragraphs)]
    raise ValueError(f"Unsupported document type: {ext}")


def extract_figures(path: str):
    """
    Extract figures from a PDF, per page.

    Two strategies combined:
      1. Embedded raster images (photos, scanned images).
      2. Vector drawing regions (diagrams drawn with lines/shapes in the PDF -
         common for Biology/Physics figures). Drawing bounding boxes are clustered
         into figure regions and each region is rendered to PNG.

    Returns list of dicts: {"page": int(1-based), "image": bytes(PNG), "caption": str}.
    Uses PyMuPDF if available; returns [] gracefully otherwise or for non-PDFs.
    """
    if os.path.splitext(path)[1].lower() != ".pdf":
        return []
    try:
        import pymupdf  # PyMuPDF (pip install pymupdf)
    except ImportError:
        return []

    figures = []
    try:
        doc = pymupdf.open(path)
    except Exception:  # noqa: BLE001 - bad PDF shouldn't block ingestion
        return []

    with doc:
        for page in doc:
            page_no = page.number + 1
            # --- 1. Embedded raster images ---
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    pix = pymupdf.Pixmap(doc, xref)
                except Exception:  # noqa: BLE001
                    continue
                try:
                    if pix.n - pix.alpha < 4:
                        data = pix.tobytes("png")
                    else:
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                        data = pix.tobytes("png")
                finally:
                    pix = None
                if len(data) > 2000:
                    figures.append(_figure_dict(page_no, data))

            # --- 2. Vector drawing regions ---
            try:
                regions = _vector_region_rects(page)
            except Exception:  # noqa: BLE001
                regions = []
            for rect in regions:
                try:
                    mat = pymupdf.Matrix(1.5, 1.5)  # 1.5x zoom for clarity
                    pix = page.get_pixmap(matrix=mat, clip=rect)
                    data = pix.tobytes("png")
                except Exception:  # noqa: BLE001
                    continue
                if data and len(data) > 2000:
                    figures.append(_figure_dict(page_no, data))
    return figures


def _figure_dict(page_no: int, data: bytes) -> dict:
    return {"page": page_no, "image": data, "caption": ""}


def _vector_region_rects(page, min_area: float = 900.0):
    """
    Cluster a page's vector drawings into non-overlapping figure regions.

    Drops tiny decorative paths (underlines, tick marks) then merges remaining
    drawing bounding boxes into regions by proximity. Returns list[pymupdf.Rect].
    """
    import pymupdf

    page_rect = page.rect
    rects = []
    for d in page.get_drawings():
        r = d.get("rect")
        if r is None:
            continue
        r = pymupdf.Rect(r)
        area = r.width * r.height
        # Drop tiny/large noise: decorative rules & full-page backgrounds.
        if area < min_area:
            continue
        if area > page_rect.width * page_rect.height * 0.8:
            continue
        rects.append(r)
    if not rects:
        return []

    # Union overlapping rectangles (with a small grow margin so parts of a
    # diagram that touch/overlap are grouped together).
    margin = 6
    union = list(rects)
    changed = True
    while changed:
        changed = False
        merged = []
        for r in union:
            grown = pymupdf.Rect(r.x0 - margin, r.y0 - margin,
                                  r.x1 + margin, r.y1 + margin)
            placed = False
            for m in merged:
                if not (grown.x0 > m.x1 or m.x0 > grown.x1 or
                        grown.y0 > m.y1 or m.y0 > grown.y1):
                    # overlap -> merge
                    m.include_rect(grown)
                    placed = True
                    changed = True
                    break
            if not placed:
                merged.append(pymupdf.Rect(grown))
        union = merged

    # Final filter: drop regions that are tiny strips after merging (e.g. nothing).
    result = []
    for r in union:
        if r.width < 40 or r.height < 40:  # at ~72pt these are unlikely figure-sized
            continue
        # clamp to page
        result.append(pymupdf.Rect(
            max(0, r.x0), max(0, r.y0),
            min(page_rect.x1, r.x1), min(page_rect.y1, r.y1),
        ))
    return result


def chunk_pages(pages: list[str], size: int | None = None,
                overlap: int | None = None) -> list[tuple[int, str]]:
    """
    Page-aware chunking. Returns list of (page_number, text) where page_number is the
    1-based PDF page the chunk's content starts on (0 for non-PDFs).
    """
    size = size or settings.RAG_CHUNK_SIZE
    overlap = overlap if overlap is not None else settings.RAG_CHUNK_OVERLAP

    def chunk_one(text: str) -> list[str]:  # reuse the existing boundary logic
        text = text.strip()
        if not text:
            return []
        out = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                window = text[start:end]
                for sep in ("\n\n", ". ", "\n", " "):
                    cut = window.rfind(sep)
                    if cut > size // 2:
                        end = start + cut + len(sep)
                        break
            out.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return [c for c in out if c]

    pieces: list[tuple[int, str]] = []
    for idx, page_text in enumerate(pages):
        page_no = idx + 1 if len(pages) > 1 else 0
        for piece in chunk_one(page_text):
            pieces.append((page_no, piece))
    return pieces


def process_document(document) -> int:
    """Ingest one SyllabusDocument; returns number of chunks indexed."""
    from apps.syllabus.models import SyllabusDocument

    document.status = SyllabusDocument.Status.PROCESSING
    document.save(update_fields=["status"])
    try:
        pages = extract_pages(document.file.path)
        pieces = chunk_pages(pages)  # list of (page_number, text)
        texts = [p[1] for p in pieces]
        embedder = get_embedder()
        vectors = embedder.embed_texts(texts) if texts else []
        DocumentChunk.objects.filter(document=document).delete()
        for i, (page_no, piece) in enumerate(pieces):
            DocumentChunk.objects.create(
                syllabus=document.syllabus,
                document=document,
                subject=document.subject,
                ordinal=i,
                page_number=page_no or None,
                text=piece,
                embedding=vectors[i] if i < len(vectors) else None,
            )
        _store_document_figures(document)
        document.chunk_count = len(texts)
        document.status = SyllabusDocument.Status.READY
        document.error = ""
        document.save(update_fields=["chunk_count", "status", "error"])
        return len(texts)
    except Exception as exc:  # noqa: BLE001 - surface failure to the uploader
        document.status = SyllabusDocument.Status.FAILED
        document.error = str(exc)
        document.save(update_fields=["status", "error"])
        raise


def _store_document_figures(document) -> int:
    """Extract and persist the document's page images as DocumentFigure rows."""
    from django.core.files.base import ContentFile

    from apps.rag.models import DocumentFigure

    DocumentFigure.objects.filter(document=document).delete()
    figures = extract_figures(document.file.path)
    for f in figures:
        figure = DocumentFigure(
            document=document,
            page_number=f["page"],
            ordinal=_page_ordinal(document.id, f["page"]),
        )
        figure.image.save(
            _figure_filename(document.id, f["page"]),
            ContentFile(f["image"]),
            save=True,
        )
    return len(figures)


def _page_ordinal(document_id: int, page: int) -> int:
    from apps.rag.models import DocumentFigure

    return DocumentFigure.objects.filter(
        document_id=document_id, page_number=page
    ).count()


def _figure_filename(document_id: int, page: int) -> str:
    return f"doc{document_id}_p{page}.png"
