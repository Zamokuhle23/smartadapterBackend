"""
Document ingestion pipeline: extract text -> chunk -> embed -> index.

Runs synchronously when Celery is not configured (CELERY_TASK_ALWAYS_EAGER or
no Redis), and as a background task otherwise.
"""

import os
from io import BytesIO

from django.conf import settings
from PIL import Image

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


_PAGE_ZOOM = 2.0  # ~144 DPI: one render per page, regions are crops


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
            kept_rasters = _raster_figures(doc, page, page_no)
            figures.extend(kept_rasters)
            raster_rects = [f["rect"] for f in kept_rasters
                            if f["rect"] is not None]

            # --- 2. Vector drawing regions (crops of one page render) ---
            try:
                regions = _vector_region_rects(page)
                regions = _split_giants(page, regions)
            except Exception:  # noqa: BLE001
                regions = []
            page_img = None
            if regions:
                try:
                    pix = page.get_pixmap(
                        matrix=pymupdf.Matrix(_PAGE_ZOOM, _PAGE_ZOOM))
                    page_img = Image.open(
                        BytesIO(pix.tobytes("png"))).convert("RGB")
                except Exception:  # noqa: BLE001
                    page_img = None
            for rect in regions:
                if any(_iou(rect, r) > 0.5 for r in raster_rects):
                    continue  # a kept raster already covers it
                if page_img is None:
                    continue
                try:
                    data = _crop_png(page_img, rect)
                except Exception:  # noqa: BLE001
                    continue
                # Byte size lies for clean line art (mostly white
                # compresses tiny): judge by dark-pixel density instead.
                if data and _dark_density(page_img, rect) >= 0.002:
                    figures.append({
                        "page": page_no,
                        "rect": [rect.x0, rect.y0, rect.x1, rect.y1],
                        "image": data,
                        "kind": "vector",
                    })
    figures.sort(key=lambda f: (f["page"], f["rect"][1] if f["rect"] else 0,
                                f["rect"][0] if f["rect"] else 0))
    return figures


def _iou(a, b) -> float:
    """Intersection-over-union for two rects (lists or pymupdf.Rects)."""
    import pymupdf

    if not isinstance(a, pymupdf.Rect):
        a = pymupdf.Rect(a)
    if not isinstance(b, pymupdf.Rect):
        b = pymupdf.Rect(b)
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    union = ((a.x1 - a.x0) * (a.y1 - a.y0)
             + (b.x1 - b.x0) * (b.y1 - b.y0) - inter)
    return inter / union if union > 0 else 0.0


#: Words marking admin furniture rather than exam content.
_FURNITURE_WORDS = (
    "examiner", "candidate", "centre number", "candidate number",
    "signature", "examinations council of eswatini",
    "do not write in this margin", "do not write in this space",
    "blank page",
)


def _region_has_furniture(words, rect) -> bool:
    joined = " ".join(
        w[4].lower() for w in words
        if w[0] >= rect.x0 - 2 and w[2] <= rect.x1 + 2
        and w[1] >= rect.y0 - 2 and w[3] <= rect.y1 + 2
    )
    return any(k in joined for k in _FURNITURE_WORDS)


def _raster_figures(doc, page, page_no: int):
    """Embedded images worth keeping, with rects for dedupe/filtering."""
    import pymupdf

    out = []
    words = page.get_text("words")
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            pix = pymupdf.Pixmap(doc, xref)
        except Exception:  # noqa: BLE001
            continue
        try:
            if pix.width < 40 or pix.height < 40:
                continue
            if pix.n - pix.alpha < 4:
                data = pix.tobytes("png")
            else:
                data = pymupdf.Pixmap(pymupdf.csRGB, pix).tobytes("png")
        except Exception:  # noqa: BLE001
            continue
        finally:
            pix = None
        if len(data) <= 2000:
            continue
        rects = page.get_image_rects(xref)
        rect = rects[0] if rects else None
        if rect is not None:
            if _region_has_furniture(words, rect):
                continue
            # Cover-page logos live in the top band; real diagrams don't.
            if page_no == 1 and rect.y1 < page.rect.height * 0.15:
                continue
            rect = [rect.x0, rect.y0, rect.x1, rect.y1]
        out.append({"page": page_no, "rect": rect,
                    "image": data, "kind": "raster"})
    return out
def _crop_png(page_img, rect) -> bytes:
    """Crop a page render to a PDF-point rect, return PNG bytes."""
    w, h = page_img.size
    box = (max(0, int(rect.x0 * _PAGE_ZOOM)),
           max(0, int(rect.y0 * _PAGE_ZOOM)),
           min(w, int(rect.x1 * _PAGE_ZOOM)),
           min(h, int(rect.y1 * _PAGE_ZOOM)))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("empty crop")
    buf = BytesIO()
    page_img.crop(box).save(buf, format="PNG")
    return buf.getvalue()


def _dark_density(page_img, rect) -> float:
    """Fraction of dark pixels inside rect (white-background diagrams ~1%+)."""
    w, h = page_img.size
    box = (max(0, int(rect.x0 * _PAGE_ZOOM)),
           max(0, int(rect.y0 * _PAGE_ZOOM)),
           min(w, int(rect.x1 * _PAGE_ZOOM)),
           min(h, int(rect.y1 * _PAGE_ZOOM)))
    crop = page_img.crop(box).convert("L")
    hist = crop.histogram()
    area = max(1, crop.size[0] * crop.size[1])
    return sum(hist[:128]) / area


def _touches_rect(path_rect, region) -> bool:
    """Overlap test safe for zero-area rule rects (fattened first)."""
    import pymupdf

    f = pymupdf.Rect(path_rect)
    f = pymupdf.Rect(f.x0 - 1, f.y0 - 1, f.x1 + 1, f.y1 + 1)
    r = region
    return not (f.x1 < r.x0 or r.x1 < f.x0 or f.y1 < r.y0 or r.y1 < f.y0)


def _split_giants(page, regions):
    """Re-split page-dominating fusions (table+grid chains) at margin 0."""
    import pymupdf

    pr = page.rect
    drawings = [d for d in page.get_drawings() if d.get("rect") is not None]
    out = []
    for rect in regions:
        if rect.width * rect.height < pr.width * pr.height * 0.6:
            out.append(rect)
            continue
        subs = [d for d in drawings if _touches_rect(d["rect"], rect)]
        parts = _vector_region_rects(page, margin=0.0, min_size=80,
                                     drawings=subs)
        # Accept the refinement when it isolated something genuinely
        # smaller (e.g. the grid out of a table+grid+answer-line fusion).
        # A lone table shatters into sub-size cells here; its content
        # survives in the text chunks, while the diagram is saved.
        orig_area = max(1.0, rect.width * rect.height)
        parts_area = sum(p.width * p.height for p in parts)
        if parts and parts_area < 0.7 * orig_area:
            out.extend(parts)
        else:
            out.append(rect)
    return out


def _vector_region_rects(page, margin: float = 3.0, min_size: int = 50,
                           drawings=None):
    """
    Cluster vector drawings into figure regions.

    ALL paths are clustered by proximity (thin line art has ~zero area, so
    there is no area pre-filter); merged regions are then filtered by size,
    shape and furniture content. Returns list[pymupdf.Rect].
    """
    import pymupdf

    if drawings is None:
        drawings = [d for d in page.get_drawings()
                    if d.get("rect") is not None]
    if not drawings:
        return []
    page_rect = page.rect
    union = [pymupdf.Rect(d["rect"]) for d in drawings]
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
                    m.include_rect(grown)
                    placed = True
                    changed = True
                    break
            if not placed:
                merged.append(pymupdf.Rect(grown))
        union = merged

    words = page.get_text("words")
    result = []
    for r in union:
        r = pymupdf.Rect(max(0, r.x0), max(0, r.y0),
                         min(page_rect.x1, r.x1), min(page_rect.y1, r.y1))
        if r.width < min_size or r.height < min_size:
            continue
        if r.width * r.height > page_rect.width * page_rect.height * 0.85:
            continue
        # Header/footer bands and full-width rules/answer lines.
        if r.y1 < page_rect.height * 0.08 and r.height < page_rect.height * 0.25:
            continue
        if r.y0 > page_rect.height * 0.92 and r.height < page_rect.height * 0.25:
            continue
        if r.width > page_rect.width * 0.5 and r.height < 15:
            continue
        if r.width < 8 and r.height > page_rect.height * 0.5:
            continue
        if _region_has_furniture(words, r):
            continue
        # Lone rules need company: a real figure is several paths. Path
        # rects are fattened first: pure horizontal/vertical rules are
        # zero-area ("empty") rects, which never intersect anything.
        def _touches(path_rect):
            f = pymupdf.Rect(path_rect)
            f = pymupdf.Rect(f.x0 - 1, f.y0 - 1, f.x1 + 1, f.y1 + 1)
            return f.intersects(r)

        hits = sum(1 for d in drawings if _touches(d["rect"]))
        if hits < 3:
            continue
        result.append(r)
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
            vec = vectors[i] if i < len(vectors) else None
            DocumentChunk.objects.create(
                syllabus=document.syllabus,
                document=document,
                subject=document.subject,
                ordinal=i,
                page_number=page_no or None,
                text=piece,
                embedding=vec,
                # pgvector search reads this column: without it the chunk
                # is invisible to retrieval on PostgreSQL.
                embedding_vec=vec if _pgvector_available() else None,
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


def _pgvector_available() -> bool:
    """True on PostgreSQL (the only backend with the vector column)."""
    from django.db import connection

    return connection.vendor == "postgresql"


def _store_document_figures(document) -> int:
    """Extract and persist figures, keeping identities stable.

    New regions are matched to existing rows by page + rect overlap (IoU):
    matches reuse their row and stable_key; genuinely new regions get fresh
    keys; rows with no counterpart are deleted (their questions stop serving
    them via the serve-time dangling check). Legacy rows without a bbox
    cannot be matched and are replaced.
    """
    from django.core.files.base import ContentFile

    from apps.rag.models import DocumentFigure
    from apps.syllabus.services.figure_keys import figure_key_for

    figures = extract_figures(document.file.path)
    existing = [f for f in DocumentFigure.objects.filter(document=document)]
    unmatched = list(existing)
    kept = 0
    # New regions must not reuse ordinals that existing rows (matched or
    # stale) still occupy; start past the page maximum.
    ordinal_counter = {}
    for f in existing:
        ordinal_counter[f.page_number] = max(
            ordinal_counter.get(f.page_number, -1), f.ordinal)

    def match(rect, page):
        best, best_iou = None, 0.0
        for cand in unmatched:
            if cand.page_number != page or not cand.bbox:
                continue
            iou = _iou(rect, cand.bbox) if rect else 0.0
            if iou > best_iou:
                best, best_iou = cand, iou
        if best is not None and best_iou > 0.5:
            unmatched.remove(best)
            return best
        return None

    for f in figures:
        page = f["page"]
        ordinal_counter[page] = ordinal_counter.get(page, -1) + 1
        row = match(f.get("rect"), page)
        if row is None:
            key = _unique_key(document, figure_key_for(document, page,
                                                       ordinal_counter[page]))
            row = DocumentFigure(document=document, page_number=page,
                                 ordinal=ordinal_counter[page],
                                 bbox=f.get("rect"), stable_key=key)
        else:
            if not row.stable_key:
                row.stable_key = _unique_key(
                    document, figure_key_for(document, row.page_number,
                                             row.ordinal))
            if row.bbox != f.get("rect"):
                row.bbox = f.get("rect")
        if not row.image or not row.image.storage.exists(row.image.name):
            row.image.save(f"{row.stable_key}.png",
                           ContentFile(f["image"]), save=False)
        row.save()
        kept += 1
    for stale in unmatched:
        stale.delete()
    return kept


def _unique_key(document, base: str) -> str:
    """First free stable_key for this document (collision suffix)."""
    from apps.rag.models import DocumentFigure

    key, i = base, 2
    while DocumentFigure.objects.filter(stable_key=key).exists():
        key = f"{base}-{i}"
        i += 1
    return key


def verify_cover_subject(path: str, codes) -> bool:
    """True when the cover (first 2 pages) mentions one of the subject codes.

    Decided by 4-digit numbers, not text length: a readable cover naming a
    DIFFERENT code (e.g. 6882 for a 6880 file) is quarantined, while a cover
    with no codes at all (scanned image) passes - never quarantine blind.
    """
    try:
        cover = "\n".join(extract_pages(path)[:2])
    except Exception:  # noqa: BLE001
        return True
    found = set(__import__("re").findall(r"\d{4}", cover))
    if not found:
        return True
    return bool(found & {str(c) for c in codes if c})


