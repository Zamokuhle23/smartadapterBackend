"""Question-boundary detection + cropping for past-paper pages.

Pipeline (digital PDFs: PyMuPDF gives text + boxes + style natively):
  page dict (words/spans with size, bold, bbox)
    -> top-level anchors  ("6.", "10)") + sub-anchors ((a), (ii))
    -> regions (anchor .. next anchor, multi-page spans)
    -> crops (render once, slice, confidence)

Confidence: bold/large sequential anchors score high; regex-only hits get
flagged for QC. Nothing here needs OCR (tesseract) on digital PDFs.
"""

import re

TOP_ANCHOR = re.compile(r"^\s*(\d{1,2})(?:[\.\)]|\s{2,}|\s*$)")
LOOSE_ANCHOR = re.compile(r"^\s*(\d{1,2})\s+\S")
SUB_ANCHOR = re.compile(r"^\s*\(?([a-z]|[ivx]+|[0-9]+)\)")
FOOTER = re.compile(r"©|specimen|turn over|^\s*page\s*\d+", re.IGNORECASE)
MARGIN_WORDS = re.compile(r"margin|do not write", re.IGNORECASE)
SUB_ANCHOR = re.compile(r"^\s*(\(?[a-z]\)|\(?[ivx]+\)|\([0-9]+\))")


def _line_text(line) -> str:
    return "".join(s.get("text", "") for s in line.get("spans", []))


def _line_size(line) -> float:
    sizes = [s.get("size", 0) for s in line.get("spans", []) if s.get("text", "").strip()]
    return max(sizes) if sizes else 0.0


def _line_bold(line) -> bool:
    return any((s.get("flags", 0) & 16) and s.get("text", "").strip()
               for s in line.get("spans", []))


def page_lines(page):
    """Flat lines with geometry: [(x0, top, text, size, bold)]."""
    out = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            text = _line_text(line)
            if not text.strip():
                continue
            x0, top, x1, bottom = line.get("bbox", (0, 0, 0, 0))
            out.append({"x0": x0, "top": top, "x1": x1, "bottom": bottom,
                        "text": text, "size": _line_size(line),
                        "bold": _line_bold(line)})
    out.sort(key=lambda ln: (round(ln["top"]), ln["x0"]))
    return out


def detect_anchors(lines, page_width: float, expected=None):
    """Top-level question starts: number + (bold|large) + left zone + sequence.

    expected threads across pages so numbering stays strictly sequential;
    returns (anchors, expected).
    """
    counts = {}
    for ln in lines:
        counts[ln["size"]] = counts.get(ln["size"], 0) + 1
    body_size = max(counts, key=lambda s: counts[s]) if counts else 10.0
    anchors = []
    for ln in lines:
        m = TOP_ANCHOR.match(ln["text"])
        loose = False
        if not m:
            # Single-space variant ("8 The function..."): only with bold,
            # left and exact sequence (kills "6 cm" false positives).
            m = LOOSE_ANCHOR.match(ln["text"])
            loose = True
        if not m:
            continue
        num = int(m.group(1))
        strong = ln["bold"] or ln["size"] >= body_size * 1.08
        left = ln["x0"] < page_width * 0.35
        if not left:
            continue
        if expected is None:
            if left and ln["size"] >= body_size * 0.99 and not loose:
                # First anchor: strict shape + body size (Cambridge numbers
                # are not bold).
                anchors.append({"number": str(num), "top": ln["top"],
                                "x0": ln["x0"], "confident": ln["bold"]})
                expected = num + 1
        elif num == expected and (strong or not loose):
            anchors.append({"number": str(num), "top": ln["top"],
                            "x0": ln["x0"],
                            "confident": strong and not loose})
            expected = num + 1
        elif (strong and not loose and expected is not None
                and expected < num <= expected + 3):
            # Papers occasionally skip a number: follow with low confidence
            # so one gap does not swallow the rest of the paper.
            anchors.append({"number": str(num), "top": ln["top"],
                            "x0": ln["x0"], "confident": False})
            expected = num + 1
    return anchors, expected


def detect_questions(doc):
    """Full-paper detection: [{number, pages: [(pno, top, bottom)], confident}]."""
    questions = []
    open_q = None
    expected = None
    page_lines_cache = {}
    for page in doc:
        pno = page.number + 1
        height = float(page.rect.height)
        lines = page_lines(page)
        page_lines_cache[pno] = lines
        anchors, expected = detect_anchors(lines, page.rect.width, expected)
        if open_q is not None:
            if anchors:
                # Previous question spills onto this page: close it here.
                open_q["pages"].append((pno, 0.0, anchors[0]["top"]))
                questions.append(open_q)
                open_q = None
            else:
                # Whole page belongs to the spilling question.
                open_q["pages"].append((pno, 0.0, height))
                continue
        for i, a in enumerate(anchors):
            bottom = anchors[i + 1]["top"] if i + 1 < len(anchors) else height
            q = {"number": a["number"],
                 "pages": [(pno, a["top"], bottom)],
                 "confident": a["confident"]}
            # Last anchor runs to page bottom: may continue next page.
            if i == len(anchors) - 1:
                open_q = q
            else:
                questions.append(q)
    if open_q is not None:
        questions.append(open_q)
    # Trim footer furniture (copyright / specimen / turn-over strips) from
    # page-bottom slices, then cap spans and drop slivers.
    for q in questions:
        trimmed = []
        for pno, top, bottom in q["pages"]:
            lines = page_lines_cache.get(pno, [])
            for ln in reversed(lines):
                if ln["top"] < top:
                    continue
                if FOOTER.search(ln["text"]):
                    bottom = min(bottom, ln["top"])
                else:
                    break
            trimmed.append((pno, top, bottom))
        q["pages"] = trimmed[:3]
    # Content x-bounds per question (trims margin strips for the crop).
    for q in questions:
        xs = []
        for pno, top, bottom in q["pages"]:
            for ln in page_lines_cache.get(pno, []):
                if not (top <= ln["top"] < bottom):
                    continue
                if not ln["text"].strip():
                    continue
                if FOOTER.search(ln["text"]):
                    continue
                if "MARGIN" in ln["text"].upper():
                    continue
                xs.append(ln["x0"])
                xs.append(ln["x1"])
        q["x0"] = max(0.0, min(xs) - 12) if xs else 0.0
        q["x1"] = max(xs) + 12 if xs else None
    kept = []
    for q in questions:
        total = sum(b - t for _, t, b in q["pages"])
        if total >= 40 or len(q["pages"]) > 1:
            kept.append(q)
    return kept


def crop_question(page_images, question, zoom: float):
    """Slice page renders into question crops. Returns [PNG bytes]."""
    from io import BytesIO

    out = []
    for pno, top, bottom in question["pages"]:
        img = page_images[pno]
        w, h = img.size
        x0 = max(0, int((question.get("x0") or 0) * zoom))
        x1 = question.get("x1")
        box = (x0, max(0, int(top * zoom)),
               min(w, int(x1 * zoom)) if x1 else w,
               min(h, int(bottom * zoom)))
        if box[3] <= box[1]:
            continue
        buf = BytesIO()
        img.crop(box).save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def _norm_suffix(raw: str) -> str:
    return raw.strip().strip("()")


def detect_parts(page, top: float, bottom: float):
    """Sub-anchors ((a), (b), (i)...) inside one question region.

    Returns [(suffix, top)] in reading order; empty when the question has
    no lettered parts (it stays a single tappable anchor).
    """
    lines = [ln for ln in page_lines(page)
             if top <= ln["top"] < bottom and ln["text"].strip()]
    width = page.rect.width
    parts = []
    for ln in lines:
        m = SUB_ANCHOR.match(ln["text"])
        if not m or ln["x0"] > width * 0.4:
            continue
        parts.append((_norm_suffix(m.group(1)), ln["top"]))
    return parts


def part_kind(page, top: float, bottom: float) -> str:
    """drawing when the region holds real vector work, else text."""
    import pymupdf

    box = pymupdf.Rect(0, top, page.rect.width, bottom)
    hits = 0
    for d in page.get_drawings():
        r = d.get("rect")
        if r is None:
            continue
        f = pymupdf.Rect(r)
        f = pymupdf.Rect(f.x0 - 1, f.y0 - 1, f.x1 + 1, f.y1 + 1)
        if not (f.x1 < box.x0 or box.x1 < f.x0 or
                f.y1 < box.y0 or box.y1 < f.y0):
            hits += 1
            if hits >= 4:
                return "drawing"
    return "text"


def redact_zones(page):
    """Page regions to hide when displaying: headers/footers with admin
    furniture, barcode clusters, marginalia strips. Returned as PDF-point
    rects; the app paints them over so barcodes never show."""
    import pymupdf

    zones = []
    W, H = page.rect.width, page.rect.height
    words = page.get_text("words")

    def band_text(top, bottom):
        return " ".join(
            w[4] for w in words if w[1] >= top and w[3] <= bottom).lower()

    for top, bottom in ((0.0, H * 0.08), (H * 0.92, H)):
        joined = band_text(top, bottom)
        if any(k in joined for k in (
                "examiner", "candidate", "centre number", "signature",
                "examinations council", "specimen", "turn over", "©")):
            zones.append([0.0, top, float(W), float(bottom)])

    # Barcode clusters: many thin vertical rules in a narrow band.
    verticals = []
    for d in page.get_drawings():
        r = d.get("rect")
        if r is None:
            continue
        r = pymupdf.Rect(r)
        if r.width < 3 and r.height > 20:
            verticals.append(r)
    verticals.sort(key=lambda r: r.x0)
    run = []
    for r in verticals:
        if run and r.x0 - run[-1].x1 > 60:
            if len(run) >= 10:
                zones.append([run[0].x0 - 4, min(v.y0 for v in run),
                              run[-1].x1 + 4, max(v.y1 for v in run)])
            run = []
        run.append(r)
    if len(run) >= 10:
        zones.append([run[0].x0 - 4, min(v.y0 for v in run),
                      run[-1].x1 + 4, max(v.y1 for v in run)])

    # Marginalia strips (rotated "do not write" text zones).
    boxes = []
    for w in words:
        if MARGIN_WORDS.search(w[4]):
            boxes.append(pymupdf.Rect(w[0], w[1], w[2], w[3]))
    if boxes:
        x0 = min(b.x0 for b in boxes) - 8
        x1 = max(b.x1 for b in boxes) + 8
        y0 = min(b.y0 for b in boxes) - 8
        y1 = max(b.y1 for b in boxes) + 8
        zones.append([max(0.0, x0), max(0.0, y0),
                      min(float(W), x1), min(float(H), y1)])
    return zones


def anchor_text(doc_path: str, page_number: int, bbox) -> str:
    """Native text inside an anchor bbox (no OCR engine needed)."""
    import pymupdf

    with pymupdf.open(doc_path) as pdf:
        page = pdf[page_number - 1]
        clip = pymupdf.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
        return page.get_text("text", clip=clip).strip()[:4000]


def anchor_marks(text: str) -> int | None:
    """Marks printed like [2] at the end of a question, if present."""
    import re

    marks = re.findall(r"\[(\d{1,2})\]", text or "")
    return int(marks[-1]) if marks else None
