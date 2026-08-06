"""Render CODE_WALKTHROUGH.md to a phone-readable PDF.

Deliberately a small hand-rolled markdown subset rather than a general
converter: we need precise control over three things a generic renderer will
not give us — code blocks that never split across a page, long code lines that
wrap inside the margin instead of overflowing, and a page size that stays
legible when a phone fits it to width.

Page size is 6x9 inches rather than A4. On a phone, a PDF is scaled to fit the
screen width, so a narrower page means proportionally larger text. A4 code at
8pt is unreadable on a handset; the same content on a 6x9 page is fine.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

PAGE_W, PAGE_H = 6 * inch, 9 * inch
MARGIN = 0.5 * inch
TEXT_W = PAGE_W - 2 * MARGIN

CODE_FONT_SIZE = 7.2
CODE_LEADING = 9.4
# Courier advance width is exactly 0.6 em. Work out how many characters fit
# inside the frame minus the code block's own padding, then wrap there.
CODE_PAD = 6
CODE_CHARS = int((TEXT_W - 2 * CODE_PAD - 4) / (CODE_FONT_SIZE * 0.6))

CODE_BG = colors.Color(0.965, 0.965, 0.945)
CODE_BORDER = colors.Color(0.86, 0.86, 0.83)
RULE = colors.Color(0.80, 0.80, 0.80)


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s: dict[str, ParagraphStyle] = {}

    # Generous leading throughout: this is read on a phone, one thumb-scroll at
    # a time, and tight leading is what makes long technical prose exhausting.
    s["body"] = ParagraphStyle(
        "body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.6,
        leading=15.2, spaceBefore=0, spaceAfter=8, alignment=TA_LEFT,
    )
    s["h1"] = ParagraphStyle(
        "h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=17,
        leading=21, spaceBefore=2, spaceAfter=10, textColor=colors.black,
    )
    s["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13.5,
        leading=17.5, spaceBefore=16, spaceAfter=7, textColor=colors.Color(0.09, 0.09, 0.12),
    )
    s["h3"] = ParagraphStyle(
        "h3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=11,
        leading=14.5, spaceBefore=13, spaceAfter=5, textColor=colors.Color(0.16, 0.16, 0.2),
    )
    s["h4"] = ParagraphStyle(
        "h4", parent=base["Heading4"], fontName="Helvetica-BoldOblique", fontSize=9.8,
        leading=13, spaceBefore=10, spaceAfter=4, textColor=colors.Color(0.25, 0.25, 0.3),
    )
    s["code"] = ParagraphStyle(
        "code", parent=base["Code"], fontName="Courier", fontSize=CODE_FONT_SIZE,
        leading=CODE_LEADING, textColor=colors.Color(0.08, 0.08, 0.08),
        leftIndent=0, rightIndent=0, spaceBefore=0, spaceAfter=0,
    )
    s["bullet"] = ParagraphStyle(
        "bullet", parent=s["body"], leftIndent=15, bulletIndent=4, spaceAfter=4.5, leading=14.4,
    )
    s["quote"] = ParagraphStyle(
        "quote", parent=s["body"], leftIndent=12, rightIndent=6,
        borderColor=colors.Color(0.55, 0.55, 0.6), borderWidth=0, borderPadding=0,
        fontName="Helvetica-Oblique", textColor=colors.Color(0.2, 0.2, 0.25),
        spaceBefore=6, spaceAfter=10,
    )
    s["toc1"] = ParagraphStyle("toc1", fontName="Helvetica-Bold", fontSize=10,
                               leading=16, spaceBefore=7)
    s["toc2"] = ParagraphStyle("toc2", fontName="Helvetica", fontSize=9,
                               leading=13.5, leftIndent=13)
    s["tablecell"] = ParagraphStyle("tablecell", fontName="Helvetica", fontSize=7.4, leading=9.8)
    s["tablehead"] = ParagraphStyle("tablehead", fontName="Helvetica-Bold", fontSize=7.4, leading=9.8)
    return s


def inline(text: str) -> str:
    """Convert inline markdown to reportlab markup, escaping XML first."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Inline code: a subtle tint rather than a box, so it does not fragment
    # the line height in running prose.
    text = re.sub(
        r"`([^`]+)`",
        r'<font face="Courier" size="8.4" color="#8a1c1c">\1</font>',
        text,
    )
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    return text


def wrap_code(line: str, width: int) -> list[str]:
    """Wrap one code line, preserving indentation on continuations.

    Long lines are the main way code overflows a narrow page. Rather than
    shrink the whole block (which would make short blocks unreadable too), we
    wrap only what needs it and mark continuations so the reader can see the
    line is logically one line.
    """
    if len(line) <= width:
        return [line]
    indent = len(line) - len(line.lstrip(" "))
    # Continuation indent: original indent plus a marker, but never so deep
    # that the continuation has no room left.
    hang = min(indent + 4, max(width - 24, 4))
    out: list[str] = []
    remaining = line
    first = True
    while remaining:
        limit = width if first else width - hang
        if len(remaining) <= limit:
            out.append(remaining if first else " " * hang + "\\ " + remaining.lstrip())
            break
        # Prefer to break at a space near the limit so tokens stay intact.
        cut = remaining.rfind(" ", max(0, limit - 22), limit)
        if cut <= indent:
            cut = limit
        piece, remaining = remaining[:cut], remaining[cut:].lstrip()
        out.append(piece if first else " " * hang + "\\ " + piece.lstrip())
        first = False
    return out


def code_block(lines: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    """A code block as a single-cell table, so it gets a background and border."""
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(wrap_code(line.rstrip("\n"), CODE_CHARS))
    body = "\n".join(wrapped) if wrapped else " "
    pre = Preformatted(body, styles["code"])
    table = Table([[pre]], colWidths=[TEXT_W])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                ("BOX", (0, 0), (-1, -1), 0.4, CODE_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), CODE_PAD),
                ("RIGHTPADDING", (0, 0), (-1, -1), CODE_PAD),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def md_table(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    n = max(len(r) for r in rows)
    data = []
    for i, row in enumerate(rows):
        padded = row + [""] * (n - len(row))
        style = styles["tablehead"] if i == 0 else styles["tablecell"]
        data.append([Paragraph(inline(c), style) for c in padded])
    # First column carries the labels and needs the room; the rest share.
    first = TEXT_W * (0.30 if n > 2 else 0.5)
    widths = [first] + [(TEXT_W - first) / (n - 1)] * (n - 1) if n > 1 else [TEXT_W]
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.91, 0.91, 0.93)),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.Color(0.75, 0.75, 0.78)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


class Doc(BaseDocTemplate):
    """Doc template that records headings for the TOC and numbers pages."""

    def __init__(self, path: str, **kw: object) -> None:
        """Build the doc template with a single full-page body frame."""
        super().__init__(path, **kw)  # type: ignore[arg-type]
        frame = Frame(MARGIN, MARGIN + 16, TEXT_W, PAGE_H - 2 * MARGIN - 16, id="body")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=self._decorate)])

    def _decorate(self, canvas: object, doc: object) -> None:  # noqa: ARG002
        canvas.saveState()  # type: ignore[attr-defined]
        page = canvas.getPageNumber()  # type: ignore[attr-defined]
        if page > 1:
            canvas.setStrokeColor(RULE)  # type: ignore[attr-defined]
            canvas.setLineWidth(0.3)  # type: ignore[attr-defined]
            canvas.line(MARGIN, MARGIN + 26, PAGE_W - MARGIN, MARGIN + 26)  # type: ignore[attr-defined]
            canvas.setFont("Helvetica", 7.5)  # type: ignore[attr-defined]
            canvas.setFillColor(colors.Color(0.4, 0.4, 0.4))  # type: ignore[attr-defined]
            canvas.drawString(MARGIN, MARGIN + 14, "Credit Risk Decisioning — Code Walkthrough")  # type: ignore[attr-defined]
            canvas.drawRightString(PAGE_W - MARGIN, MARGIN + 14, str(page))  # type: ignore[attr-defined]
        canvas.restoreState()  # type: ignore[attr-defined]

    def afterFlowable(self, flowable: object) -> None:  # noqa: N802  (reportlab hook name)
        """Feed headings into the TOC with their real page numbers."""
        if not isinstance(flowable, Paragraph):
            return
        name = flowable.style.name
        if name not in ("h2", "h3"):
            return
        level = 0 if name == "h2" else 1
        text = re.sub(r"<[^>]+>", "", flowable.getPlainText())
        self.notify("TOCEntry", (level, text, self.page))


def convert(md_path: Path, pdf_path: Path) -> None:
    styles = build_styles()
    lines = md_path.read_text(encoding="utf-8").split("\n")

    story: list[object] = []
    i = 0
    first_h1 = True

    while i < len(lines):
        line = lines[i]

        # ---- fenced code ----
        if line.lstrip().startswith("```"):
            i += 1
            buf: list[str] = []
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            block = code_block(buf, styles)
            # KeepTogether stops a snippet being split across a page. Blocks
            # taller than a page cannot be kept whole, and reportlab handles
            # that by splitting anyway rather than erroring.
            story.append(Spacer(1, 3))
            story.append(KeepTogether(block))
            story.append(Spacer(1, 9))
            continue

        # ---- tables ----
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(
            r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]
        ):
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw = lines[i].strip().strip("|")
                if not re.match(r"^[\s:|-]+$", raw):
                    rows.append([c.strip() for c in raw.split("|")])
                i += 1
            story.append(Spacer(1, 3))
            story.append(KeepTogether(md_table(rows, styles)))
            story.append(Spacer(1, 10))
            continue

        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            story.append(Spacer(1, 5))
            i += 1
            continue

        # ---- headings ----
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            depth, text = len(m.group(1)), m.group(2)
            if depth == 1:
                if not first_h1:
                    story.append(PageBreak())
                first_h1 = False
                story.append(Paragraph(inline(text), styles["h1"]))
            elif depth == 2:
                # Major sections start on a fresh page: it makes the document
                # navigable on a phone, where you scroll rather than flip.
                if len(story) > 4:
                    story.append(PageBreak())
                story.append(Paragraph(inline(text), styles["h2"]))
            else:
                story.append(Paragraph(inline(text), styles[f"h{depth}"]))
            i += 1
            continue

        # ---- blockquote ----
        if stripped.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            story.append(Paragraph(inline(" ".join(buf)), styles["quote"]))
            continue

        # ---- lists ----
        if re.match(r"^([-*]|\d+\.)\s+", stripped):
            while i < len(lines) and re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                item = re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[i])
                marker = "•"
                num = re.match(r"^\s*(\d+)\.\s+", lines[i])
                if num:
                    marker = num.group(1) + "."
                story.append(Paragraph(inline(item), styles["bullet"], bulletText=marker))
                i += 1
            story.append(Spacer(1, 4))
            continue

        # ---- paragraph ----
        buf = []
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].lstrip().startswith("```")
            and not lines[i].strip().startswith("|")
            and not lines[i].strip().startswith(">")
            and not re.match(r"^#{1,4}\s", lines[i].strip())
            and not re.match(r"^([-*]|\d+\.)\s+", lines[i].strip())
            and lines[i].strip() != "---"
        ):
            buf.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline(" ".join(buf)), styles["body"]))

    # ---- table of contents, inserted after the title block ----
    toc = TableOfContents()
    toc.levelStyles = [styles["toc1"], styles["toc2"]]
    head = [
        Paragraph("Contents", styles["h1"]),
        Spacer(1, 6),
        toc,
        PageBreak(),
    ]
    # Keep the document's own title page, then TOC, then the body.
    split = 0
    for idx, flow in enumerate(story):
        if isinstance(flow, Paragraph) and flow.style.name == "h2":
            split = idx
            break
    story = story[:split] + head + story[split:]

    doc = Doc(
        str(pdf_path),
        pagesize=(PAGE_W, PAGE_H),
        title="Credit Risk Decisioning - Code Walkthrough",
        author="Code walkthrough",
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )
    # multiBuild resolves TOC page numbers, which need a second pass.
    doc.multiBuild(story)


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    convert(src, dst)
    print(f"wrote {dst.resolve()}  ({dst.stat().st_size / 1024:.0f} KB)")
