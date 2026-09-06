"""Render the aggregate Markdown handover to PDF (reportlab 4.4+)."""
import argparse
import html
from pathlib import Path
import re
import textwrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, Preformatted


def render(markdown, output):
    source_text = Path(markdown).read_text()
    status = "Research results pending" if "No completed, frozen-test Benin results" in source_text else "Completed analysis"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Body", fontName="Helvetica", fontSize=9.4, leading=13.5, spaceAfter=7,
                             textColor=colors.HexColor("#233047")))
    styles.add(ParagraphStyle("SmallCell", parent=styles["Body"], fontSize=8.0, leading=10.4, spaceAfter=0))
    styles.add(ParagraphStyle("Section", parent=styles["Heading2"], fontSize=12, leading=16, spaceBefore=12,
                             spaceAfter=7, textColor=colors.HexColor("#163d59")))
    styles.add(ParagraphStyle("CodeLine", fontName="Courier", fontSize=7.1, leading=9.8, spaceAfter=3,
                             backColor=colors.HexColor("#f2f5f7"), borderPadding=5))
    def inline(text):
        text = html.escape(text)
        text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" color="#17618c">\1</a>', text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8">\1</font>', text)
        return text
    story, paragraph, table, code = [], [], [], None
    def flush():
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), styles["Body"]))
            paragraph.clear()
        if table:
            cells = [[Paragraph(inline(c), styles["SmallCell"]) for c in row] for row in table]
            widths = [126, 160, 213] if len(cells[0]) == 3 else [499 / len(cells[0])] * len(cells[0])
            t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e1ecf3")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 7),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                                   ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#9db1c0")),
                                   ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor("#d9e1e7"))]))
            story.append(t)
            story.append(Spacer(1, 8))
            table.clear()
    for line in source_text.splitlines():
        if line.startswith("```"):
            flush()
            if code is None:
                code = []
            else:
                for command in code:
                    story.append(Paragraph(html.escape(command), styles["CodeLine"]))
                story.append(Spacer(1, 8))
                code = None
            continue
        if code is not None:
            code.append(line)
        elif line.startswith("# "):
            flush()
            story.append(Paragraph(inline(line[2:]), styles["Title"]))
            story.append(Paragraph("6 September 2026 | Internship handover | " + status, styles["Body"]))
        elif line.startswith("## "):
            flush()
            story.append(Paragraph(inline(line[3:]), styles["Section"]))
        elif line.startswith("|"):
            if paragraph:
                flush()
            if not re.match(r"^\|[\s:|\-]+$", line):
                table.append([c.strip() for c in line.strip("|").split("|")])
        elif not line.strip():
            flush()
        else:
            paragraph.append(line)
    flush()
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#d9e1e7"))
        canvas.line(48, 38, A4[0] - 48, 38)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5b6779"))
        canvas.drawString(48, 25, "Benin paired-depth DANN | Aggregate information only")
        canvas.drawRightString(A4[0] - 48, 25, str(doc.page))
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=48, rightMargin=48, topMargin=42, bottomMargin=52,
                            title="Benin paired-depth adaptation: internship handover", author="Luis Falke")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("markdown")
    p.add_argument("output")
    args = p.parse_args()
    render(args.markdown, args.output)
