"""
PDF report builder
==================
Renders the EXACT structured report document produced by
`services.report_data.build_report` (the same dict the JSON report endpoint
returns) into a professional A4 PDF. It never recomputes or reinterprets
predictions — no model code runs here.

Design rules (master plan Part E):
- clean, professional typography; no decorative styling
- title/date/analysis ID, executive summary, media metadata table, timeline
  chart (videos), face + LIME images, technical info, warnings, limitations
- explicit research limitations + uncalibrated-score framing on every copy
- page numbers; graceful handling of missing charts/artifacts
"""

import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, Image as RLImage, KeepTogether, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

logger = logging.getLogger("deepfake.pdf")

# ---- palette (print-friendly) --------------------------------------------
INK = colors.HexColor("#0F172A")
MUTED = colors.HexColor("#64748B")
LINE = colors.HexColor("#D8DFE6")
ACCENT = colors.HexColor("#0E7490")
REAL = colors.HexColor("#15803D")
FAKE = colors.HexColor("#BE123C")
SURFACE = colors.HexColor("#F4F7FA")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm

_TITLE = "DeepFake Analysis Report"
_FOOTER_NOTE = (
    "Research prototype — model scores are uncalibrated outputs, not proof of "
    "manipulation or authenticity."
)


def _styles():
    base = dict(fontName="Helvetica", textColor=INK, fontSize=9.5, leading=13.5)
    return {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=20,
                                leading=24, textColor=INK),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=9.5,
                                   leading=13, textColor=MUTED),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12.5,
                             leading=16, textColor=INK, spaceBefore=4),
        "body": ParagraphStyle("body", **base),
        "muted": ParagraphStyle("muted", fontName="Helvetica", fontSize=8.5,
                                leading=12, textColor=MUTED),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=9, leading=12,
                               textColor=INK),
        "cell_muted": ParagraphStyle("cell_muted", fontName="Helvetica", fontSize=8.5,
                                     leading=11.5, textColor=MUTED),
        "verdict": ParagraphStyle("verdict", fontName="Helvetica-Bold", fontSize=22,
                                  leading=26, alignment=TA_CENTER),
        "score": ParagraphStyle("score", fontName="Helvetica-Bold", fontSize=13,
                                leading=17, alignment=TA_CENTER, textColor=ACCENT),
    }


def _kv_table(rows, col1=55 * mm):
    """Two-column metadata table: muted label | monospaced-ish value."""
    data = [[Paragraph(f"<font color='#64748B'>{k}</font>", _styles()["cell_muted"]),
             Paragraph(str(v), _styles()["cell"])] for k, v in rows]
    t = Table(data, colWidths=[col1, PAGE_W - 2 * MARGIN - col1], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _section_heading(text):
    return [
        Paragraph(text, _styles()["h2"]),
        HRFlowable(width="100%", thickness=0.8, color=LINE, spaceAfter=6),
    ]


def _timeline_chart(report: dict) -> Optional[bytes]:
    """Render the window-score timeline with matplotlib using REAL timestamps."""
    windows = (report.get("timeline") or {}).get("windows") or []
    pts = [(w.get("start_seconds"), w.get("end_seconds"), w.get("score"), w.get("label"))
           for w in windows]
    pts = [p for p in pts if p[2] is not None]
    if not pts:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mids, scores = [], []
        for start, end, score, _label in pts:
            start = start or 0.0
            end = end if end is not None else start
            mids.append((start + end) / 2)
            scores.append(score)
        threshold = (report.get("timeline") or {}).get("threshold")

        fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=160)
        ax.plot(mids, scores, "-o", color="#0E7490", linewidth=1.6, markersize=4)
        if threshold is not None:
            ax.axhline(threshold, color="#BE123C", linestyle="--", linewidth=1.1)
            ax.text(0.995, threshold, f" threshold {threshold}", va="bottom", ha="right",
                    fontsize=7.5, color="#BE123C", transform=ax.get_yaxis_transform())
        ax.set_xlabel("video time (s)", fontsize=8, color="#64748B")
        ax.set_ylabel("fake score", fontsize=8, color="#64748B")
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=8, colors="#64748B")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#D8DFE6")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001 - chart is optional
        logger.warning("Timeline chart rendering failed: %s", exc)
        return None


def _read_artifact(artifacts_dir: Path, artifact_id: Optional[str]) -> Optional[bytes]:
    """Load an artifact image strictly by validated id (no path composition from user data)."""
    import re

    if not artifact_id or not re.fullmatch(r"[A-Za-z0-9_]{4,64}", artifact_id):
        return None
    path = artifacts_dir / f"{artifact_id}.png"
    try:
        if path.parent != artifacts_dir.resolve() or not path.is_file():
            return None
        return path.read_bytes()
    except OSError:
        return None


def _image_flowables(png_bytes: Optional[bytes], caption: str, missing: str):
    if png_bytes:
        img = RLImage(io.BytesIO(png_bytes), width=64 * mm, height=64 * mm)
        return [img, Spacer(1, 2), Paragraph(caption, _styles()["muted"])]
    return [Paragraph(missing, _styles()["muted"])]


def build_pdf(report: dict, artifacts_dir: Path) -> bytes:
    """Render the structured report document into PDF bytes."""
    S = _styles()
    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=16 * mm, bottomMargin=16 * mm,
        title=_TITLE, author="DeepFake Forensics research platform",
    )
    frame = Frame(MARGIN, 16 * mm, PAGE_W - 2 * MARGIN, PAGE_H - 32 * mm, id="main")

    def _page_decor(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 9 * mm, _FOOTER_NOTE)
        canvas.drawRightString(PAGE_W - MARGIN, 9 * mm, f"Page {canvas.getPageNumber()}")
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 12 * mm, PAGE_W - MARGIN, 12 * mm)
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=_page_decor)])

    overview = report.get("overview") or {}
    media = report.get("media") or {}
    prediction = report.get("prediction") or {}
    technical = report.get("technical") or {}
    model = technical.get("model") or {}
    explain = report.get("explainability") or {}
    warnings = technical.get("warnings") or []
    limitations = report.get("limitations") or []
    assessment = overview.get("assessment") or "—"
    is_fake = str(assessment).upper() == "FAKE"

    story = []

    # ---- title block ----
    story.append(Paragraph(_TITLE, S["title"]))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "Explainable Deepfake Detection — Xception-LSTM with LIME · Research prototype",
        S["subtitle"]))
    analysis_id = report.get("analysis_id") or "—"
    created = overview.get("created_at") or "—"
    story.append(Spacer(1, 4))
    story.append(_kv_table([
        ("Analysis ID", analysis_id),
        ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
        ("Analyzed at", created),
        ("Status", report.get("generated_from_status") or "—"),
    ]))
    story.append(Spacer(1, 10))

    # ---- executive summary ----
    story += _section_heading("1. Executive summary")
    verdict_color = "#BE123C" if is_fake else "#15803D"
    exec_tbl = Table(
        [[Paragraph(f"<font color='{verdict_color}'>{assessment}</font>", S["verdict"]),
          Paragraph(f"raw fake score<br/><font size=14><b>{prediction.get('fake_score')}</b></font>",
                    S["score"]),
          Paragraph(f"decision threshold<br/><font size=11><b>{prediction.get('threshold')}"
                    f"</b></font> <font size=8 color='#64748B'>({prediction.get('threshold_version')})</font>",
                    S["score"]),
          Paragraph(f"model version<br/><font size=11><b>{overview.get('model_version') or '—'}</b></font>",
                    S["score"])]],
        colWidths=[(PAGE_W - 2 * MARGIN) / 4] * 4, hAlign="LEFT")
    exec_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
        ("BOX", (0, 0), (-1, -1), 0.8, LINE),
        ("LINEAFTER", (0, 0), (-2, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(exec_tbl)
    story.append(Spacer(1, 6))
    story.append(Paragraph(overview.get("assessment_basis") or "", S["body"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>This score is an uncalibrated sigmoid output — it is not a probability "
        "and not forensic proof.</b>", S["body"]))

    # ---- media information ----
    story += _section_heading("2. Media information")
    def fmt_size(n):
        try:
            n = float(n)
            return f"{n / (1024 * 1024):.2f} MB" if n >= 1024 * 1024 else f"{n / 1024:.1f} KB"
        except (TypeError, ValueError):
            return "—"
    story.append(_kv_table([
        ("Filename", media.get("filename", "—")),
        ("Media type", overview.get("media_type") or "—"),
        ("File size", fmt_size(media.get("size_bytes"))),
        ("Resolution", f"{media['width']}×{media['height']}" if media.get("width") and media.get("height") else "—"),
        ("Duration", f"{float(media['duration_seconds']):.2f} s" if media.get("duration_seconds") else "—"),
        ("FPS", media.get("fps", "—")),
        ("SHA-256", str(media.get("sha256") or "—")),
        ("Frames extracted / analyzed",
         f"{media.get('frames_extracted', '—')} / {media.get('frames_analyzed', '—')}"),
    ]))

    # ---- prediction analysis ----
    story += _section_heading("3. Prediction analysis")
    var = prediction.get("window_variation")
    method_names = {
        "full_sequence": "Full-sequence score",
        "sliding_window_ensemble": "Sliding-window ensemble (full 32-frame score + 10-frame windows, averaged)",
        "replicated_frame_sequence": "Replicated-frame compatibility mode (video model applied to a single face crop)",
    }
    rows = [
        ("Raw model output (fake score)", prediction.get("fake_score")),
        ("Decision method", method_names.get(prediction.get("method"), prediction.get("method"))),
        ("Decision threshold", f"{prediction.get('threshold')} ({prediction.get('threshold_version')})"),
        ("Score calibrated?", "No — sigmoid output, uncalibrated"),
        ("Model version", overview.get("model_version") or "—"),
        ("Window score variation",
         (f"min {var['min']:.4f} · max {var['max']:.4f} · stdev {var['stdev']:.4f}"
          if var else "— (no window scores)")),
    ]
    story.append(_kv_table(rows))

    # ---- timeline chart (videos with windows) ----
    chart_png = _timeline_chart(report)
    if chart_png:
        img = RLImage(io.BytesIO(chart_png), width=170 * mm, height=61 * mm)
        story.append(KeepTogether([
            *_section_heading("4. Video prediction timeline"),
            img,
            Spacer(1, 2),
            Paragraph(
                "Window scores are plotted against the source video timeline using fps and "
                "frame indices captured during face extraction. The dashed line marks the "
                "active decision threshold.", S["muted"]),
        ]))

    # ---- visual evidence ----
    face_id = lime_id = None
    artifacts = report.get("artifacts") or {}
    if isinstance(artifacts, dict):
        face_id = (artifacts.get("processed_face") or "").rsplit("/", 1)[-1] or None
        lime_id = (artifacts.get("lime") or "").rsplit("/", 1)[-1] or None
    if not lime_id:
        lime_id = explain.get("artifact_id")
    face_png = _read_artifact(artifacts_dir, face_id)
    lime_png = _read_artifact(artifacts_dir, lime_id)

    story += _section_heading("5. Visual evidence")
    col_w = (PAGE_W - 2 * MARGIN - 8 * mm) / 2
    face_col = _image_flowables(face_png, "Processed face (elliptical mask applied)",
                                "Processed face image not available.")
    lime_col = _image_flowables(lime_png, "LIME explanation heatmap",
                                "LIME was not requested or is unavailable for this analysis.")
    img_tbl = Table([[face_col, lime_col]], colWidths=[col_w + 4 * mm, col_w + 4 * mm],
                    hAlign="LEFT")
    img_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(img_tbl)
    lime_limitation = explain.get("limitation") or (
        "LIME highlights regions of a single replicated frame that most influenced the "
        "model's response. It does NOT prove those pixels were manipulated and does not "
        "explain the temporal (LSTM) pathway.")
    story.append(Spacer(1, 4))
    story.append(Paragraph(lime_limitation, S["muted"]))

    # ---- technical details ----
    story += _section_heading("6. Technical details")
    story.append(_kv_table([
        ("Model", f"{model.get('name', 'Xception-LSTM')} · version {model.get('version', '—')}"),
        ("Checkpoint SHA-256", str(model.get("checkpoint_sha256") or "—")),
        ("Sequence length", f"{model.get('input_frames', '—')} frames @ {model.get('input_size', '—')}×{model.get('input_size', '—')} px"),
        ("Preprocessing", "OpenCV DNN face detection (confidence 0.5) → 20% margin crop → "
                          "299×299 resize → elliptical background mask → [0, 1] normalization"),
        ("Prediction method", method_names.get(prediction.get("method"), prediction.get("method"))),
        ("Processing duration",
         f"{technical['processing_duration_seconds']:.1f} s"
         if technical.get("processing_duration_seconds") is not None else "—"),
    ]))

    # ---- interpretation ----
    story += _section_heading("7. Interpretation")
    story.append(Paragraph(report.get("interpretation") or "—", S["body"]))

    # ---- warnings ----
    if warnings:
        story += _section_heading("8. Warnings")
        warn_rows = [[Paragraph(f"• {w}", S["body"])] for w in warnings]
        wt = Table(warn_rows, colWidths=[PAGE_W - 2 * MARGIN], hAlign="LEFT")
        wt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFBEB")),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#F2D98D")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(wt)

    # ---- limitations ----
    story += _section_heading("9. Research limitations")
    lim_rows = [[Paragraph(f"{i + 1}. {l}", S["body"])] for i, l in enumerate(limitations)]
    if lim_rows:
        lt = Table(lim_rows, colWidths=[PAGE_W - 2 * MARGIN], hAlign="LEFT")
        lt.setStyle(TableStyle([
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(lt)

    doc.build(story)
    return buf.getvalue()


def pdf_filename(analysis_id: str) -> str:
    return f"deepfake_report_{(analysis_id or 'report')[:8]}.pdf"
