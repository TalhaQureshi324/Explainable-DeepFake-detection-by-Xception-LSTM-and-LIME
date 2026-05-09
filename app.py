"""
app.py
======
Gradio frontend for Deepfake Detection Model.
Supports English & Urdu language switching + Light/Dark theme toggle.
Deploy on Hugging Face Spaces.

Model files (upload ONE of these to the Space root):
  - deepfake_detector_model_v2.keras   (preferred, auto-detected)
  - deepfake_detector_model.keras      (fallback)

Use Git LFS for the .keras model file:
  git lfs track "*.keras"
"""

import os
import sys
import tempfile
import urllib.request
import traceback
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import cv2
import gradio as gr
from PIL import Image

# ============================== CONFIG ==============================

MODEL_PATH = "deepfake_detector_model_v2.keras"
FALLBACK_MODEL_PATH = "deepfake_detector_model.keras"

NUM_FRAMES = 32
WINDOW_SIZE = 10
STRIDE = 5
DECISION_THRESHOLD = 0.75

PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
)
CAFFEMODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)

PROTOTXT_PATH = "deploy.prototxt"
CAFFEMODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

LANG_MAP = {"English": "en", "اردو": "ur"}
THEME_MAP = {"Light": "light", "Dark": "dark"}

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# ============================== TRANSLATIONS ==============================

TEXTS = {
    "en": {
        "title": "# 🎭 DeepFake Detection",
        "desc": "Upload a **video** or **image** to detect whether it is **REAL** or **FAKE**.",
        "upload_label": "Upload Video or Image",
        "xai_label": "Generate XAI Explanation (LIME)",
        "xai_info": "Slower (~10-30s) but highlights regions that influenced the decision.",
        "analyze_btn": "🔍 Analyze",
        "preview_label": "Processed Face Frame",
        "xai_label_img": "XAI Heatmap",
        "result_label": "Result",
        "lang_label": "Language / زبان",
        "theme_label": "Theme",
        "light": "Light",
        "dark": "Dark",
        "tips": (
            "---\n"
            "**Tips for best results:**\n"
            "- Use videos/images with a **clear, frontal face**.\n"
            "- Avoid heavy occlusions (sunglasses, masks) if possible.\n"
            "- The model was trained on 299×299 face crops; low-resolution inputs may reduce accuracy."
        ),
        "no_model": "### ❌ Model not loaded.\nPlease upload `deepfake_detector_model_v2.keras` or `deepfake_detector_model.keras` to the Space root.",
        "no_file": "### ⚠️ Please upload a file first.",
        "no_faces_video": "### ❌ No faces detected in the video.\nTry a clip with clear frontal faces.",
        "no_faces_image": "### ❌ No face detected in the image.\nTry a photo with a clear frontal face.",
        "unsupported": "### ⚠️ Unsupported file format.\nUpload MP4, AVI, MOV, MKV, WEBM, JPG, JPEG, PNG, or BMP.",
        "pred_failed": "### ❌ Prediction failed (ensemble returned no scores).",
        "verdict_fake": "🟥 FAKE",
        "verdict_real": "🟩 REAL",
        "confidence": "Confidence",
        "avg_score": "Average Score",
        "std_dev": "Std Deviation",
        "windows": "Ensemble Windows",
        "input_type": "Input Type",
        "video_frames": "Video ({} face frames extracted)",
        "image_single": "Image (single frame)",
        "xai_not_installed": "⚠️ XAI requested but `lime` / `scikit-image` are not installed.",
        "xai_failed": "⚠️ XAI generation failed: {}",
        "analysis_crashed": "### ❌ Analysis Crashed",
        "error": "Error",
        "click_expand": "Click to expand traceback",
        "common_causes": (
            "**Common causes:**\n"
            "- Model expects a different frame count than provided.\n"
            "- Uploaded video uses a codec OpenCV can't read on Linux.\n"
            "- DNN face detector files are missing or corrupted.\n"
            "- Out-of-memory during prediction."
        ),
        "processing_video": "Preprocessing video (target {} frames)...",
        "processing_image": "Preprocessing image...",
        "running_ensemble": "Running ensemble prediction with {}-frame model...",
        "generating_xai": "Generating XAI heatmap...",
        "done": "Done.",
    },
    "ur": {
        "title": "# 🎭 ڈیپ فیک ڈیٹیکشن",
        "desc": "یہ جاننے کے لیے ایک **ویڈیو** یا **تصویر** اپ لوڈ کریں کہ یہ **اصلی** ہے یا **جعلی**۔",
        "upload_label": "ویڈیو یا تصویر اپ لوڈ کریں",
        "xai_label": "XAI وضاحت تیار کریں (LIME)",
        "xai_info": "آہستہ (~10-30s) لیکن وہ علاقے نمایاں کرتا ہے جو فیصلے پر اثر انداز ہوئے۔",
        "analyze_btn": "🔍 تجزیہ کریں",
        "preview_label": "پروسیس شدہ چہرے کا فریم",
        "xai_label_img": "XAI ہیٹ میپ",
        "result_label": "نتیجہ",
        "lang_label": "Language / زبان",
        "theme_label": "تھیم",
        "light": "روشن",
        "dark": "تاریک",
        "tips": (
            "---\n"
            "**بہترین نتائج کے لیے تجاویز:**\n"
            "- واضح اور سیدھے چہرے والے ویڈیوز/تصاویر استعمال کریں۔\n"
            "- ممکن ہو تو بھاری رکاوٹوں (دھوپ کا چشمہ، ماسک) سے گریز کریں۔\n"
            "- ماڈل 299×299 چہرے کے کروپس پر تربیت یافتہ ہے؛ کم ریزولوشن ان پٹس درستگی کم کر سکتے ہیں۔"
        ),
        "no_model": "### ❌ ماڈل لوڈ نہیں ہوا۔\nبراہ کرم `deepfake_detector_model_v2.keras` یا `deepfake_detector_model.keras` اپ لوڈ کریں۔",
        "no_file": "### ⚠️ براہ کرم پہلے فائل اپ لوڈ کریں۔",
        "no_faces_video": "### ❌ ویڈیو میں کوئی چہرہ نہیں ملا۔\nواضح سیدھے چہرے والے کلپ کے ساتھ کوشش کریں۔",
        "no_faces_image": "### ❌ تصویر میں کوئی چہرہ نہیں ملا۔\nواضح سیدھے چہرے والی تصویر کے ساتھ کوشش کریں۔",
        "unsupported": "### ⚠️ غیر تعاون یافتہ فائل فارمیٹ۔\nMP4، AVI، MOV، MKV، WEBM، JPG، JPEG، PNG، یا BMP اپ لوڈ کریں۔",
        "pred_failed": "### ❌ پیشن گوئی ناکام ہو گئی (انسائبل نے کوئی اسکور واپس نہیں کیے)۔",
        "verdict_fake": "🟥 جعلی",
        "verdict_real": "🟩 اصلی",
        "confidence": "اعتماد",
        "avg_score": "اوسط اسکور",
        "std_dev": "معیاری انحراف",
        "windows": "انسائبل ونڈوز",
        "input_type": "ان پٹ کی قسم",
        "video_frames": "ویڈیو ({} چہرے کے فریمز نکالے گئے)",
        "image_single": "تصویر (ایک فریم)",
        "xai_not_installed": "⚠️ XAI کی درخواست کی گئی لیکن `lime` / `scikit-image` انسٹال نہیں ہیں۔",
        "xai_failed": "⚠️ XAI تیار کرنے میں ناکامی: {}",
        "analysis_crashed": "### ❌ تجزیہ ناکام ہو گیا",
        "error": "خرابی",
        "click_expand": "تفصیل دیکھنے کے لیے کلک کریں",
        "common_causes": (
            "**عام وجوہات:**\n"
            "- ماڈل کے متوقع فریم کاؤنٹ اور فراہم کردہ میں فرق۔\n"
            "- اپ لوڈ کردہ ویڈیو کا کوڈیک لینکس پر OpenCV کے لیے ناقابل مطالعہ۔\n"
            "- DNN فیس ڈیٹیکٹر فائلیں غائب یا خراب ہیں۔\n"
            "- پیشن گوئی کے دوران میموری ختم۔"
        ),
        "processing_video": "ویڈیو پروسیسنگ (ہدف {} فریمز)...",
        "processing_image": "تصویر پروسیسنگ...",
        "running_ensemble": "{}-فریم ماڈل کے ساتھ انسائبل پیشن گوئی چل رہی ہے...",
        "generating_xai": "XAI ہیٹ میپ تیار ہو رہی ہے...",
        "done": "ہو گیا۔",
    },
}

# ============================== THEME CSS ==============================
# KEY FIX: Light mode uses CSS custom properties (--color-*) that Gradio
# actually reads, plus aggressive overrides for every component that was
# appearing dark. Dark mode was already working so it is unchanged.

THEME_CSS = {
    "light": """
<style id="app-theme-style">
  /* ── Gradio CSS variable overrides (the real fix) ── */
  :root, .gradio-container {
    --color-background-primary: #FFFFFF !important;
    --color-background-secondary: #F1F3F4 !important;
    --color-background-tertiary: #E8EAED !important;
    --color-text-primary: #202124 !important;
    --color-text-secondary: #5F6368 !important;
    --color-text-body: #202124 !important;
    --color-text-label: #202124 !important;
    --color-text-subdued: #5F6368 !important;
    --color-border-primary: #DADCE0 !important;
    --color-border-secondary: #E8EAED !important;
    --color-accent-soft: #E8F0FE !important;
    --body-text-color: #202124 !important;
    --body-text-color-subdued: #5F6368 !important;
    --block-background-fill: #FFFFFF !important;
    --block-border-color: #DADCE0 !important;
    --block-label-background-fill: #F1F3F4 !important;
    --block-label-text-color: #202124 !important;
    --block-title-text-color: #202124 !important;
    --input-background-fill: #FFFFFF !important;
    --input-border-color: #DADCE0 !important;
    --checkbox-background-color: #FFFFFF !important;
    --checkbox-border-color: #DADCE0 !important;
    --panel-background-fill: #F8F9FA !important;
    --panel-border-color: #E8EAED !important;
    --button-primary-background-fill: #1A73E8 !important;
    --button-primary-text-color: #FFFFFF !important;
    --button-secondary-background-fill: #F1F3F4 !important;
    --button-secondary-text-color: #202124 !important;
    --background-fill-primary: #FFFFFF !important;
    --background-fill-secondary: #F1F3F4 !important;
    --neutral-100: #F1F3F4 !important;
    --neutral-200: #E8EAED !important;
    --neutral-700: #5F6368 !important;
    --neutral-800: #3C4043 !important;
    --neutral-900: #202124 !important;
  }

  /* ── Page / container background ── */
  body,
  .gradio-container,
  .gradio-container > *,
  .main,
  footer { background: #FFFFFF !important; color: #202124 !important; }

  /* ── All text everywhere ── */
  *:not(button):not(.gr-button-primary):not([class*="btn-primary"]) {
    color: #202124 !important;
  }

  /* ── Markdown / prose ── */
  .gr-markdown,
  .gr-markdown p,
  .gr-markdown h1,
  .gr-markdown h2,
  .gr-markdown h3,
  .gr-markdown li,
  .gr-markdown strong,
  .prose, .prose p, .prose li, .prose strong,
  [class*="markdown"] p,
  [class*="markdown"] li,
  [class*="markdown"] strong { color: #202124 !important; }

  /* ── Labels ── */
  label, .label, span.label,
  .block-label, [class*="block-label"],
  [class*="label-wrap"] span { color: #202124 !important; }

  /* ── Header bar ── */
  .app-header {
    background: #F8F9FA !important;
    border: 1px solid #E8EAED !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin-bottom: 20px !important;
  }

  /* ── Gradio panels / blocks ── */
  .gr-box, .gr-panel, .gr-form, .gr-group,
  [class*="panel"], [class*="gr-box"],
  .block, .contain, .wrap,
  [data-testid="block"] { background: #FFFFFF !important; border-color: #DADCE0 !important; }

  /* ── Upload area ── */
  .upload-section [data-testid="block"],
  .upload-section .gr-file-upload,
  .upload-section .upload-container,
  .upload-section [class*="upload"],
  [class*="upload-box"],
  [class*="file-upload"],
  .svelte-uploadfile,
  [data-testid="upload-btn"],
  .file-preview,
  .upload-area,
  /* Target the actual drop zone rendered by Gradio */
  .gr-upload,
  .gr-file {
    background: #F1F3F4 !important;
    border: 2px dashed #BDC1C6 !important;
    color: #202124 !important;
    border-radius: 8px !important;
  }
  .upload-section [class*="upload"]:hover,
  [class*="upload-box"]:hover { background: #E8F0FE !important; border-color: #1A73E8 !important; }

  /* Gradio 4.x uses a <label> wrapping the drop zone */
  label[class*="upload"],
  label.svelte-upload { background: #F1F3F4 !important; border: 2px dashed #BDC1C6 !important; }

  /* ── Image output panels ── */
  .result-section [data-testid="image"],
  .result-section [class*="image-container"],
  .result-section .gr-image,
  [data-testid="image"],
  [class*="image-preview"],
  [class*="image-container"] {
    background: #F8F9FA !important;
    border: 1px solid #DADCE0 !important;
    border-radius: 8px !important;
  }
  /* Image placeholder icon area */
  [data-testid="image"] [class*="placeholder"],
  [class*="image-container"] [class*="placeholder"],
  [class*="empty-image"] {
    background: #F1F3F4 !important;
    color: #5F6368 !important;
  }

  /* ── Inputs / dropdowns / selects ── */
  input, select, textarea,
  .gr-input, .gr-dropdown, .gr-textarea,
  [class*="gr-input"], [class*="gr-dropdown"],
  [class*="input-container"] {
    background: #FFFFFF !important;
    color: #202124 !important;
    border-color: #DADCE0 !important;
  }
  input::placeholder, textarea::placeholder { color: #9AA0A6 !important; }

  /* ── Radio / Checkbox ── */
  .xai-option,
  [class*="checkbox-wrap"],
  [class*="radio-wrap"] {
    background: #FFFFFF !important;
    border: 1px solid #DADCE0 !important;
    border-radius: 8px !important;
    color: #202124 !important;
  }
  input[type="checkbox"], input[type="radio"] { accent-color: #1A73E8 !important; }

  /* ── Primary button (keep blue, white text) ── */
  .gr-button-primary,
  button[class*="primary"],
  button.primary {
    background: #1A73E8 !important;
    color: #FFFFFF !important;
    border: none !important;
  }
  .gr-button-primary:hover,
  button[class*="primary"]:hover { background: #1765CC !important; }

  /* ── Tables ── */
  table { border: 1px solid #DADCE0 !important; background: #FFFFFF !important; }
  th { background: #F1F3F4 !important; color: #202124 !important; border-color: #DADCE0 !important; }
  td { background: #FFFFFF !important; color: #202124 !important; border-bottom: 1px solid #F1F3F4 !important; }

  /* ── Scrollbars ── */
  ::-webkit-scrollbar-track { background: #F1F3F4 !important; }
  ::-webkit-scrollbar-thumb { background: #BDC1C6 !important; border-radius: 4px !important; }
  ::-webkit-scrollbar-thumb:hover { background: #9AA0A6 !important; }

  /* ── Dropdown menus ── */
  [class*="dropdown-menu"], [class*="options"],
  .gr-dropdown-menu { background: #FFFFFF !important; border-color: #DADCE0 !important; }
  [class*="dropdown-option"]:hover,
  [class*="option"]:hover { background: #E8F0FE !important; color: #202124 !important; }

  /* ── Code / pre blocks ── */
  pre, code { background: #F1F3F4 !important; color: #202124 !important; border: 1px solid #E8EAED !important; }

  /* ── Details/summary ── */
  details { background: #F8F9FA !important; border: 1px solid #E8EAED !important; border-radius: 8px !important; }
  summary { color: #202124 !important; }

  /* ── Force SVG icons to dark colour ── */
  svg path, svg rect, svg circle, svg line { stroke: #5F6368 !important; }
</style>
""",
    "dark": """
    <style id="app-theme-style">
      .gradio-container { background: #0A0E17 !important; }
      .app-header {
        background: #151B27 !important;
        border: 1px solid #1E293B !important;
        border-radius: 12px !important;
        padding: 10px 16px !important;
        margin-bottom: 16px !important;
      }
      .app-header label, .app-header .label {
        color: #F1F5F9 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
      }
      .app-header .gr-dropdown, .app-header .gr-radio {
        background: #1E293B !important;
        color: #E2E8F0 !important;
        border-color: #334155 !important;
        border-radius: 8px !important;
      }
      .gr-box, .gr-panel, .gr-form, .gr-padded, .gr-group {
        background: #151B27 !important;
        border: 1px solid #1E293B !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.2) !important;
      }
      label, .label, h1, h2, h3, h4, h5, h6 { color: #F1F5F9 !important; font-weight: 600 !important; }
      p, span, .prose { color: #F1F5F9 !important; }
      .gr-markdown, .gr-markdown p, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3, .gr-markdown h4 { color: #F1F5F9 !important; }
      .gr-markdown strong { color: #F1F5F9 !important; font-weight: 700 !important; }
      .gr-markdown a { color: #94A3B8 !important; border-bottom: 1px solid #334155 !important; text-decoration: none !important; }
      .gr-markdown a:hover { color: #E2E8F0 !important; border-bottom-color: #64748B !important; }
      .upload-section .gr-file-upload, .upload-section .upload-container {
        background: #0F172A !important;
        border: 2px dashed #334155 !important;
        border-radius: 12px !important;
      }
      .upload-section .gr-file-upload:hover, .upload-section .upload-container:hover {
        background: #1E293B !important;
        border-color: #64748B !important;
      }
      .result-section .gr-image {
        background: #151B27 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
      }
      .result-section .gr-image .label, .result-section .gr-image label {
        background: #1E293B !important;
        color: #F1F5F9 !important;
        font-weight: 600 !important;
        padding: 10px 14px !important;
        border-bottom: 1px solid #334155 !important;
        border-radius: 12px 12px 0 0 !important;
      }
      .xai-option {
        background: #0F172A !important;
        border: 1px solid #1E293B !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
      }
      .xai-option label { color: #F1F5F9 !important; font-weight: 600 !important; }
      .xai-option .info { color: #94A3B8 !important; font-size: 0.85rem !important; }
      .xai-option input[type="checkbox"] { accent-color: #64748B !important; }
      .gr-button-primary {
        background: #2B6CB0 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        transition: all 0.2s ease !important;
      }
      .gr-button-primary:hover { background: #3182CE !important; box-shadow: 0 6px 16px rgba(0,0,0,0.4) !important; transform: translateY(-1px) !important; }
      .gr-button-primary:active { transform: translateY(0px) scale(0.98) !important; }
      .gr-button-secondary { background: #1E293B !important; color: #E2E8F0 !important; border: 1px solid #334155 !important; border-radius: 10px !important; }
      .gr-input, .gr-textarea, .gr-dropdown, select, input[type="text"], input[type="number"] {
        background: #1E293B !important; color: #F1F5F9 !important; border-color: #334155 !important; border-radius: 8px !important;
      }
      .gr-input:focus, .gr-textarea:focus, .gr-dropdown:focus, select:focus {
        border-color: #64748B !important; box-shadow: 0 0 0 3px rgba(100, 116, 139, 0.15) !important; outline: none !important;
      }
      table { color: #F1F5F9 !important; border-collapse: separate !important; border-spacing: 0 !important; border-radius: 10px !important; overflow: hidden !important; border: 1px solid #334155 !important; }
      th { background: #1E293B !important; color: #F1F5F9 !important; font-weight: 700 !important; border-color: #334155 !important; padding: 12px 14px !important; text-transform: uppercase !important; font-size: 0.75rem !important; letter-spacing: 0.5px !important; }
      td { border-color: #1E293B !important; padding: 12px 14px !important; }
      tr:nth-child(even) td { background: #111827 !important; }
      tr:hover td { background: #1E293B !important; }
      .gr-checkradio, .gr-checkbox, .gr-radio { color: #F1F5F9 !important; }
      input[type="checkbox"], input[type="radio"] { accent-color: #64748B !important; }
      ::-webkit-scrollbar { width: 8px !important; height: 8px !important; }
      ::-webkit-scrollbar-track { background: #111827 !important; border-radius: 4px !important; }
      ::-webkit-scrollbar-thumb { background: #334155 !important; border-radius: 4px !important; }
      ::-webkit-scrollbar-thumb:hover { background: #475569 !important; }
      .gr-dropdown-menu, .options { background: #151B27 !important; border-color: #334155 !important; border-radius: 10px !important; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3) !important; }
      .gr-dropdown-option:hover, .options .option:hover { background: #1E293B !important; color: #F1F5F9 !important; }
      details { background: #0F172A !important; border: 1px solid #334155 !important; border-radius: 10px !important; padding: 12px 16px !important; }
      summary { color: #F1F5F9 !important; font-weight: 600 !important; cursor: pointer !important; }
      pre, code { background: #1E293B !important; color: #E2E8F0 !important; border-radius: 6px !important; border: 1px solid #334155 !important; }
    </style>
    """,
}

# ============================== DOWNLOAD DNN FILES ==============================


def download_file(url: str, path: str) -> None:
    if not os.path.exists(path):
        print(f"[DeepFake App] Downloading {path} ...")
        try:
            urllib.request.urlretrieve(url, path)
            size = os.path.getsize(path)
            if path.endswith(".prototxt") and size > 1_000_000:
                raise RuntimeError(f"Downloaded file {path} is too large ({size} bytes); likely an HTML error page.")
            if path.endswith(".caffemodel") and size < 1_000_000:
                raise RuntimeError(f"Downloaded file {path} is too small ({size} bytes); likely an HTML error page.")
            print(f"[DeepFake App] Downloaded: {path} ({size} bytes)")
        except Exception as exc:
            print(f"[DeepFake App] ERROR downloading {path}: {exc}")
            raise


try:
    download_file(PROTOTXT_URL, PROTOTXT_PATH)
    download_file(CAFFEMODEL_URL, CAFFEMODEL_PATH)
except Exception as e:
    print(f"[DeepFake App] WARNING: Could not download DNN detector files: {e}")


# ============================== FACE EXTRACTOR ==============================


class FaceExtractor:
    """OpenCV DNN face extractor with elliptical background removal."""

    def __init__(self):
        self.confidence_threshold = 0.5
        self.image_size = (299, 299)
        self.margin = 0.2
        self.do_remove_background = True
        self.face_mask_scale = (0.85, 0.90)

        if not os.path.exists(PROTOTXT_PATH):
            raise FileNotFoundError(f"Prototxt missing: {PROTOTXT_PATH}")
        if not os.path.exists(CAFFEMODEL_PATH):
            raise FileNotFoundError(f"CaffeModel missing: {CAFFEMODEL_PATH}")

        self.face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

    def detect_faces(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2, y2, confidence))

        faces.sort(key=lambda x: x[4], reverse=True)
        return faces[:1]

    def remove_background(self, face_img: np.ndarray) -> np.ndarray:
        h, w = face_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        scale_w, scale_h = self.face_mask_scale
        axes = (int(w * scale_w / 2), int(h * scale_h / 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        kernel_size = max(3, int(min(h, w) * 0.03))
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        mask = mask.astype(np.float32) / 255.0

        if len(face_img.shape) == 3:
            mask = np.expand_dims(mask, axis=-1)

        result = (face_img.astype(np.float32) * mask).astype(np.uint8)
        return result

    def extract_face(self, frame: np.ndarray, bbox):
        x1, y1, x2, y2 = bbox[:4]
        h, w = frame.shape[:2]
        margin_x = int((x2 - x1) * self.margin)
        margin_y = int((y2 - y1) * self.margin)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        face = cv2.resize(face, self.image_size)
        if self.do_remove_background:
            face = self.remove_background(face)
        return face

    def process_video(self, video_path: str, output_dir: str, max_frames: int = 32) -> int:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[FaceExtractor] WARNING: cv2.VideoCapture could not open {video_path}")
            return 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            print(f"[FaceExtractor] WARNING: total_frames={total_frames} for {video_path}")
            return 0

        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

        os.makedirs(output_dir, exist_ok=True)
        extracted_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in frame_indices:
                faces_data = self.detect_faces(frame)
                for face_idx, face_data in enumerate(faces_data):
                    face = self.extract_face(frame, face_data)
                    if face is not None:
                        out_path = os.path.join(
                            output_dir, f"frame_{extracted_count:04d}_face_{face_idx}.jpg"
                        )
                        cv2.imwrite(out_path, face)
                        extracted_count += 1
            frame_idx += 1
            if frame_idx > frame_indices[-1]:
                break

        cap.release()
        return extracted_count

    def process_image(self, image_path: str):
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        faces_data = self.detect_faces(frame)
        if not faces_data:
            return None
        face = self.extract_face(frame, faces_data[0])
        if face is None:
            return None
        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)


# ============================== PREPROCESSING ==============================


def preprocess_video(video_path: str, temp_dir: str, num_frames: int):
    extractor = FaceExtractor()
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    temp_output = os.path.join(temp_dir, video_stem)
    os.makedirs(temp_output, exist_ok=True)

    count = extractor.process_video(video_path, temp_output, max_frames=num_frames)
    if count == 0:
        return None, 0

    frames = []
    frame_files = sorted([f for f in os.listdir(temp_output) if f.endswith(".jpg")])
    for frame_file in frame_files:
        frame_path = os.path.join(temp_output, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    return frames, count


def preprocess_image(image_path: str):
    extractor = FaceExtractor()
    return extractor.process_image(image_path)


def pad_frames(frames, target_length: int):
    frames = list(frames)
    if len(frames) == 0:
        return None
    while len(frames) < target_length:
        frames.append(frames[-1].copy())
    return np.array(frames, dtype=np.float32)


# ============================== ENSEMBLE PREDICTION ==============================


def ensemble_predict(model, frames_list, num_frames: int, window_size: int, stride: int):
    if len(frames_list) == 0:
        return None, [], []

    normalized = [f.astype(np.float32) / 255.0 for f in frames_list]
    predictions = []
    labels = []

    full_seq = pad_frames(normalized, num_frames)
    if full_seq is not None:
        full_pred = float(model.predict(np.expand_dims(full_seq, axis=0), verbose=0)[0][0])
        predictions.append(full_pred)
        labels.append(f"Full-{num_frames}")

    n = len(normalized)
    if n >= window_size:
        for start in range(0, n - window_size + 1, stride):
            window = normalized[start : start + window_size]
            padded = pad_frames(window, num_frames)
            if padded is not None:
                pred = float(model.predict(np.expand_dims(padded, axis=0), verbose=0)[0][0])
                predictions.append(pred)
                labels.append(f"Win-{start}-{start + window_size - 1}")
    else:
        padded = pad_frames(normalized, num_frames)
        if padded is not None:
            pred = float(model.predict(np.expand_dims(padded, axis=0), verbose=0)[0][0])
            predictions.append(pred)
            labels.append(f"Short-{n}")

    avg_pred = float(np.mean(predictions))
    return avg_pred, predictions, labels


# ============================== XAI ==============================


def generate_xai(model, raw_frames, num_frames: int):
    if not LIME_AVAILABLE:
        return None

    normalized = [f.astype(np.float32) / 255.0 for f in raw_frames]
    padded = pad_frames(normalized, num_frames)
    if padded is None:
        return None

    target_frame = padded[num_frames // 2]
    explainer = lime_image.LimeImageExplainer()

    def predict_wrapper(images):
        responses = []
        for img in images:
            seq = np.stack([img] * num_frames, axis=0)
            res = model.predict(np.expand_dims(seq, axis=0), verbose=0)
            responses.append([1 - res[0][0], res[0][0]])
        return np.array(responses)

    explanation = explainer.explain_instance(
        target_frame.astype("double"),
        predict_wrapper,
        top_labels=1,
        hide_color=0,
        num_samples=80,
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False,
    )

    explanation_img = mark_boundaries(temp, mask)
    explanation_img = (explanation_img * 255).astype(np.uint8)
    return Image.fromarray(explanation_img)


# ============================== LOAD MODEL ==============================

print("[DeepFake App] Loading model...")
MODEL = None
DETECTED_NUM_FRAMES = NUM_FRAMES

for path_candidate in [MODEL_PATH, FALLBACK_MODEL_PATH]:
    if os.path.exists(path_candidate):
        try:
            MODEL = tf.keras.models.load_model(path_candidate, compile=False)
            print(f"[DeepFake App] Loaded model from: {path_candidate}")
            try:
                input_shape = MODEL.input_shape
                if input_shape and len(input_shape) >= 2 and input_shape[1] is not None:
                    DETECTED_NUM_FRAMES = int(input_shape[1])
                    print(f"[DeepFake App] Auto-detected model expects {DETECTED_NUM_FRAMES} frames.")
                else:
                    print(f"[DeepFake App] Could not auto-detect frames from input_shape={input_shape}")
            except Exception as e:
                print(f"[DeepFake App] Could not inspect model input shape: {e}")
            break
        except Exception as exc:
            print(f"[DeepFake App] ERROR loading {path_candidate}: {exc}")

if MODEL is None:
    print("[DeepFake App] WARNING: No model loaded. Please upload a .keras model file.")


# ============================== UI UPDATE HANDLER ==============================


def update_ui(lang_display: str, theme_display: str):
    lang = LANG_MAP.get(lang_display, "en")
    theme = THEME_MAP.get(theme_display, "light")
    t = TEXTS[lang]
    return [
        gr.update(value=t["title"]),           # 0 title_md
        gr.update(value=t["desc"]),            # 1 desc_md
        gr.update(label=t["upload_label"]),    # 2 input_file
        gr.update(label=t["xai_label"], info=t["xai_info"]),  # 3 enable_xai
        gr.update(value=t["analyze_btn"]),     # 4 analyze_btn
        gr.update(label=t["preview_label"]),   # 5 preview_img
        gr.update(label=t["xai_label_img"]),   # 6 xai_img
        gr.update(value=t["tips"]),            # 7 tips_md
        gr.update(value=THEME_CSS[theme]),     # 8 theme_html
    ]


# ============================== GRADIO HANDLER ==============================


def predict_deepfake(input_file, enable_xai, lang_display):
    lang = LANG_MAP.get(lang_display, "en")
    t = TEXTS[lang]

    if MODEL is None:
        return t["no_model"], None, None
    if input_file is None:
        return t["no_file"], None, None

    input_path = str(input_file) if not isinstance(input_file, str) else input_file
    print(f"[Predict] Received file: {input_path}")

    try:
        ext = os.path.splitext(input_path)[1].lower()
        is_video = ext in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv")
        is_image = ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        if not is_video and not is_image:
            return t["unsupported"], None, None

        with tempfile.TemporaryDirectory() as temp_dir:
            if is_video:
                print(t["processing_video"].format(DETECTED_NUM_FRAMES))
                raw_frames, count = preprocess_video(input_path, temp_dir, num_frames=DETECTED_NUM_FRAMES)
                if raw_frames is None or count == 0:
                    return t["no_faces_video"], None, None
                input_type = t["video_frames"].format(count)
            else:
                print(t["processing_image"])
                face = preprocess_image(input_path)
                if face is None:
                    return t["no_faces_image"], None, None
                raw_frames = [face]
                input_type = t["image_single"]

            print(t["running_ensemble"].format(DETECTED_NUM_FRAMES))
            avg_pred, all_preds, all_labels = ensemble_predict(
                MODEL, raw_frames, num_frames=DETECTED_NUM_FRAMES, window_size=WINDOW_SIZE, stride=STRIDE
            )

            if avg_pred is None:
                return t["pred_failed"], None, None

            verdict = "FAKE" if avg_pred > DECISION_THRESHOLD else "REAL"
            confidence = avg_pred if verdict == "FAKE" else 1 - avg_pred
            std_dev = float(np.std(all_preds))

            verdict_key = "verdict_fake" if verdict == "FAKE" else "verdict_real"

            result_md = (
                f"## {t[verdict_key]}\n\n"
                f"| {t['confidence']} | {confidence * 100:.2f}% |\n"
                f"| {t['avg_score']} | {avg_pred:.4f} *(1 = Fake, 0 = Real)* |\n"
                f"| {t['std_dev']} | {std_dev:.4f} |\n"
                f"| {t['windows']} | {len(all_preds)} |\n"
                f"| {t['input_type']} | {input_type} |"
            )

            normalized = [f.astype(np.float32) / 255.0 for f in raw_frames]
            padded = pad_frames(normalized, DETECTED_NUM_FRAMES)
            if padded is not None:
                preview_frame = padded[DETECTED_NUM_FRAMES // 2]
            else:
                preview_frame = normalized[len(normalized) // 2]
            preview_pil = Image.fromarray((np.clip(preview_frame, 0, 1) * 255).astype(np.uint8))

            xai_pil = None
            if enable_xai:
                if not LIME_AVAILABLE:
                    result_md += f"\n\n> {t['xai_not_installed']}"
                else:
                    try:
                        print(t["generating_xai"])
                        xai_pil = generate_xai(MODEL, raw_frames, num_frames=DETECTED_NUM_FRAMES)
                    except Exception as exc:
                        print(f"[XAI Error] {exc}")
                        result_md += f"\n\n> {t['xai_failed'].format(exc)}"

            print(t["done"])
            return result_md, preview_pil, xai_pil

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[Predict CRASH] {exc}\n{tb}")
        error_md = (
            f"{t['analysis_crashed']}\n\n"
            f"**{t['error']}:** `{exc}`\n\n"
            f"<details>\n"
            f"<summary>{t['click_expand']}</summary>\n\n"
            f"```\n{tb}\n```\n\n"
            f"</details>\n\n"
            f"{t['common_causes']}"
        )
        return error_md, None, None


# ============================== GRADIO UI ==============================

# Use gr.themes.Base() but initialise with light colours so the page starts
# correctly before any JavaScript / CSS injection kicks in.
light_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
).set(
    body_background_fill="#FFFFFF",
    body_text_color="#202124",
    block_background_fill="#FFFFFF",
    block_border_color="#DADCE0",
    block_label_background_fill="#F1F3F4",
    block_label_text_color="#202124",
    block_title_text_color="#202124",
    input_background_fill="#FFFFFF",
    input_border_color="#DADCE0",
    checkbox_background_color="#FFFFFF",
    button_primary_background_fill="#1A73E8",
    button_primary_text_color="#FFFFFF",
    panel_background_fill="#F8F9FA",
    panel_border_color="#E8EAED",
)

with gr.Blocks(title="DeepFake Detector", theme=light_theme) as demo:
    # Invisible HTML for dynamic theme CSS injection
    theme_html = gr.HTML(value=THEME_CSS["light"])

    # Header row: Language + Theme
    with gr.Row(elem_classes="app-header"):
        lang_dropdown = gr.Dropdown(
            choices=["English", "اردو"],
            value="English",
            label="Language / زبان",
            scale=1,
        )
        theme_radio = gr.Radio(
            choices=["Light", "Dark"],
            value="Light",
            label="Theme / تھیم",
            scale=1,
        )

    title_md = gr.Markdown(TEXTS["en"]["title"])
    desc_md = gr.Markdown(TEXTS["en"]["desc"])

    with gr.Row():
        with gr.Column(scale=1, elem_classes="upload-section"):
            input_file = gr.File(
                label=TEXTS["en"]["upload_label"],
                file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv",
                            ".jpg", ".jpeg", ".png", ".bmp", ".webp"],
            )
            enable_xai = gr.Checkbox(
                label=TEXTS["en"]["xai_label"],
                value=False,
                info=TEXTS["en"]["xai_info"],
                elem_classes="xai-option",
            )
            analyze_btn = gr.Button(TEXTS["en"]["analyze_btn"], variant="primary")

        with gr.Column(scale=1, elem_classes="result-section"):
            result_md = gr.Markdown()
            with gr.Row():
                preview_img = gr.Image(label=TEXTS["en"]["preview_label"], type="pil")
                xai_img = gr.Image(label=TEXTS["en"]["xai_label_img"], type="pil")

    tips_md = gr.Markdown(TEXTS["en"]["tips"])

    # Language / Theme change triggers UI update
    lang_dropdown.change(
        fn=update_ui,
        inputs=[lang_dropdown, theme_radio],
        outputs=[title_md, desc_md, input_file, enable_xai, analyze_btn, preview_img, xai_img, tips_md, theme_html],
    )
    theme_radio.change(
        fn=update_ui,
        inputs=[lang_dropdown, theme_radio],
        outputs=[title_md, desc_md, input_file, enable_xai, analyze_btn, preview_img, xai_img, tips_md, theme_html],
    )

    # Analyze button
    analyze_btn.click(
        fn=predict_deepfake,
        inputs=[input_file, enable_xai, lang_dropdown],
        outputs=[result_md, preview_img, xai_img],
    )

if __name__ == "__main__":
    demo.launch()