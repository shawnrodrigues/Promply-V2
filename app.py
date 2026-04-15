from flask import Flask, request, render_template, jsonify
import os
import re
import time
import traceback
import sys
from dotenv import load_dotenv
import chromadb
import torch

# Set transformers offline mode, but allow override via environment variables.
os.environ['TRANSFORMERS_OFFLINE'] = os.getenv('TRANSFORMERS_OFFLINE', '1')
os.environ['HF_HUB_OFFLINE'] = os.getenv('HF_HUB_OFFLINE', '1')

from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text
from llama_cpp import Llama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import fitz  # PyMuPDF
try:
    import pytesseract
except Exception:
    pytesseract = None
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import io
try:
    import google.generativeai as genai  # online search (optional for offline mode)
except ImportError:
    genai = None
import platform 
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from functools import lru_cache

# Always flush console logs so upload/chat progress appears in real time.
import builtins as _builtins
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return _builtins.print(*args, **kwargs)

# Also ensure stderr is unbuffered for immediate visibility
sys.stderr = __import__('io').TextIOWrapper(
    __import__('sys').stderr.buffer,
    encoding=sys.stderr.encoding,
    line_buffering=True
)

# Helper function for critical logging - writes to both stdout and stderr for guaranteed visibility
def log_step(message):
    """Log critical steps to console only"""
    timestamp = time.strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    _builtins.print(formatted_msg, flush=True)

# Configure Tesseract for Windows when available.
if pytesseract is not None and platform.system() == 'Windows':
    # Common Tesseract installation paths on Windows
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe'
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"✓ Tesseract found at: {path}")
            break
    else:
        print("⚠ WARNING: Tesseract not found at standard locations.")

from openai import OpenAI
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

load_dotenv()

UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'uploads/images'
VECTORDB_FOLDER = 'vector_store'
ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif'}

MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(VECTORDB_FOLDER, exist_ok=True)

app = Flask(__name__)
print("🔧 Flask app created successfully")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Add HTTP request logging to show all API calls
# @app.before_request
# def log_request():
#     """Log incoming HTTP requests"""
#     # Just log to console quickly, avoid file I/O per request
#     if request.path in ['/chat', '/upload']:
#         print(f"📡 {request.method} {request.path}")

# @app.after_request
# def log_response(response):
#     """Log outgoing HTTP responses"""
#     if request.path in ['/chat', '/upload']:
#         print(f"📡 {response.status_code}")
#     return response

# Always use GPU for embedding checking if the gpu is working good or no hehehe
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available - GPU required for this app.")

device = "cuda"
print(f"SentenceTransformer running on: {device}")

#loading the embedding model

try:
    embed_model = SentenceTransformer("all-mpnet-base-v2", device=device)
    print("✓ Loaded embedding model successfully")
except Exception as e:
    print(f"ERROR: Could not load embedding model: {e}")
    print("SOLUTION: The model needs to be downloaded once with internet connection.")
    print("After the first download, the model will be cached locally and work offline.")
    raise RuntimeError(f"Failed to load embedding model. Please connect to internet for first-time setup. Error: {e}")

OFFLINE_ONLY = True
SEARCH_ENGINE = "gemini"  # Changed from duckduckgo to gemini because i got timed out
ENABLE_HANDWRITING_OCR = True
STRICT_GPU_MODE = os.getenv("STRICT_GPU_MODE", "true").lower() != "false"
ENABLE_TESSERACT_OCR = os.getenv("ENABLE_TESSERACT_OCR", "true").lower() != "false"
SOURCE_LOCK_ENABLED = False
SOURCE_LOCK_FILE = os.getenv("SOURCE_LOCK_FILE", "").strip()
LAST_UPLOADED_FILE = ""
SESSION_UPLOADED_FILES = set()
HANDWRITING_SOURCE_FILES = {}
FAST_RESULT_LIMIT = 10
FULL_RESULT_LIMIT = 20
QUERY_EMBED_CACHE_SIZE = 64
RELEVANCE_DISTANCE_THRESHOLD = float(os.getenv("RELEVANCE_DISTANCE_THRESHOLD", "1.35"))
EARLY_EXPANSION_DISTANCE_THRESHOLD = float(os.getenv("EARLY_EXPANSION_DISTANCE_THRESHOLD", "0.85"))
RELAXED_RELEVANCE_DISTANCE_THRESHOLD = float(os.getenv("RELAXED_RELEVANCE_DISTANCE_THRESHOLD", "1.65"))
SECOND_PASS_RESULT_LIMIT = int(os.getenv("SECOND_PASS_RESULT_LIMIT", "20"))
MIN_OCR_IMAGE_DIMENSION = int(os.getenv("MIN_OCR_IMAGE_DIMENSION", "900"))
OCR_TESSERACT_CONFIG = os.getenv("TESSERACT_CONFIG", "--oem 3 --psm 6")
OCR_TESSERACT_VARIANTS = [
    ("balanced", "--oem 3 --psm 6"),
    ("balanced", "--oem 3 --psm 11"),
    ("high_contrast", "--oem 3 --psm 4"),
]
PAGE_OCR_DPI = int(os.getenv("PAGE_OCR_DPI", "220"))
MAX_SOURCE_FILES_PER_QUERY = int(os.getenv("MAX_SOURCE_FILES_PER_QUERY", "4"))
AUTO_SCOPE_LAST_UPLOAD = os.getenv("AUTO_SCOPE_LAST_UPLOAD", "true").lower() != "false"
DOCUMENT_TEXT_CACHE = {}
OCR_TEXT_MIN_SCORE = float(os.getenv("OCR_TEXT_MIN_SCORE", "4.5"))
ENABLE_LEXICAL_FALLBACK = os.getenv("ENABLE_LEXICAL_FALLBACK", "true").lower() == "true"
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "350"))
ONLINE_MAX_TOKENS = int(os.getenv("ONLINE_MAX_TOKENS", "350"))
LEXICAL_STOP_WORDS = {
    "what", "is", "are", "the", "a", "an", "how", "why", "when", "where",
    "who", "which", "tell", "me", "about", "please", "find", "show", "give",
    "my", "his", "her", "their", "does", "did", "can", "you", "explain"
}

trocr_processor = None
trocr_model = None
trocr_device = "cuda"
trocr_model_name = os.getenv("TROCR_MODEL_NAME", "microsoft/trocr-base-handwritten")
ENABLE_PRINTED_TROCR = os.getenv("ENABLE_PRINTED_TROCR", "true").lower() != "false"
trocr_printed_processor = None
trocr_printed_model = None
trocr_printed_model_name = os.getenv("TROCR_PRINTED_MODEL_NAME", "microsoft/trocr-base-printed")
trocr_local_only = (
    os.getenv("TROCR_LOCAL_ONLY", "").strip().lower() == "true"
    or os.environ.get("TRANSFORMERS_OFFLINE", "1") == "1"
    or os.environ.get("HF_HUB_OFFLINE", "1") == "1"
)

if STRICT_GPU_MODE and ENABLE_TESSERACT_OCR:
    # Tesseract runs on CPU — it complements GPU TrOCR, doesn't compete
    print("[GPU MODE] STRICT_GPU_MODE is enabled; Tesseract OCR (CPU) remains available as fallback.")

if ENABLE_TESSERACT_OCR and pytesseract is None:
    ENABLE_TESSERACT_OCR = False
    print("[OCR] pytesseract is unavailable, disabling Tesseract OCR.")

# Load TrOCR handwritten model.
if ENABLE_HANDWRITING_OCR:
    try:
        trocr_processor = TrOCRProcessor.from_pretrained(
            trocr_model_name,
            local_files_only=trocr_local_only,
        )
        trocr_model = VisionEncoderDecoderModel.from_pretrained(
            trocr_model_name,
            local_files_only=trocr_local_only,
        ).to(trocr_device)
        trocr_model.eval()
        print(f"✓ Handwriting OCR enabled with TrOCR on: {trocr_device}")
    except Exception as e:
        ENABLE_HANDWRITING_OCR = False
        print(f"⚠ Handwriting OCR unavailable: {e}")
        print("  Continuing with available OCR engines.")

if ENABLE_PRINTED_TROCR:
    try:
        trocr_printed_processor = TrOCRProcessor.from_pretrained(
            trocr_printed_model_name,
            local_files_only=trocr_local_only,
        )
        trocr_printed_model = VisionEncoderDecoderModel.from_pretrained(
            trocr_printed_model_name,
            local_files_only=trocr_local_only,
        ).to(trocr_device)
        trocr_printed_model.eval()
        print(f"✓ Printed OCR enabled with TrOCR on: {trocr_device}")
    except Exception as e:
        ENABLE_PRINTED_TROCR = False
        print(f"⚠ Printed TrOCR unavailable: {e}")

if STRICT_GPU_MODE and (
    (trocr_model is None or trocr_processor is None)
    and (trocr_printed_model is None or trocr_printed_processor is None)
):
    raise RuntimeError(
        "STRICT_GPU_MODE requires at least one GPU TrOCR model, but none loaded. "
        f"Handwritten='{trocr_model_name}', printed='{trocr_printed_model_name}', local_only={trocr_local_only}.\n\n"
        "One-time setup (with internet) to cache both models locally:\n"
        "  Windows PowerShell:\n"
        "  $env:TRANSFORMERS_OFFLINE='0'; $env:HF_HUB_OFFLINE='0'; "
        "python -c \"from transformers import TrOCRProcessor, VisionEncoderDecoderModel; "
        "for m in ['microsoft/trocr-base-handwritten','microsoft/trocr-base-printed']: "
        "TrOCRProcessor.from_pretrained(m); VisionEncoderDecoderModel.from_pretrained(m)\"\n"
        "Then restart normally in offline mode."
    )

print("=" * 60)
print("PROMPTLY STARTING...")
print("=" * 60)
print(f"Initial Mode: {'OFFLINE MODE' if OFFLINE_ONLY else 'ONLINE MODE'}")
print("=" * 60)

def embed_text(text):
    return embed_model.encode(text, show_progress_bar=False)

def embed_text_single(text):
    """Embed a single string. For batch embedding, pass a list to embed_text()."""
    return embed_model.encode([text], show_progress_bar=False)[0]

@lru_cache(maxsize=QUERY_EMBED_CACHE_SIZE)
def _cached_query_embedding(text):
    return embed_model.encode([text], show_progress_bar=False)[0]

def get_query_embedding(text):
    return _cached_query_embedding(text)

def parse_pdf_text(path):
    """Extract textual content from PDFs with graceful fallbacks."""
    primary_text = ""

    try:
        primary_text = extract_text(path) or ""
    except Exception as e:
        print(f"  ⚠ PDFMiner extraction failed: {e}")

    primary_text = normalize_extracted_text(primary_text)
    if primary_text and score_text_quality(primary_text) >= OCR_TEXT_MIN_SCORE:
        return primary_text

    if primary_text:
        print("  ⚠ PDFMiner text looked noisy; falling back to page text extraction")

    # Fallback: use PyMuPDF's text extractor when PDFMiner returns nothing
    try:
        fallback_pages = []
        with fitz.open(path) as doc:
            for page_number, page in enumerate(doc, start=1):
                page_text = normalize_extracted_text(page.get_text("text"))
                if page_text and score_text_quality(page_text) >= OCR_TEXT_MIN_SCORE:
                    fallback_pages.append(page_text)
        if fallback_pages:
            print("  🔁 PDF text extracted via PyMuPDF fallback")
            return "\n".join(fallback_pages)
    except Exception as e:
        print(f"  ❌ PyMuPDF text extraction failed: {e}")

    return ""

def normalize_extracted_text(text):
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"-\n(?=\w)", "", text)

    lines = []
    for raw_line in text.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue

        compact = line.replace(" ", "")
        alnum_ratio = sum(ch.isalnum() for ch in compact) / len(compact) if compact else 0
        if len(compact) >= 8 and alnum_ratio < 0.25 and not re.search(r"\d", compact):
            continue

        lines.append(line)

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines).strip()


def score_text_quality(text):
    if not text:
        return 0.0

    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return 0.0

    words = re.findall(r"[A-Za-z0-9']+", compact)
    length = len(compact)
    word_count = len(words)
    letters = sum(ch.isalpha() for ch in compact)
    alnum = sum(ch.isalnum() for ch in compact)
    weird = sum(
        not (ch.isalnum() or ch.isspace() or ch in ".,;:!?()[]{}-_/\\'\"@#&%+*=<>")
        for ch in compact
    )
    avg_word_len = sum(len(word) for word in words) / word_count if word_count else 0
    digit_ratio = sum(ch.isdigit() for ch in compact) / length

    score = 0.0
    score += min(length / 40.0, 4.0)
    score += min(word_count / 4.0, 5.0)
    score += (alnum / length) * 4.0
    score += (letters / length) * 3.0
    score += min(avg_word_len / 6.0, 2.0)
    score -= (weird / length) * 10.0

    if word_count <= 2 and length > 60:
        score -= 2.0
    if digit_ratio > 0.4 and letters < max(5, int(length * 0.1)):
        score -= 2.0

    return score


def select_best_candidate(candidates):
    best_candidate = ("", False, 0.0)

    for candidate in candidates:
        text = normalize_extracted_text(candidate[0])
        if not text:
            continue

        handwriting_hit = bool(candidate[1])
        score = candidate[2] if len(candidate) > 2 and candidate[2] is not None else score_text_quality(text)

        if score > best_candidate[2] or (score == best_candidate[2] and len(text) > len(best_candidate[0])):
            best_candidate = (text, handwriting_hit, score)

    return best_candidate


def merge_text_candidates(texts):
    merged_lines = []
    seen = set()

    for text in texts:
        cleaned = normalize_extracted_text(text)
        if not cleaned:
            continue
        for line in cleaned.split("\n"):
            key = line.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged_lines.append(line.strip())

    return "\n".join(merged_lines).strip()


def preprocess_image_for_ocr(image, mode="balanced"):
    """Normalize contrast and size so OCR stays consistent across uploads."""
    gray = image.convert("L")
    width, height = gray.size
    min_dim = max(1, min(width, height))

    if min_dim < MIN_OCR_IMAGE_DIMENSION:
        scale = MIN_OCR_IMAGE_DIMENSION / min_dim
        new_size = (int(width * scale), int(height * scale))
        gray = gray.resize(new_size, Image.LANCZOS)

    # Cap very large images to prevent GPU OOM
    max_dim = max(gray.size)
    if max_dim > 4000:
        scale = 4000 / max_dim
        gray = gray.resize((int(gray.size[0] * scale), int(gray.size[1] * scale)), Image.LANCZOS)

    gray = ImageOps.autocontrast(gray)
    if mode == "high_contrast":
        gray = ImageEnhance.Contrast(gray).enhance(1.8)
        gray = ImageEnhance.Sharpness(gray).enhance(1.5)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
        gray = gray.point(lambda x: 255 if x > 140 else 0)
    elif mode == "handwriting":
        # Gentler preprocessing that preserves faint pen/pencil strokes
        gray = ImageEnhance.Contrast(gray).enhance(1.6)
        gray = ImageEnhance.Sharpness(gray).enhance(1.3)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
    else:
        gray = ImageEnhance.Contrast(gray).enhance(1.4)
        gray = ImageEnhance.Sharpness(gray).enhance(1.5)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))

    return gray


def segment_text_lines(image):
    """Split an image into individual text line crops using horizontal projection.

    TrOCR is a single-line OCR model — feeding it a full page produces garbage.
    This function detects line boundaries by looking at row-wise ink density
    and returns a list of cropped PIL images, one per text line.
    """
    import numpy as np

    gray = image.convert("L")
    arr = np.array(gray)

    # Binarize: ink pixels = 1 (dark on white background)
    threshold = 200
    binary = (arr < threshold).astype(np.uint8)

    # Horizontal projection — count ink pixels per row
    row_sums = binary.sum(axis=1)

    # Find row ranges that contain text (non-zero projection)
    min_ink = max(2, int(arr.shape[1] * 0.005))  # at least 0.5% of row width
    in_line = False
    lines = []
    start = 0

    for i, val in enumerate(row_sums):
        if val >= min_ink and not in_line:
            start = i
            in_line = True
        elif val < min_ink and in_line:
            if i - start > 20:  # reject thin ruled-paper lines (need >20px for real text)
                lines.append((start, i))
            in_line = False

    if in_line and len(arr) - start > 20:
        lines.append((start, len(arr)))

    if not lines:
        # No lines detected — return the whole image as one line
        return [image]

    # Add vertical padding around each line
    pad = 6
    width = image.size[0]
    crops = []
    for top, bottom in lines:
        top = max(0, top - pad)
        bottom = min(arr.shape[0], bottom + pad)
        crop = image.crop((0, top, width, bottom))
        crops.append(crop)

    return crops

TROCR_BATCH_SIZE = int(os.getenv("TROCR_BATCH_SIZE", "8"))


def _batch_trocr(line_crops, processor, model, label="TrOCR"):
    """Run TrOCR on a list of line crops using batched GPU inference.

    Instead of processing one line at a time, groups lines into batches
    of TROCR_BATCH_SIZE and runs them through the GPU in parallel.
    Uses greedy decoding (no beam search) for maximum speed.
    """
    all_lines = []
    total = len(line_crops)

    for batch_start in range(0, total, TROCR_BATCH_SIZE):
        batch_crops = line_crops[batch_start:batch_start + TROCR_BATCH_SIZE]
        rgb_batch = [crop.convert("RGB") for crop in batch_crops]

        inputs = processor(images=rgb_batch, return_tensors="pt", padding=True)
        pixel_values = inputs.pixel_values.to(trocr_device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=128,
            )

        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for line_text in decoded:
            text = line_text.strip()
            if text:
                all_lines.append(text)

        del pixel_values, generated_ids

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_lines


def ocr_with_trocr(image):
    """Run handwritten OCR on a PIL image with TrOCR (batched)."""
    if not ENABLE_HANDWRITING_OCR or trocr_processor is None or trocr_model is None:
        return ""

    try:
        preprocessed = preprocess_image_for_ocr(image, mode="handwriting")
        line_crops = segment_text_lines(preprocessed)
        if len(line_crops) > 50:
            line_crops = line_crops[:50]
        print(f"    [TrOCR-HW] Batch processing {len(line_crops)} line(s) in groups of {TROCR_BATCH_SIZE}")

        all_lines = _batch_trocr(line_crops, trocr_processor, trocr_model, "HW")
        text = "\n".join(all_lines)
        return normalize_extracted_text(text)
    except Exception as e:
        print(f"  ⚠ TrOCR handwriting failed: {e}")
        return ""

def ocr_with_trocr_printed(image):
    """Run printed-text OCR on a PIL image with TrOCR (batched)."""
    if not ENABLE_PRINTED_TROCR or trocr_printed_processor is None or trocr_printed_model is None:
        return ""

    try:
        preprocessed = preprocess_image_for_ocr(image, mode="balanced")
        line_crops = segment_text_lines(preprocessed)
        if len(line_crops) > 50:
            line_crops = line_crops[:50]
        print(f"    [TrOCR-Print] Batch processing {len(line_crops)} line(s) in groups of {TROCR_BATCH_SIZE}")

        all_lines = _batch_trocr(line_crops, trocr_printed_processor, trocr_printed_model, "Print")
        text = "\n".join(all_lines)
        return normalize_extracted_text(text)
    except Exception as e:
        print(f"  ⚠ Printed TrOCR failed: {e}")
        return ""

def run_ocr_pipeline_on_image(image):
    """Run OCR on a single image.

    Strategy (saves GPU memory):
      1. Try Tesseract variants (CPU, if enabled)
      2. Try TrOCR printed model first — most documents are printed
      3. Only if printed result is poor, try TrOCR handwritten model
    This avoids loading both models' tensors simultaneously.
    """
    candidates = []

    # Phase 1: Tesseract (CPU) — always available as fallback regardless of GPU mode
    if ENABLE_TESSERACT_OCR and pytesseract is not None:
        for mode, config in OCR_TESSERACT_VARIANTS:
            try:
                processed_image = preprocess_image_for_ocr(image, mode=mode)
                text = pytesseract.image_to_string(processed_image, config=config)
                cleaned = normalize_extracted_text(text)
                if cleaned:
                    candidates.append((cleaned, False, score_text_quality(cleaned)))
            except Exception as e:
                print(f"  ⚠ Tesseract OCR failed for mode={mode}, config={config}: {e}")

        try:
            raw_text = pytesseract.image_to_string(image, config=OCR_TESSERACT_CONFIG)
            cleaned_raw = normalize_extracted_text(raw_text)
            if cleaned_raw:
                candidates.append((cleaned_raw, False, score_text_quality(cleaned_raw) - 0.5))
        except Exception as e:
            print(f"  ⚠ Raw Tesseract OCR failed: {e}")

    # Phase 2: TrOCR printed (GPU) — try this first
    printed_score = 0.0
    if ENABLE_PRINTED_TROCR:
        printed_trocr_text = ocr_with_trocr_printed(image)
        if printed_trocr_text:
            printed_score = score_text_quality(printed_trocr_text)
            candidates.append((printed_trocr_text, False, printed_score))

    # Phase 3: TrOCR handwritten (GPU) — only if printed result was poor
    if ENABLE_HANDWRITING_OCR and printed_score < 6.0:
        trocr_text = ocr_with_trocr(image)
        if trocr_text:
            candidates.append((trocr_text, True, score_text_quality(trocr_text)))

    if not candidates:
        return "", False, 0.0

    best_text, best_handwriting, best_score = select_best_candidate(candidates)
    return best_text, best_handwriting, best_score

def extract_images_and_ocr(pdf_path):
    """Extract images from PDF and perform OCR with proper error handling."""
    doc = fitz.open(pdf_path)
    image_texts = []
    total_images = 0
    successful_ocr = 0
    handwriting_successful_ocr = 0
    fallback_pages = 0
    
    try:
        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            images = page.get_images(full=True)
            total_images += len(images)
            page_candidates = []
            
            for img in images:
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))

                    text, handwriting_hit, score = run_ocr_pipeline_on_image(image)
                    if text.strip():
                        page_candidates.append((text, handwriting_hit, score))
                except Exception as img_error:
                    print(f"  ⚠ OCR failed for image on page {page_number + 1}: {str(img_error)}")
                    continue

            best_page_text, best_page_handwriting, best_page_score = select_best_candidate(page_candidates)

            if best_page_text and best_page_score >= OCR_TEXT_MIN_SCORE:
                image_texts.append(best_page_text)
                successful_ocr += 1
                if best_page_handwriting:
                    handwriting_successful_ocr += 1
            else:
                try:
                    pix = page.get_pixmap(dpi=PAGE_OCR_DPI, alpha=False)
                    raster_bytes = pix.tobytes("png")
                    with Image.open(io.BytesIO(raster_bytes)) as raster_image:
                        text, handwriting_hit, score = run_ocr_pipeline_on_image(raster_image)

                    fallback_pages += 1
                    raster_candidates = page_candidates[:]
                    if text.strip():
                        raster_candidates.append((text, handwriting_hit, score))

                    best_text, best_handwriting, best_score = select_best_candidate(raster_candidates)
                    if best_text and best_score >= OCR_TEXT_MIN_SCORE:
                        image_texts.append(best_text)
                        successful_ocr += 1
                        if best_handwriting:
                            handwriting_successful_ocr += 1
                        print(f"  🔁 Page {page_number + 1}: fallback raster OCR captured text")
                except Exception as fallback_error:
                    print(f"  ⚠ Page {page_number + 1} fallback OCR failed: {fallback_error}")
                    continue

            # Free GPU memory after each page
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(
            f"  📊 OCR Stats: {total_images} images found, {successful_ocr} successfully processed"
            f" (handwriting hits: {handwriting_successful_ocr}, rasterized pages: {fallback_pages})"
        )
        if total_images > 0 and successful_ocr == 0:
            print("  ⚠ WARNING: No text extracted from images. Check Tesseract installation.")
        
        return {
            "text": "\n".join(image_texts),
            "images_found": total_images,
            "images_processed": successful_ocr,
            "handwriting_images_processed": handwriting_successful_ocr,
            "pages_rasterized": fallback_pages,
        }
    except Exception as e:
        print(f"  ❌ OCR Error: {str(e)}")
        return {
            "text": "",
            "images_found": 0,
            "images_processed": 0,
            "handwriting_images_processed": 0,
            "pages_rasterized": fallback_pages,
            "error": str(e),
        }
    finally:
        doc.close()

def process_image_file(image_path):
    try:
        image = Image.open(image_path)
        text, handwriting_hit, _ = run_ocr_pipeline_on_image(image)
        processed = 1 if text.strip() else 0
        return {
            "text": text,
            "images_found": 1,
            "images_processed": processed,
            "handwriting_images_processed": 1 if handwriting_hit else 0,
        }
    except Exception as e:
        print(f"  ❌ Image OCR Error: {str(e)}")
        return {
            "text": "",
            "images_found": 1,
            "images_processed": 0,
            "handwriting_images_processed": 0,
            "error": str(e),
        }

chroma_client = chromadb.PersistentClient(path=VECTORDB_FOLDER)

def get_or_recreate_collection(client, name, expected_dim):
    try:
        col = client.get_collection(name=name)
        dummy_vec = embed_text("test")
        if len(dummy_vec) != expected_dim:
            client.delete_collection(name)
            col = client.create_collection(name=name)
        return col
    except Exception:
        return client.create_collection(name=name)

expected_dim = embed_model.get_sentence_embedding_dimension()
collection = get_or_recreate_collection(chroma_client, "manuals", expected_dim)

print("Initializing LLaMA with GPU...")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=8192,
    n_batch=256
)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global LAST_UPLOADED_FILE, SOURCE_LOCK_FILE, HANDWRITING_SOURCE_FILES, SESSION_UPLOADED_FILES
    try:
        # File uploads use multipart/form-data, not application/json
        # Explicitly access files without triggering JSON parsing
        if 'pdf' not in request.files and 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400
        
        file = request.files.get("pdf") or request.files.get("file")

        filename = file.filename or ""
        if not filename:
            return jsonify({"status": "error", "message": "Uploaded file is missing a valid name"})

        ext = os.path.splitext(filename)[1].lower()
        is_pdf = ext == ".pdf"
        is_image = ext in ALLOWED_IMAGE_EXTENSIONS

        if not is_pdf and not is_image:
            return jsonify({
                "status": "error",
                "message": "Unsupported file type. Upload a PDF or one of: " + ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
            })

        save_dir = UPLOAD_FOLDER if is_pdf else IMAGE_FOLDER
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)

        print(f"Processing upload: {filename} ({'PDF' if is_pdf else 'Image'})")
        try:
            file.save(path)
        except PermissionError:
            # Windows can lock a file temporarily (previewers, scanners, etc.).
            # Save this upload with a unique on-disk name while keeping logical source metadata unchanged.
            base, ext = os.path.splitext(filename)
            fallback_name = f"{base}__{int(time.time())}_{os.getpid()}{ext}"
            fallback_path = os.path.join(save_dir, fallback_name)
            print(f"  ⚠ File lock detected for {filename}; retrying as {fallback_name}")
            file.save(fallback_path)
            path = fallback_path

        combined_text = ""
        pdf_text = ""
        text_length = 0
        ocr_length = 0
        ocr_result = {"text": "", "images_found": 0, "images_processed": 0, "handwriting_images_processed": 0}

        if is_pdf:
            print("  📄 Extracting text from PDF...")
            pdf_text = parse_pdf_text(path) or ""
            text_length = len(pdf_text.strip())
            print(f"  ✓ Extracted {text_length} characters of text")

            # Only run expensive OCR if native text extraction was poor.
            # PDFs with rich native text (e.g. digital docs) don't need OCR.
            if text_length < 200:
                print("  🖼️ Native text insufficient — running OCR on images...")
                ocr_result = extract_images_and_ocr(path)
                image_text = ocr_result.get("text", "")
                ocr_length = len(image_text.strip())
                print(f"  ✓ Extracted {ocr_length} characters from OCR")

                if ocr_length > 0:
                    preview = image_text[:200].replace('\n', ' ').strip()
                    print(f"  📋 OCR Preview: {preview}...")

                if pdf_text and image_text:
                    combined_text = pdf_text + "\n" + image_text
                else:
                    combined_text = pdf_text or image_text
            else:
                print(f"  ✓ Native text sufficient ({text_length} chars) — skipping OCR")
                combined_text = pdf_text
        else:
            print("  🖼️ Running OCR on uploaded image...")
            ocr_result = process_image_file(path)
            combined_text = ocr_result.get("text", "")
            text_length = len(combined_text.strip())
            ocr_length = text_length
            if text_length > 0:
                preview = combined_text[:200].replace('\n', ' ').strip()
                print(f"  📋 OCR Preview: {preview}...")
            else:
                print("  ⚠ No text extracted from the uploaded image.")

        if not combined_text.strip():
            message = (
                "No text could be extracted from the uploaded file. "
                "This usually means the document is image-only and OCR could not detect readable text."
            )
            message += (
                f" OCR diagnostics: images_found={ocr_result.get('images_found', 0)}, "
                f"images_processed={ocr_result.get('images_processed', 0)}, "
                f"handwriting_images_processed={ocr_result.get('handwriting_images_processed', 0)}, "
                f"gpu_trOCR_handwritten={ENABLE_HANDWRITING_OCR}, gpu_trOCR_printed={ENABLE_PRINTED_TROCR}."
            )
            if ocr_result.get("error"):
                message += f" Details: {ocr_result['error']}"
            return jsonify({"status": "error", "message": message})

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )
        chunks = text_splitter.split_text(combined_text)
        DOCUMENT_TEXT_CACHE[filename] = combined_text

        file_has_handwriting = ocr_result.get("handwriting_images_processed", 0) > 0
        HANDWRITING_SOURCE_FILES[filename] = file_has_handwriting

        doc_label = "PDF" if is_pdf else "Image"

        try:
            collection.delete(where={"source_file": filename})
            print(f"🧹 Cleared previous chunks for {filename}")
        except Exception:
            # Safe to ignore when file has not been seen before
            pass

        print(f"{doc_label} split into {len(chunks)} chunk(s). Embedding & adding to vector DB...")
        # Batch-embed all chunks at once (much faster than one-by-one)
        all_embeddings = embed_text(chunks)
        for idx, (chunk, emb) in tqdm(enumerate(zip(chunks, all_embeddings)), total=len(chunks), desc="Processing chunks"):
            collection.upsert(
                documents=[chunk],
                embeddings=[emb.tolist()],
                ids=[f"{filename}_{idx}"],
                metadatas=[{
                    "source_file": filename,
                    "chunk_index": idx,
                    "handwriting_source": file_has_handwriting
                }]
            )

        LAST_UPLOADED_FILE = filename
        SESSION_UPLOADED_FILES.add(filename)
        if SOURCE_LOCK_ENABLED and not SOURCE_LOCK_FILE:
            SOURCE_LOCK_FILE = filename

        print(f"Upload complete: {filename}")

        response_data = {
            "status": "success",
            "message": f"{doc_label} uploaded and processed",
            "mode": "offline" if OFFLINE_ONLY else "online",
            "file_type": doc_label.lower(),
            "stats": {
                "text_characters": text_length,
                "ocr_characters": ocr_length,
                "images_found": ocr_result.get("images_found", 0),
                "images_processed": ocr_result.get("images_processed", 0),
                "handwriting_images_processed": ocr_result.get("handwriting_images_processed", 0),
                "pages_rasterized": ocr_result.get("pages_rasterized", 0),
                "total_chunks": len(chunks)
            },
            "source_lock": {
                "enabled": SOURCE_LOCK_ENABLED,
                "active_file": get_effective_source_file()
            },
            "handwriting": {
                "file_contains_handwriting_ocr": file_has_handwriting
            }
        }

        if ocr_result.get("error"):
            response_data["warning"] = f"OCR Error: {ocr_result['error']}"
        elif ocr_result.get("images_found", 0) > 0 and ocr_result.get("images_processed", 0) == 0:
            response_data["warning"] = "Images found but OCR failed. Check Tesseract installation."

        return jsonify(response_data)
    except Exception as e:
        print("❌ Unhandled upload error:", e)
        print(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Upload processing failed for this file: {e}"
        }), 500

def format_response(raw_response):
    """Enhanced formatting for better readability"""
    response = raw_response.strip()
    
    # Remove excessive whitespace while preserving intentional line breaks
    lines = response.split('\n')
    formatted = []
    
    for line in lines:
        line = line.strip()
        if line:
            formatted.append(line)
        elif formatted and formatted[-1] != '':  # Preserve paragraph breaks
            formatted.append('')
    
    # Remove trailing empty lines
    while formatted and formatted[-1] == '':
        formatted.pop()
    
    return '\n'.join(formatted)

def generate_answer(context, question):
    """Generate answer from context using LLM with timeout protection"""
    # Truncate context to fit within the model's context window.
    # Mistral-7B with n_ctx=8192: reserve ~1200 tokens for prompt/answer,
    # rough estimate 1 token ~ 4 chars, so cap context at ~24000 chars.
    max_context_chars = 24000
    if len(context) > max_context_chars:
        context = context[:max_context_chars]
        print(f"[LLM] Context truncated to {max_context_chars} chars to fit context window")

    prompt = f"""You are a document assistant. Answer the question using ONLY the document content below.

Rules:
- ONLY use document sections that are directly relevant to the specific topic in the question.
- IGNORE any document sections about unrelated topics, languages, or subjects.
- Start with a clear, direct answer to the question.
- Expand with relevant details, examples, or explanations found in the document.
- Organize longer answers with numbered points or short paragraphs.
- Quote key terms or definitions exactly as they appear in the document.
- If the answer is not in the document, respond only: This information was not found in the uploaded documents.
- Do not add outside knowledge or make assumptions beyond what the document states.

Document Content:
{context}

Question: {question}

Answer:"""

    try:
        print("[LLM] Generating answer from context...")
        llm_start = time.perf_counter()
        output = llm(
            prompt,
            max_tokens=ANSWER_MAX_TOKENS,
            temperature=0.3,
            top_p=0.9,
            stop=["Question:", "Document Content:", "\n\nNote:"],
            echo=False
        )
        llm_time = time.perf_counter() - llm_start
        
        answer = output["choices"][0]["text"].strip()
        tokens_generated = output.get("usage", {}).get("completion_tokens", 0) or len(answer.split())
        print(f"[LLM] ✓ Generated {tokens_generated} tokens in {llm_time:.1f}s ({tokens_generated/llm_time:.0f} tokens/sec)")
        
        if not answer:
            return "The model could not generate a response. Try rephrasing your question."
        return answer
        
    except Exception as e:
        print(f"[LLM] Generation error: {e}")
        # Fallback: return relevant context snippets if LLM fails
        chunks = context.split('\n\n---\n\n')
        summary = f"(LLM unavailable - showing {len(chunks)} relevant section(s))\n\n"
        for i, chunk in enumerate(chunks[:3], 1):
            preview = chunk[:300].strip()
            if preview:
                summary += f"{i}. {preview}...\n\n"
        return summary.strip()

def get_effective_source_file():
    """Returns the active single-source file when source lock is enabled."""
    if not SOURCE_LOCK_ENABLED:
        return ""
    if SOURCE_LOCK_FILE:
        return SOURCE_LOCK_FILE
    if LAST_UPLOADED_FILE:
        return LAST_UPLOADED_FILE

    # Backward-compatible fallback for older chunks without metadata.
    try:
        all_ids = collection.get(ids=None).get("ids", [])
        if not all_ids:
            return ""

        files = []
        seen = set()
        for chunk_id in all_ids:
            filename = "_".join(chunk_id.split("_")[:-1])
            if filename and filename not in seen:
                files.append(filename)
                seen.add(filename)

        for filename in files:
            if "intro to go" in filename.lower():
                return filename

        return files[-1] if files else ""
    except Exception:
        return ""

def lexical_fallback_search(query, active_source_file=None):
    """Simple keyword-based search used when semantic retrieval misses short facts."""
    if not ENABLE_LEXICAL_FALLBACK:
        return None

    if not DOCUMENT_TEXT_CACHE:
        return None

    query_lower = query.lower()
    tokens = [
        token for token in re.findall(r"[a-z0-9]+", query_lower)
        if len(token) > 2 and token not in LEXICAL_STOP_WORDS
    ]
    if not tokens:
        stripped = query_lower.strip()
        if not stripped:
            return None
        tokens = [stripped]

    candidates = DOCUMENT_TEXT_CACHE.items()
    if active_source_file:
        text = DOCUMENT_TEXT_CACHE.get(active_source_file)
        candidates = [(active_source_file, text)] if text else []

    best_match = None
    best_score = 0

    for filename, text in candidates:
        if not text:
            continue
        lower_text = text.lower()
        match_index = -1
        score = 0

        for token in tokens:
            idx = lower_text.find(token)
            if idx != -1:
                score += 1
                if match_index == -1 or idx < match_index:
                    match_index = idx

        if score > 0 and match_index != -1:
            snippet_start = max(0, match_index - 400)
            snippet_end = min(len(text), match_index + 400)
            snippet = text[snippet_start:snippet_end].strip()
            if score > best_score:
                best_match = (filename, snippet)
                best_score = score

    return best_match

def search_online(query):
    if OFFLINE_ONLY:
        print("🔒 ONLINE SEARCH BLOCKED: OFFLINE mode is enabled")
        return "Online search is disabled while OFFLINE mode is enabled."
    
    if genai is None:
        print("⚠️ Google Generative AI not available")
        return "Google Generative AI module not installed. Please install it with: pip install google-generativeai"

    print("\n" + "="*60)
    print(f"🌐 ONLINE SEARCH INITIATED")
    print(f"Search Engine: {SEARCH_ENGINE.upper()}")
    print(f"Query: {query}")
    print("="*60)
    
    # Quick internet connectivity check
    try:
        import urllib.request
        urllib.request.urlopen('https://www.google.com', timeout=5)
        print("✓ Internet connection verified")
    except Exception as e:
        print(f"✗ Internet connectivity issue: {e}")
        return "Unable to connect to the internet. Please check your network connection and try again."

    if SEARCH_ENGINE == "duckduckgo":
        if DDGS is None:
            return "DuckDuckGo search module not installed. Please install it with: pip install duckduckgo-search"
        try:
            print("🔍 Starting DuckDuckGo Search (free, no API key required)...")
            print("⏳ Fetching search results from DuckDuckGo...")
            
            results = []
            
            try:
                # Simplified single attempt approach
                print("   Attempting search...")
                with DDGS() as ddgs:
                    results = [r for r in ddgs.text(query, region='wt-wt', safesearch='moderate', timelimit='y', max_results=10)]
                
                if results:
                    print(f"   ✓ Got {len(results)} results")
                else:
                    print("   ✗ No results returned")
                    
            except Exception as e:
                error_details = str(e)
                print(f"   ✗ Search failed: {error_details[:200]}")
                
                # Check if it's a rate limit or blocking issue
                if "ratelimit" in error_details.lower() or "202" in error_details or "blocked" in error_details.lower():
                    return ("DuckDuckGo is currently rate-limiting or blocking requests.\n\n"
                           "This is a known issue with DuckDuckGo's free search.\n\n"
                           "Please try one of these alternatives:\n"
                           "• Wait 1-2 minutes and try again\n"
                           "• Switch to Gemini search engine (requires free API key from https://makersuite.google.com/app/apikey)\n"
                           "• Switch to OpenAI search engine (requires API key)\n\n"
                           "To switch search engines, use the settings in your interface.")
                else:
                    raise
            
            print(f"✅ Retrieved {len(results)} search results")

            if not results:
                print("⚠️ No search results found for this query after all attempts")
                print("="*60 + "\n")
                return ("No search results found from DuckDuckGo.\n\n"
                       "DuckDuckGo may be rate-limiting requests or temporarily unavailable.\n\n"
                       "Recommendations:\n"
                       "• Wait 1-2 minutes before trying again\n"
                       "• Try a different search query\n"
                       "• Switch to Gemini (free, get API key at: https://makersuite.google.com/app/apikey)\n"
                       "• Switch to OpenAI (paid, requires API key)\n\n"
                       "Note: DuckDuckGo often blocks automated searches. Gemini is recommended for better reliability.")

            print("📝 Processing search results...")
            
            # Filter and validate results for relevance
            query_lower = query.lower()
            # Remove common words that don't help with relevance
            stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who', 'which'}
            query_words = set(word for word in query_lower.split() if word not in stop_words and len(word) > 1)
            
            filtered_results = []
            for i, r in enumerate(results, 1):
                title = r.get('title', '').lower()
                body = r.get('body', '').lower()
                combined = title + " " + body
                
                # Check if any significant query word appears in the result
                has_match = False
                for word in query_words:
                    # Use 'in' to handle partial matches (e.g., "c++" in "C++ programming")
                    if word in combined:
                        has_match = True
                        break
                
                if has_match:
                    print(f"   [{i}] ✓ RELEVANT: {r.get('title', 'No title')[:60]}...")
                    filtered_results.append(r)
                else:
                    print(f"   [{i}] ✗ FILTERED: {r.get('title', 'No title')[:60]}... (not relevant)")
            
            # If filtering removed everything, just use all results
            if not filtered_results:
                print("⚠️ Relevance filter too strict, using all results")
                filtered_results = results
            else:
                print(f"📊 Kept {len(filtered_results)} relevant results out of {len(results)}")
            
            formatted = []
            for i, r in enumerate(filtered_results, 1):
                formatted.append(
                    f"Source {i}: {r.get('title', 'No title')}\n{r.get('body', '')}\nURL: {r.get('href', '')}"
                )

            context = "\n\n---\n\n".join(formatted)
            print("\n🤖 Generating answer using local LLaMA model...")

            prompt = f"""You are a helpful AI assistant. Use the following online search results to answer the question.

CRITICAL INSTRUCTIONS:
- ONLY use information from the provided search results
- DO NOT include information about unrelated topics
- If the search results mention irrelevant topics, IGNORE them completely
- Focus exclusively on answering the specific question asked
- Use simple dashes (-) for bullet points
- Use numbers (1. 2. 3.) for sequential steps
- Start with a direct, comprehensive answer
- Include relevant details from the search results
- Cite sources when mentioning specific information (e.g., "According to Source 1...")
- Do NOT use special symbols like • or ** or other formatting marks

Search Results:
{context}

Question: {query}

Answer (provide ONLY information relevant to the question):"""

            try:
                output = llm(prompt, max_tokens=ONLINE_MAX_TOKENS, temperature=0.5, stop=["Question:", "Search Results:"])
                answer = format_response(output['choices'][0]['text'])
                
                # Clear indication that this is NOT from uploaded documents
                disclaimer = "\n\nNote: This information was generated online because we couldn't find it in your uploaded documents.\n\n"
                
                sources = "\n\nSources:\n"
                for i, r in enumerate(filtered_results, 1):
                    sources += f"  {i}. {r.get('title', 'No title')}\n     {r.get('href', '')}\n"
                
                print("✅ Answer generated successfully")
                print("📊 Formatting response with sources...")
                print("="*60)
                print("🎉 DuckDuckGo search completed successfully!")
                print("="*60 + "\n")
                return disclaimer + answer + "\n\n" + sources
            except Exception as e:
                print(f"❌ Error generating answer: {e}")
                print("="*60 + "\n")
                return f"Error processing search results: {e}"
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ DuckDuckGo search failed: {error_msg[:200]}")
            print("="*60 + "\n")
            
            # Provide helpful error message based on the error type
            if "timeout" in error_msg.lower():
                return ("DuckDuckGo search timed out.\n\n"
                       "This usually means DuckDuckGo is blocking automated requests.\n\n"
                       "Solutions:\n"
                       "• Switch to Gemini (recommended, free): Get API key at https://makersuite.google.com/app/apikey\n"
                       "• Wait several minutes before trying DuckDuckGo again\n"
                       "• Use OpenAI (paid) as alternative")
            elif "ratelimit" in error_msg.lower() or "rate" in error_msg.lower() or "429" in error_msg or "202" in error_msg:
                return ("DuckDuckGo is rate-limiting or blocking automated requests.\n\n"
                       "This is a common issue with DuckDuckGo's free service.\n\n"
                       "Recommended solution:\n"
                       "• Switch to Gemini search engine (free, more reliable)\n"
                       "  Get API key at: https://makersuite.google.com/app/apikey\n"
                       "• Add the API key to your .env file as: GEMINI_API_KEY=your_key_here\n"
                       "• Change search engine to Gemini in the interface")
            else:
                return (f"DuckDuckGo search encountered an error.\n\n"
                       f"Error details: {error_msg[:300]}\n\n"
                       "Recommendations:\n"
                       "• DuckDuckGo frequently blocks automated searches\n"
                       "• Switch to Gemini (free, get API key at: https://makersuite.google.com/app/apikey)\n"
                       "• Or use OpenAI (paid, requires API key)\n\n"
                       "Gemini is the recommended alternative for reliable web search.")

    elif SEARCH_ENGINE == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print("❌ Gemini API key not found in .env file")
            print("="*60 + "\n")
            return "Gemini API key not found in .env file."
        
        try:
            print("🔍 Using Gemini AI (requires API key)...")
            print("⏳ Configuring Gemini API...")
            genai.configure(api_key=gemini_api_key)
            
            # List available models and find one that supports generateContent
            print("   Finding available Gemini models...")
            try:
                available_models = genai.list_models()
                suitable_model = None
                
                for m in available_models:
                    # Find a model that supports generateContent
                    if 'generateContent' in m.supported_generation_methods:
                        suitable_model = m.name
                        print(f"   ✓ Found working model: {suitable_model}")
                        break
                
                if not suitable_model:
                    return "Gemini API error: No models available that support text generation. Please verify your API key at https://makersuite.google.com/app/apikey"
                
                model = genai.GenerativeModel(suitable_model)
                
            except Exception as e:
                print(f"   ✗ Could not list models: {str(e)[:200]}")
                return f"Gemini API error: Could not access Gemini models. Your API key may be invalid or expired. Error: {str(e)[:200]}\n\nPlease get a new API key at: https://makersuite.google.com/app/apikey"
            
            print("⏳ Sending query to Gemini AI...")
            
            enhanced_query = f"""Answer the following question in a clear, professional format:

IMPORTANT:
- Use simple dashes (-) for bullet points
- Use numbers (1. 2. 3.) for sequential steps
- Write clear paragraphs separated by blank lines
- Do NOT use special symbols like • or ** or other formatting marks
- Keep the language professional and easy to read

Question: {query}

Provide a detailed, well-structured answer:"""
            
            response = model.generate_content(enhanced_query)
            print("✅ Gemini AI response received")
            print("="*60)
            print("🎉 Gemini search completed successfully!")
            print("="*60 + "\n")
            disclaimer = "\n\nNote: This information was generated online because we couldn't find it in your uploaded documents.\n\n"
            return disclaimer + format_response(response.text)
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            print("="*60 + "\n")
            return f"Gemini API error: {e}"

    elif SEARCH_ENGINE == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("❌ OpenAI API key not found in .env file")
            print("="*60 + "\n")
            return "OpenAI API key not found in .env file."
        
        try:
            print("🔍 Using OpenAI GPT (requires API key)...")
            print("⏳ Sending query to OpenAI...")
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Structure your responses clearly using simple dashes (-) for bullet points, numbers (1. 2. 3.) for steps, and clear paragraphs. Do NOT use special symbols like • or ** or other formatting marks. Keep the language professional and easy to read."},
                    {"role": "user", "content": query}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            print("✅ OpenAI response received")
            print("="*60)
            print("🎉 OpenAI search completed successfully!")
            print("="*60 + "\n")
            disclaimer = "\n\nNote: This information was generated online because we couldn't find it in your uploaded documents.\n\n"
            return disclaimer + format_response(response.choices[0].message.content)
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            print("="*60 + "\n")
            return f"OpenAI API error: {e}"

    return "Search engine not configured."

@app.route("/test", methods=["POST"])
def test():
    # Minimal test - no file I/O, no complex ops
    return jsonify({"response": "Test OK"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        log_step("🔄 [CHAT] Request received")
        print(f"🔄 Chat request received")
        return chat_impl()
    except Exception as e:
        import traceback
        error_msg = str(e)[:300]
        log_step(f"❌ EXCEPTION: {error_msg}")
        print(f"❌ EXCEPTION in chat route:")
        print(traceback.format_exc())
        return jsonify({"response": f"Error: {error_msg}"}), 500

def chat_impl():
    request_started = time.perf_counter()
    query = request.json.get("query", "")
    if not query:
        return jsonify({"response": "No query provided."})
    
    log_step("\n" + "="*60)
    log_step(f"💬 NEW QUERY: {query[:80]}")
    log_step("="*60)
    
    print("\n" + "#"*60)
    print(f"💬 NEW QUERY RECEIVED")
    print(f"Query: {query}")
    print(f"⚡ Model Settings: ctx=8192, max_output={ANSWER_MAX_TOKENS} tokens, batch=256")
    print("#"*60)
    print(f"🔧 Current Mode: {'OFFLINE' if OFFLINE_ONLY else 'ONLINE'}")
    active_source_file = get_effective_source_file()

    if SOURCE_LOCK_ENABLED and active_source_file:
        query_source_file = active_source_file
    else:
        query_source_file = None

    # If ONLINE mode is enabled, search the web directly
    if not OFFLINE_ONLY:
        log_step("🌐 ONLINE MODE: Searching the web directly...")
        print("🌐 ONLINE MODE: Searching the web directly...")
        print("#"*60)
        response = search_online(query)
        return jsonify({"response": response})

    # OFFLINE MODE: Check if we have documents
    try:
        doc_count = collection.count()
        has_documents = doc_count > 0
        log_step(f"📚 Documents in database: {doc_count} chunks")
        print(f"📚 Documents in database: {'Yes' if has_documents else 'No'} (chunks: {doc_count})")
    except Exception:
        has_documents = False
        log_step("⚠️ Could not check document database")
        print("⚠️ Could not check document database")

    # If no documents and offline mode
    if not has_documents:
        log_step("❌ No documents uploaded")
        print("❌ No documents uploaded and in OFFLINE mode")
        print("#"*60 + "\n")
        return jsonify({"response": "No documents uploaded. Please upload a PDF document to get started."})

    # Query the vector database
    log_step(f"🔍 Searching {doc_count} chunks for relevant context...")
    print("🔍 Searching in uploaded documents...")
    print("⏳ Generating query embedding...")
    print(f"   Search limits: fast={FAST_RESULT_LIMIT}, full={FULL_RESULT_LIMIT}, file_cap={MAX_SOURCE_FILES_PER_QUERY}")
    if SOURCE_LOCK_ENABLED and active_source_file:
        print(f"📌 Source lock active: {active_source_file}")
    elif query_source_file:
        print(f"📁 Query scope: {query_source_file}")

    # Scope search: source-lock overrides everything, otherwise search all session files
    if query_source_file:
        where_filter = {"source_file": query_source_file}
    elif AUTO_SCOPE_LAST_UPLOAD and len(SESSION_UPLOADED_FILES) > 0:
        if len(SESSION_UPLOADED_FILES) == 1:
            where_filter = {"source_file": list(SESSION_UPLOADED_FILES)[0]}
        else:
            where_filter = {"source_file": {"$in": list(SESSION_UPLOADED_FILES)}}
        log_step(f"📂 Auto-scoped to {len(SESSION_UPLOADED_FILES)} uploaded file(s)")
        print(f"📂 Auto-scoped to {len(SESSION_UPLOADED_FILES)} uploaded file(s)")
    else:
        where_filter = None
    embedding_started = time.perf_counter()
    query_embedding = get_query_embedding(query)
    embedding_ms = (time.perf_counter() - embedding_started) * 1000
    log_step(f"⏱ Embedding ready ({embedding_ms:.0f}ms)")
    print(f"⏱ Embedding generated in {embedding_ms:.1f} ms")

    def run_vector_search(limit):
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": limit,
            "include": ["documents", "distances", "metadatas"]
        }
        if where_filter:
            kwargs["where"] = where_filter
        return collection.query(**kwargs)

    def process_results(results, limit, max_files):
        docs = results.get("documents", [[]])[0] if results.get("documents") else []
        ids = results.get("ids", [[]])[0] if results.get("ids") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []

        if not docs:
            return {
                "context": "",
                "context_chunks": [],
                "distances": [],
                "source_files": [],
                "handwriting_source_files": [],
                "best_distance": None
            }

        combined = [
            {
                'id': chunk_id,
                'chunk': chunk,
                'distance': distance,
                'file': (metadata.get('source_file') if isinstance(metadata, dict) and metadata.get('source_file') else "_".join(chunk_id.split("_")[:-1])),
                'chunk_index': metadata.get('chunk_index') if isinstance(metadata, dict) else None,
                'handwriting_source': bool(metadata.get('handwriting_source')) if isinstance(metadata, dict) else False
            }
            for chunk_id, chunk, distance, metadata in zip(ids, docs, distances, metadatas)
        ]

        filtered = []
        allowed_files = []
        for item in combined:
            file_name = item['file']
            if max_files and file_name not in allowed_files:
                if len(allowed_files) >= max_files:
                    continue
                allowed_files.append(file_name)
            filtered.append(item)

        selected = filtered[:limit]
        context_chunks = [item['chunk'] for item in selected]
        selected_distances = [item['distance'] for item in selected]
        context = "\n\n---\n\n".join(context_chunks)
        source_files = list(dict.fromkeys(item['file'] for item in selected))
        handwriting_files = list(dict.fromkeys(item['file'] for item in selected if item.get('handwriting_source')))

        best_distance = min(selected_distances) if selected_distances else None

        return {
            "context": context,
            "context_chunks": context_chunks,
            "distances": selected_distances,
            "source_files": source_files,
            "handwriting_source_files": handwriting_files,
            "best_distance": best_distance,
            "selected_count": len(selected)
        }

    file_cap = MAX_SOURCE_FILES_PER_QUERY

    # Single broad search instead of multiple passes — faster and more reliable
    search_limit = FULL_RESULT_LIMIT
    search_started = time.perf_counter()
    results = run_vector_search(search_limit)
    search_ms = (time.perf_counter() - search_started) * 1000
    result_count = len(results.get('documents', [[]])[0]) if results.get('documents') else 0
    log_step(f"🔍 Got {result_count} results in {search_ms:.0f}ms")
    payload = process_results(results, FULL_RESULT_LIMIT, file_cap)

    context = payload["context"]
    context_chunks = payload["context_chunks"]
    distances = payload["distances"]
    source_files = payload["source_files"]
    handwriting_source_files = payload["handwriting_source_files"]
    best_distance = payload["best_distance"]

    # Check relevance
    is_relevant = False
    
    if distances and len(distances) > 0:
        best_distance = min(distances)
        is_relevant = best_distance < RELEVANCE_DISTANCE_THRESHOLD
        if is_relevant:
            log_step(f"✅ Found relevant information from: {', '.join(source_files)}")
        else:
            log_step(f"⚠️  Best distance {best_distance:.3f} > threshold {RELEVANCE_DISTANCE_THRESHOLD}")

    # If initial search missed, try relaxed threshold + lexical fallback
    if not context or not is_relevant:
        # Check if the results we already have pass a relaxed threshold
        if context and best_distance is not None and best_distance < RELAXED_RELEVANCE_DISTANCE_THRESHOLD:
            log_step(f"✅ Passes relaxed threshold ({best_distance:.3f} < {RELAXED_RELEVANCE_DISTANCE_THRESHOLD})")
            is_relevant = True
        else:
            print("\n❌ Semantic search missed. Attempting lexical fallback...")

            fallback_match = lexical_fallback_search(query, query_source_file)

            if fallback_match:
                log_step(f"✅ Found via keyword search")
                fallback_file, fallback_snippet = fallback_match
                print(f"   🔎 Lexical fallback hit in {fallback_file}")
                response = generate_answer(fallback_snippet, query)
                print("#"*60 + "\n")
                return jsonify({"response": response})

            log_step("❌ No relevant information found")
            return jsonify({"response": "No relevant information found in uploaded documents. Try rephrasing your question with more specific keywords."})

    # Generate answer from local documents
    log_step(f"🤖 Generating answer...")
    generation_started = time.perf_counter()
    response = generate_answer(context, query)
    generation_ms = (time.perf_counter() - generation_started) * 1000
    log_step(f"✅ Done ({generation_ms:.0f}ms)")
    
    # Calculate total time
    total_ms = (time.perf_counter() - request_started) * 1000
    print(f"\n⏱️  TOTAL TIME: {total_ms/1000:.1f} seconds")
    print(f"   └─ LLM Generation: {generation_ms/1000:.1f}s")
    print(f"   └─ Search + Other: {(total_ms - generation_ms)/1000:.1f}s")
    
    # Show all source files used (console only)
    if source_files:
        source_labels = []
        for source_file in source_files:
            if source_file in handwriting_source_files:
                source_labels.append(f"{source_file} [handwritten OCR]")
            else:
                source_labels.append(source_file)
        source_indicator = f"Sources: {', '.join(source_labels)}\n\n"
    else:
        source_indicator = "Sources: none\n\n"
    
    response = source_indicator + response
    return jsonify({"response": response})

@app.route("/status", methods=["GET"])
def status():
    try:
        count = len(collection.get(ids=None)["ids"])
    except:
        count = "unknown"
    
    return jsonify({
        "offline_only": OFFLINE_ONLY,
        "collection_documents": count,
        "mode": "offline" if OFFLINE_ONLY else "online",
        "search_engine": SEARCH_ENGINE,
        "handwriting_ocr_enabled": ENABLE_HANDWRITING_OCR,
        "printed_trocr_enabled": ENABLE_PRINTED_TROCR,
        "strict_gpu_mode": STRICT_GPU_MODE,
        "tesseract_ocr_enabled": ENABLE_TESSERACT_OCR,
        "trocr_device": trocr_device,
        "source_lock_enabled": SOURCE_LOCK_ENABLED,
        "source_lock_file": get_effective_source_file(),
        "last_uploaded_file": LAST_UPLOADED_FILE,
        "session_uploaded_files": list(SESSION_UPLOADED_FILES)
    })

@app.route("/set-source-file", methods=["POST"])
def set_source_file():
    global SOURCE_LOCK_FILE

    source_file = request.json.get("source_file", "").strip()
    SOURCE_LOCK_FILE = source_file

    if SOURCE_LOCK_FILE:
        print(f"Source lock file set to: {SOURCE_LOCK_FILE}")
        message = f"Source lock file set to {SOURCE_LOCK_FILE}"
    else:
        print("Source lock file cleared (will use latest upload when available)")
        message = "Source lock file cleared"

    return jsonify({
        "status": "ok",
        "source_lock_enabled": SOURCE_LOCK_ENABLED,
        "source_lock_file": get_effective_source_file(),
        "message": message
    })

@app.route("/set-search-engine", methods=["POST"])
def set_search_engine():
    global SEARCH_ENGINE
    engine = request.json.get("engine", "duckduckgo").lower()
    
    valid_engines = ["duckduckgo", "gemini", "openai"]
    if engine not in valid_engines:
        return jsonify({
            "status": "error",
            "message": f"Invalid engine. Choose from: {', '.join(valid_engines)}"
        }), 400
    
    SEARCH_ENGINE = engine
    print(f"Search engine changed to: {engine}")
    
    return jsonify({
        "status": "ok",
        "search_engine": SEARCH_ENGINE,
        "message": f"Search engine changed to {engine}"
    })

@app.route("/toggle", methods=["POST"])
def toggle():
    global OFFLINE_ONLY
    OFFLINE_ONLY = request.json.get("offline", True)
    
    mode = "OFFLINE" if OFFLINE_ONLY else "ONLINE"
    print(f"Switched to {mode} mode")
    
    return jsonify({
        "status": "ok",
        "offline_only": OFFLINE_ONLY,
        "message": f"Successfully switched to {mode} mode"
    })

@app.route("/debug/database", methods=["GET"])
def debug_database():
    """Diagnostic endpoint to inspect what's stored in the vector database"""
    try:
        # Get all documents
        all_data = collection.get()
        ids = all_data.get('ids', [])
        documents = all_data.get('documents', [])
        
        print("\n" + "="*60)
        print("📊 DATABASE DIAGNOSTIC")
        print("="*60)
        print(f"Total chunks stored: {len(ids)}")
        
        # Group by file
        files = {}
        for doc_id, doc in zip(ids, documents):
            filename = "_".join(doc_id.split("_")[:-1])  # Remove chunk number
            if filename not in files:
                files[filename] = []
            files[filename].append(doc)
        
        print(f"Files in database: {len(files)}")
        for filename, chunks in files.items():
            print(f"  • {filename}: {len(chunks)} chunks")
        
        # Show sample of first few chunks
        sample_chunks = []
        for i, (doc_id, doc) in enumerate(zip(ids[:5], documents[:5])):
            preview = doc[:200].replace('\n', ' ')
            sample_chunks.append({
                "id": doc_id,
                "preview": preview,
                "length": len(doc)
            })
            print(f"\n  Sample chunk {i+1}:")
            print(f"    ID: {doc_id}")
            print(f"    Length: {len(doc)} chars")
            print(f"    Preview: {preview}...")
        
        print("="*60 + "\n")
        
        return jsonify({
            "total_chunks": len(ids),
            "files": {filename: len(chunks) for filename, chunks in files.items()},
            "sample_chunks": sample_chunks
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/debug/file/<path:filename>", methods=["GET"])
def debug_file(filename):
    """Inspect all chunks from a specific file"""
    try:
        # Get all documents
        all_data = collection.get()
        ids = all_data.get('ids', [])
        documents = all_data.get('documents', [])
        
        # Filter for this specific file
        file_chunks = []
        for doc_id, doc in zip(ids, documents):
            if doc_id.startswith(filename + "_"):
                file_chunks.append({
                    "id": doc_id,
                    "content": doc,
                    "length": len(doc),
                    "preview": doc[:300].replace('\n', ' ')
                })
        
        print("\n" + "="*60)
        print(f"📄 FILE INSPECTION: {filename}")
        print("="*60)
        print(f"Found {len(file_chunks)} chunks")
        for i, chunk in enumerate(file_chunks):
            print(f"\nChunk {i+1}:")
            print(f"Length: {chunk['length']} chars")
            print(f"Preview: {chunk['preview']}...")
        print("="*60 + "\n")
        
        if not file_chunks:
            return jsonify({"error": f"File '{filename}' not found in database"}), 404
        
        return jsonify({
            "filename": filename,
            "chunk_count": len(file_chunks),
            "chunks": file_chunks
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    flask_debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"  # Enable debug to see exceptions
    flask_reloader = os.getenv("FLASK_USE_RELOADER", "false").lower() == "true"

    print("\nStarting Flask server...")
    print("Server will be available at: http://localhost:6969")
    print("Ready to process documents and queries!")
    print(f"Flask debug={flask_debug}, reloader={flask_reloader}")
    print("=" * 60 + "\n")
    
    # Enable logging to see HTTP requests and errors
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Show all Flask errors and requests
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(log_handler)
    
    # Also log werkzeug requests (Flask's underlying server)
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.INFO)
    
    app.run(debug=flask_debug, use_reloader=flask_reloader, port=6969)

# Made with love by CoCo