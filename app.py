from flask import Flask, request, render_template, jsonify
import os
import re
from dotenv import load_dotenv
import chromadb
import torch

# Set offline mode for transformers to prevent network calls during startup
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text
from llama_cpp import Llama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import io
import google.generativeai as genai
import platform
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from functools import lru_cache

# Configure Tesseract for Windows
if platform.system() == 'Windows':
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
        print("  OCR may not work. Install from: https://github.com/UB-Mannheim/tesseract/wiki")
from openai import OpenAI
from duckduckgo_search import DDGS

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
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Always use GPU for embedding
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available - GPU required for this app.")

device = "cuda"
print(f"SentenceTransformer running on: {device}")

# Load SentenceTransformer (offline mode enabled via environment variables)
try:
    embed_model = SentenceTransformer("all-mpnet-base-v2", device=device)
    print("✓ Loaded embedding model successfully")
except Exception as e:
    print(f"ERROR: Could not load embedding model: {e}")
    print("SOLUTION: The model needs to be downloaded once with internet connection.")
    print("After the first download, the model will be cached locally and work offline.")
    raise RuntimeError(f"Failed to load embedding model. Please connect to internet for first-time setup. Error: {e}")

OFFLINE_ONLY = True
SEARCH_ENGINE = "gemini"  # Changed from duckduckgo to gemini for better reliability
ENABLE_HANDWRITING_OCR = True
SOURCE_LOCK_ENABLED = False
SOURCE_LOCK_FILE = os.getenv("SOURCE_LOCK_FILE", "").strip()
LAST_UPLOADED_FILE = ""
HANDWRITING_SOURCE_FILES = {}
FAST_RESULT_LIMIT = 6
FULL_RESULT_LIMIT = 12
QUERY_EMBED_CACHE_SIZE = 64
RELEVANCE_DISTANCE_THRESHOLD = float(os.getenv("RELEVANCE_DISTANCE_THRESHOLD", "1.25"))
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
ENABLE_LEXICAL_FALLBACK = os.getenv("ENABLE_LEXICAL_FALLBACK", "false").lower() == "true"
LEXICAL_STOP_WORDS = {
    "what", "is", "are", "the", "a", "an", "how", "why", "when", "where",
    "who", "which", "tell", "me", "about", "please", "find", "show", "give",
    "my", "his", "her", "their", "does", "did", "can", "you", "explain"
}

trocr_processor = None
trocr_model = None
trocr_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load TrOCR handwritten model (optional, offline-first).
if ENABLE_HANDWRITING_OCR:
    try:
        trocr_model_name = "microsoft/trocr-base-handwritten"
        trocr_processor = TrOCRProcessor.from_pretrained(trocr_model_name)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_model_name).to(trocr_device)
        trocr_model.eval()
        print(f"✓ Handwriting OCR enabled with TrOCR on: {trocr_device}")
    except Exception as e:
        ENABLE_HANDWRITING_OCR = False
        print(f"⚠ Handwriting OCR unavailable: {e}")
        print("  Continuing with Tesseract OCR only.")

print("=" * 60)
print("PROMPTLY STARTING...")
print("=" * 60)
print(f"Initial Mode: {'OFFLINE MODE' if OFFLINE_ONLY else 'ONLINE MODE'}")
print("=" * 60)

def embed_text(text):
    return embed_model.encode([text])[0]

@lru_cache(maxsize=QUERY_EMBED_CACHE_SIZE)
def _cached_query_embedding(text):
    return embed_model.encode([text])[0]

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

    gray = ImageOps.autocontrast(gray)
    if mode == "high_contrast":
        gray = ImageEnhance.Contrast(gray).enhance(2.2)
        gray = ImageEnhance.Sharpness(gray).enhance(1.8)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
        gray = gray.point(lambda x: 255 if x > 170 else 0)
    else:
        gray = ImageEnhance.Contrast(gray).enhance(1.4)
        gray = ImageEnhance.Sharpness(gray).enhance(1.5)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))

    return gray

def ocr_with_trocr(image):
    """Run handwritten OCR on a PIL image with TrOCR."""
    if not ENABLE_HANDWRITING_OCR or trocr_processor is None or trocr_model is None:
        return ""

    try:
        rgb_image = preprocess_image_for_ocr(image, mode="high_contrast").convert("RGB")
        pixel_values = trocr_processor(images=rgb_image, return_tensors="pt").pixel_values.to(trocr_device)
        generated_ids = trocr_model.generate(pixel_values)
        text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return normalize_extracted_text(text)
    except Exception as e:
        print(f"  ⚠ TrOCR failed: {e}")
        return ""

def run_ocr_pipeline_on_image(image):
    candidates = []

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

    if ENABLE_HANDWRITING_OCR:
        trocr_text = ocr_with_trocr(image)
        if trocr_text:
            candidates.append((trocr_text, True, score_text_quality(trocr_text) + 1.0))

    if not candidates:
        return "", False, 0.0

    best_text, best_handwriting, best_score = select_best_candidate(candidates)

    handwriting_texts = [candidate[0] for candidate in candidates if candidate[1] and candidate[2] >= best_score - 0.75]
    if handwriting_texts and not best_handwriting:
        merged = merge_text_candidates([best_text] + handwriting_texts)
        merged_score = score_text_quality(merged)
        if merged and merged_score >= best_score - 0.25:
            best_text = merged
            best_score = merged_score
            best_handwriting = True

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
    n_ctx=4096,
    n_batch=512
)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global LAST_UPLOADED_FILE, SOURCE_LOCK_FILE, HANDWRITING_SOURCE_FILES

    file = request.files.get("pdf") or request.files.get("file")
    if not file:
        return jsonify({"status": "error", "message": "No file uploaded"})

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
    file.save(path)

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

        print("  🖼️ Extracting images and performing OCR...")
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
        message = "No text could be extracted from the uploaded file. Ensure the content is clear and try again."
        if ocr_result.get("error"):
            message += f" Details: {ocr_result['error']}"
        return jsonify({"status": "error", "message": message})

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120
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

    print(f"{doc_label} split into {len(chunks)} chunk(s). Adding to vector DB...")
    for idx, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Processing chunks"):
        collection.add(
            documents=[chunk],
            embeddings=[embed_text(chunk)],
            ids=[f"{filename}_{idx}"],
            metadatas=[{
                "source_file": filename,
                "chunk_index": idx,
                "handwriting_source": file_has_handwriting
            }]
        )

    LAST_UPLOADED_FILE = filename
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
    prompt = f"""You are a helpful and thorough document assistant. Answer the question using ONLY the document content provided below. Do NOT use outside knowledge.

STRICT RULES:
- If the answer is clearly present, give a COMPREHENSIVE and DETAILED response using ALL relevant information from the context
- If the answer is NOT in the document at all, respond only with: "This information was not found in the uploaded documents."
- Do NOT guess or fill in gaps from general knowledge
- Never add "additional information" about topics that were not requested.
- Do not mention any language, technology, person, or topic that is not directly supported by the provided document context.
- If the context does not contain a direct answer, say the answer was not found instead of trying to connect unrelated material.

FORMATTING RULES:
- Start with a direct 1-2 sentence summary answer
- Then expand with full details, examples, and specifics from the document
- Use dashes (-) for bullet point lists
- Use numbers (1. 2. 3.) for steps or procedures
- Separate sections with a blank line for readability
- Do NOT use symbols like * or # or bold markers
- Write in clear, professional paragraphs where appropriate

Document Content:
{context}

Question: {question}

Detailed Answer:"""

    try:
        output = llm(prompt, max_tokens=1500, temperature=0.4, stop=["Question:", "Document Content:"])
        return format_response(output['choices'][0]['text'])
    except Exception as e:
        return f"Error generating answer: {e}"

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
                output = llm(prompt, max_tokens=1000, temperature=0.7, stop=["Question:", "Search Results:"])
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

@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("query", "")
    if not query:
        return jsonify({"response": "No query provided."})
    
    print("\n" + "#"*60)
    print(f"💬 NEW QUERY RECEIVED")
    print(f"Query: {query}")
    print("#"*60)
    print(f"🔧 Current Mode: {'OFFLINE' if OFFLINE_ONLY else 'ONLINE'}")
    active_source_file = get_effective_source_file()

    if SOURCE_LOCK_ENABLED and active_source_file:
        query_source_file = active_source_file
    elif AUTO_SCOPE_LAST_UPLOAD and LAST_UPLOADED_FILE:
        query_source_file = LAST_UPLOADED_FILE
        print(f"📂 Auto-focused on most recent upload: {query_source_file}")
    else:
        query_source_file = None

    # If ONLINE mode is enabled, search the web directly
    if not OFFLINE_ONLY:
        print("🌐 ONLINE MODE: Searching the web directly...")
        print("#"*60)
        response = search_online(query)
        return jsonify({"response": response})

    # OFFLINE MODE: Check if we have documents
    try:
        doc_count = collection.count()
        has_documents = doc_count > 0
        print(f"📚 Documents in database: {'Yes' if has_documents else 'No'} (chunks: {doc_count})")
    except Exception:
        has_documents = False
        print("⚠️ Could not check document database")

    # If no documents and offline mode
    if not has_documents:
        print("❌ No documents uploaded and in OFFLINE mode")
        print("#"*60 + "\n")
        return jsonify({"response": "No documents uploaded. Please upload a PDF document to get started."})

    # Query the vector database
    print("🔍 Searching in uploaded documents...")
    print("⏳ Generating query embedding...")
    if SOURCE_LOCK_ENABLED and active_source_file:
        print(f"📌 Source lock active: {active_source_file}")
    elif query_source_file:
        print(f"📁 Query scope: {query_source_file}")

    where_filter = {"source_file": query_source_file} if query_source_file else None
    query_embedding = get_query_embedding(query)

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

        print(f"\n📁 Selected chunks (top {len(selected)} by relevance):")
        for item in selected:
            source_type = "handwriting" if item.get('handwriting_source') else "standard"
            print(f"   • [{item['distance']:.4f}] {item['file']} ({source_type}) — {item['chunk'][:80].replace(chr(10), ' ')}...")

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

    file_cap = 1 if query_source_file else MAX_SOURCE_FILES_PER_QUERY

    search_limit = max(FAST_RESULT_LIMIT, FULL_RESULT_LIMIT)
    print(f"⚡ Running vector search (top {search_limit})")
    results = run_vector_search(search_limit)
    payload = process_results(results, FAST_RESULT_LIMIT, file_cap)

    needs_expansion = (
        FULL_RESULT_LIMIT > FAST_RESULT_LIMIT and (
            len(payload["context_chunks"]) < 3 or
            (payload["best_distance"] is not None and payload["best_distance"] > 0.85)
        )
    )

    if needs_expansion:
        print(f"⚡ Sparse results, reusing cached search results up to top {FULL_RESULT_LIMIT} chunks...")
        payload = process_results(results, FULL_RESULT_LIMIT, file_cap)

    context = payload["context"]
    context_chunks = payload["context_chunks"]
    distances = payload["distances"]
    source_files = payload["source_files"]
    handwriting_source_files = payload["handwriting_source_files"]
    best_distance = payload["best_distance"]

    # Check relevance
    is_relevant = False
    
    print("\n📊 Analyzing relevance of search results...")
    print(f"   Found {len(context_chunks)} chunks from {len(source_files)} file(s)")
    if SOURCE_LOCK_ENABLED and active_source_file:
        print(f"   Source lock file: {active_source_file}")
    
    # Show previews of top 5 matches with ALL scores
    if context_chunks and distances:
        print(f"   Top {len(context_chunks)} matches across {len(source_files)} file(s):")
        for i in range(len(context_chunks)):
            preview = context_chunks[i][:150].replace('\n', ' ')
            print(f"   [{i+1}] Distance: {distances[i]:.4f} | Preview: {preview}...")
    
    if distances and len(distances) > 0:
        best_distance = min(distances)
        is_relevant = best_distance < RELEVANCE_DISTANCE_THRESHOLD
        print(f"   Best match distance: {best_distance:.4f}")
        print(f"   Relevance threshold: {RELEVANCE_DISTANCE_THRESHOLD:.2f}")
        print(f"   Result: {'✅ RELEVANT' if is_relevant else '❌ NOT RELEVANT (try rephrasing)'}")

    # If not relevant in offline mode
    if not context or not is_relevant:
        print("\n❌ Semantic search missed. Attempting lexical fallback...")
        if distances and len(distances) > 0:
            print(f"   Closest match was {best_distance:.4f} (threshold: {RELEVANCE_DISTANCE_THRESHOLD:.2f})")

        fallback_match = lexical_fallback_search(query, query_source_file)

        if fallback_match:
            fallback_file, fallback_snippet = fallback_match
            print(f"   🔎 Lexical fallback hit in {fallback_file}")
            response = generate_answer(fallback_snippet, query)
            response = f"Sources: {fallback_file}\n\n" + response
            print("#"*60 + "\n")
            return jsonify({"response": response})

        print("   • Try using keywords directly (e.g., use 'engineer' instead of 'profession')")
        print("   • Try asking differently (e.g., 'Tell me about X' instead of 'What is X')")
        print("   • The semantic embedding might not capture the relationship")
        print("🔒 Current Mode: OFFLINE - Will not search online")
        print("#"*60 + "\n")
        if SOURCE_LOCK_ENABLED and active_source_file:
            return jsonify({"response": f"No relevant information found in '{active_source_file}'. Try rephrasing your question with more specific keywords from that file."})
        return jsonify({"response": "No relevant information found in uploaded documents. Try rephrasing your question with more specific keywords from the document."})

    # Generate answer from local documents
    print("\n" + "*"*60)
    print(f"✅ Using information from: {', '.join(source_files)}")
    print("*"*60)
    print("🤖 Generating answer using local LLaMA model...")
    response = generate_answer(context, query)
    print("✅ Answer generated successfully from local documents")
    print("#"*60 + "\n")
    
    # Show all source files used
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
        "source_lock_enabled": SOURCE_LOCK_ENABLED,
        "source_lock_file": get_effective_source_file(),
        "last_uploaded_file": LAST_UPLOADED_FILE
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
    print("\nStarting Flask server...")
    print("Server will be available at: http://localhost:6969")
    print("Ready to process documents and queries!")
    print("=" * 60 + "\n")
    app.run(debug=True, port=6969)

# Made with love by CoCo
