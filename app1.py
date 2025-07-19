from flask import Flask, request, render_template, jsonify
import os
from dotenv import load_dotenv
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text
from llama_cpp import Llama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from googleapiclient.discovery import build

load_dotenv()

UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'uploads/images'
VECTORDB_FOLDER = 'vector_store'

MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(VECTORDB_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Always use GPU for embedding
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA not available — GPU required for this app.")

device = "cuda"
print(f"✅ SentenceTransformer running on: {device}")
embed_model = SentenceTransformer("all-mpnet-base-v2", device=device)

OFFLINE_ONLY = True  # default state of online/offline toggle

def embed_text(text):
    return embed_model.encode([text])[0]

def parse_pdf_text(path):
    return extract_text(path)

def extract_images_and_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    image_texts = []
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            image_texts.append(text)
    return "\n".join(image_texts)

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

llm_ctx = 4096
llm_gpu_layers = -1  # fully use GPU
print("✅ Initializing LLaMA … with GPU only.")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=llm_gpu_layers,
    n_ctx=llm_ctx
)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["pdf"]
    if file:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        text = parse_pdf_text(path)
        image_text = extract_images_and_ocr(path)
        combined_text = text + "\n" + image_text

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(combined_text)

        print(f"📄 PDF split into {len(chunks)} chunks. Adding to vector DB …")
        for idx, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="🔷 Processing chunks"):
            collection.add(
                documents=[chunk],
                embeddings=[embed_text(chunk)],
                ids=[f"{file.filename}_{idx}"]
            )
        print("✅ Upload and processing complete.")
        return jsonify({"status": "success", "message": "✅ PDF uploaded & processed"})
    return jsonify({"status": "error", "message": "No file uploaded"})

@app.route("/chat", methods=["POST"])
def chat():
    query = request.json["query"]

    results = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=5
    )
    context = "\n".join(results['documents'][0])

    if not context and not OFFLINE_ONLY:
        return jsonify({"response": search_online(query)})
    elif not context:
        return jsonify({"response": "No relevant information found in uploaded manuals."})

    response_text = generate_answer(context, query)
    return jsonify({"response": response_text})

def generate_answer(context, question):
    prompt = f"""
You are a helpful assistant. Answer the question based on the following manual content.

Manual Content:
\"\"\"
{context}
\"\"\"

Question: {question}
Answer:"""

    try:
        output = llm(prompt, max_tokens=300, stop=["\n"])
        answer = output['choices'][0]['text'].strip()
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"

@app.route("/status", methods=["GET"])
def status():
    try:
        count = len(collection.get(ids=None)["ids"])
    except:
        count = "unknown"
    return jsonify({
        "embedding_device": device,
        "llm_gpu_layers": llm_gpu_layers,
        "offline_only": OFFLINE_ONLY,
        "collection_documents": count
    })

@app.route("/toggle", methods=["POST"])
def toggle():
    global OFFLINE_ONLY
    OFFLINE_ONLY = request.json["offline"]
    return jsonify({"status": "ok", "offline_only": OFFLINE_ONLY})

def search_online(query):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cx = os.getenv("GOOGLE_CX")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if google_api_key and google_cx:
        service = build("customsearch", "v1", developerKey=google_api_key)
        res = service.cse().list(q=query, cx=google_cx).execute()
        if 'items' in res:
            return res['items'][0]['snippet']
        else:
            return "No results found online."
    elif gemini_api_key:
        return f"Gemini search stub for query: '{query}'"
    else:
        return "No online search configured."

if __name__ == "__main__":
    app.run(debug=True)
