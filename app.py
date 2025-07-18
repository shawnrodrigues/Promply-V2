from flask import Flask, request, render_template, jsonify
import os
from dotenv import load_dotenv
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text
from googleapiclient.discovery import build
from llama_cpp import Llama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Load .env
load_dotenv()

# Config
UPLOAD_FOLDER = 'uploads'
VECTORDB_FOLDER = 'vector_store'
OFFLINE_ONLY = True
GPU_ONLY = True  # ✅ new toggle

MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORDB_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Detect GPU and initialize embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
if GPU_ONLY and device != "cuda":
    raise RuntimeError("🚫 GPU_ONLY is enabled but no CUDA GPU was detected!")

print(f"✅ SentenceTransformer initialized on: {device}")
embed_model_name = "all-mpnet-base-v2"
embed_model = SentenceTransformer(embed_model_name, device=device)

def embed_text(text):
    return embed_model.encode([text])[0]

def parse_pdf(path):
    return extract_text(path)

# ChromaDB client
chroma_client = chromadb.PersistentClient(path=VECTORDB_FOLDER)

# ✅ Auto-detect and fix collection dimension mismatch
def get_or_recreate_collection(client, name, expected_dim):
    try:
        col = client.get_collection(name=name)
        dummy_vec = embed_text("test")
        if len(dummy_vec) != expected_dim:
            print(f"⚠️ Detected embedding dimension mismatch: expected {expected_dim}, got {len(dummy_vec)}. Recreating collection.")
            client.delete_collection(name)
            col = client.create_collection(name=name)
        return col
    except Exception:
        return client.create_collection(name=name)

expected_dim = SentenceTransformer(embed_model_name).get_sentence_embedding_dimension()
collection = get_or_recreate_collection(chroma_client, "manuals", expected_dim)

# Local LLM
llm_ctx = 4096
llm_gpu_layers = -1 if GPU_ONLY else 0  # ✅ use all GPU layers if GPU_ONLY
print(f"✅ Initializing LLaMA on {'GPU' if llm_gpu_layers != 0 else 'CPU'} …")
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
        text = parse_pdf(path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(text)

        print(f"📄 PDF split into {len(chunks)} chunks. Adding to vector DB (embeddings on {device}) …")
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

@app.route("/toggle", methods=["POST"])
def toggle():
    global OFFLINE_ONLY
    OFFLINE_ONLY = request.json["offline"]
    return jsonify({"status": "ok", "offline_only": OFFLINE_ONLY})

@app.route("/toggle_gpu", methods=["POST"])
def toggle_gpu():
    global GPU_ONLY, llm_gpu_layers, llm, embed_model, device
    GPU_ONLY = request.json["gpu_only"]
    print(f"🔄 Toggling GPU_ONLY to {GPU_ONLY}")

    # re-init device & embed_model
    device = "cuda" if torch.cuda.is_available() and GPU_ONLY else "cpu"
    embed_model = SentenceTransformer(embed_model_name, device=device)

    # re-init llm
    llm_gpu_layers = -1 if GPU_ONLY else 0
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=llm_gpu_layers,
        n_ctx=llm_ctx
    )

    return jsonify({"status": "ok", "gpu_only": GPU_ONLY})

@app.route("/status", methods=["GET"])
def status():
    try:
        count = len(collection.get(ids=None)["ids"])
    except:
        count = "unknown"
    return jsonify({
        "embedding_model": embed_model_name,
        "embedding_device": device,
        "embedding_dim": expected_dim,
        "llm_model": os.path.basename(MODEL_PATH),
        "llm_gpu_layers": llm_gpu_layers,
        "llm_context_length": llm_ctx,
        "offline_only": OFFLINE_ONLY,
        "gpu_only": GPU_ONLY,
        "collection_documents": count
    })

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
