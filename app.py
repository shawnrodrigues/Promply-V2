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
import google.generativeai as genai
from openai import OpenAI
from duckduckgo_search import DDGS

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

# Always use GPU for embedding
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available - GPU required for this app.")

device = "cuda"
print(f"SentenceTransformer running on: {device}")
embed_model = SentenceTransformer("all-mpnet-base-v2", device=device)

OFFLINE_ONLY = True
SEARCH_ENGINE = "duckduckgo"

print("=" * 60)
print("PROMPTLY STARTING...")
print("=" * 60)
print(f"Initial Mode: {'OFFLINE MODE' if OFFLINE_ONLY else 'ONLINE MODE'}")
print("=" * 60)

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
        for img in images:
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

print("Initializing LLaMA with GPU...")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=4096
)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("pdf")
    if not file:
        return jsonify({"status": "error", "message": "No file uploaded"})
    
    print(f"Processing upload: {file.filename}")
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

    print(f"PDF split into {len(chunks)} chunks. Adding to vector DB...")
    for idx, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Processing chunks"):
        collection.add(
            documents=[chunk],
            embeddings=[embed_text(chunk)],
            ids=[f"{file.filename}_{idx}"]
        )
    
    print(f"Upload complete: {file.filename}")
    return jsonify({
        "status": "success",
        "message": "PDF uploaded and processed",
        "mode": "offline" if OFFLINE_ONLY else "online"
    })

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
    prompt = f"""You are a helpful AI assistant. Answer the question based on the following manual content.

IMPORTANT: Structure your response clearly using:
• Use bullet points (•) for listing features, items, or key points
• Use numbered lists (1., 2., 3.) for sequential steps or procedures  
• Write clear paragraphs separated by blank lines
• Start with a direct answer to the question
• Include relevant details and examples from the context
• End with any important notes or warnings if applicable

Manual Content:
{context}

Question: {question}

Answer (provide a well-structured, detailed response):"""

    try:
        output = llm(prompt, max_tokens=1000, temperature=0.7, stop=["Question:", "Manual Content:"])
        return format_response(output['choices'][0]['text'])
    except Exception as e:
        return f"Error generating answer: {e}"

def search_online(query):
    print(f"Searching online using {SEARCH_ENGINE.upper()}")

    if SEARCH_ENGINE == "duckduckgo":
        try:
            print("Using DuckDuckGo Search (free)")
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=5))

            if not results:
                return "No search results found."

            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"Source {i}: {r.get('title', 'No title')}\n{r.get('body', '')}\nURL: {r.get('href', '')}"
                )

            context = "\n\n---\n\n".join(formatted)

            prompt = f"""You are a helpful AI assistant. Use the following online search results to answer the question.

IMPORTANT: Structure your response clearly using:
• Use bullet points (•) for listing features, items, or key points
• Use numbered lists (1., 2., 3.) for sequential steps or procedures
• Write clear paragraphs separated by blank lines
• Start with a direct, comprehensive answer
• Include relevant details from the search results
• Cite sources when mentioning specific information (e.g., "According to Source 1...")
• End with a summary or conclusion if appropriate

Search Results:
{context}

Question: {query}

Answer (provide a well-structured, detailed response):"""

            try:
                output = llm(prompt, max_tokens=1000, temperature=0.7, stop=["Question:", "Search Results:"])
                answer = format_response(output['choices'][0]['text'])
                
                sources = "\n\n" + "=" * 50 + "\n\nSources:\n"
                for i, r in enumerate(results, 1):
                    sources += f"  {i}. {r.get('title', 'No title')}\n     {r.get('href', '')}\n"
                
                print("DuckDuckGo search completed")
                return answer + sources
            except Exception as e:
                print(f"Error generating answer: {e}")
                return f"Error processing search results: {e}"
                
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
            return f"DuckDuckGo search failed: {e}"

    elif SEARCH_ENGINE == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            return "Gemini API key not found in .env file."
        
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            enhanced_query = f"""Answer the following question in a well-structured format:

• Use bullet points for lists
• Use numbered steps for procedures
• Write clear paragraphs
• Include examples when relevant

Question: {query}"""
            
            response = model.generate_content(enhanced_query)
            return format_response(response.text)
        except Exception as e:
            return f"Gemini API error: {e}"

    elif SEARCH_ENGINE == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return "OpenAI API key not found in .env file."
        
        try:
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Always structure your responses with bullet points, numbered lists, and clear paragraphs for better readability."},
                    {"role": "user", "content": query}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return format_response(response.choices[0].message.content)
        except Exception as e:
            return f"OpenAI API error: {e}"

    return "Search engine not configured."

@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("query", "")
    if not query:
        return jsonify({"response": "No query provided."})
    
    print(f"Processing query: '{query[:50]}...'")

    # Check if we have documents
    try:
        all_docs = collection.get(limit=1)
        has_documents = len(all_docs.get('ids', [])) > 0
    except:
        has_documents = False

    # If no documents and offline mode
    if not has_documents:
        if OFFLINE_ONLY:
            return jsonify({"response": "No documents uploaded. Please upload documents to get started."})
        else:
            response = search_online(query)
            return jsonify({"response": response})

    # Query the vector database
    results = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=5
    )

    context = "\n".join(results.get("documents", [[]])[0]) if results.get("documents") else ""

    # Check relevance
    distances = results.get('distances', [[]])[0] if results.get('distances') else []
    is_relevant = False
    
    if distances and len(distances) > 0:
        best_distance = min(distances)
        is_relevant = best_distance < 0.8
        print(f"Best match distance: {best_distance:.3f} ({'relevant' if is_relevant else 'not relevant'})")

    # If not relevant and online mode, search online
    if (not context or not is_relevant) and not OFFLINE_ONLY:
        print("No relevant local context, searching online...")
        response = search_online(query)
        return jsonify({"response": response})
    
    # If not relevant and offline mode
    if not context or not is_relevant:
        return jsonify({"response": "No relevant information found in uploaded documents."})

    # Generate answer from local documents
    print("Using local document context")
    response = generate_answer(context, query)
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
        "search_engine": SEARCH_ENGINE
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

if __name__ == "__main__":
    print("\nStarting Flask server...")
    print("Server will be available at: http://localhost:6969")
    print("Ready to process documents and queries!")
    print("=" * 60 + "\n")
    app.run(debug=True, port=6969)

# Made with love by CoCo
