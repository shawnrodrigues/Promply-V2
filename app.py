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
SEARCH_ENGINE = "gemini"  # Changed from duckduckgo to gemini for better reliability

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
• ONLY use information from the provided search results
• DO NOT include information about unrelated topics
• If the search results mention irrelevant topics, IGNORE them completely
• Focus exclusively on answering the specific question asked
• Structure your response clearly using bullet points and numbered lists
• Start with a direct, comprehensive answer
• Include relevant details from the search results
• Cite sources when mentioning specific information (e.g., "According to Source 1...")

Search Results:
{context}

Question: {query}

Answer (provide ONLY information relevant to the question):"""

            try:
                output = llm(prompt, max_tokens=1000, temperature=0.7, stop=["Question:", "Search Results:"])
                answer = format_response(output['choices'][0]['text'])
                
                # Clear indication that this is NOT from uploaded documents
                disclaimer = "\n\n" + "⚠️ " + "="*50 + "\n"
                disclaimer += "📌 IMPORTANT: This information was NOT found in your uploaded documents.\n"
                disclaimer += "🌐 This answer is based on online web search results.\n"
                disclaimer += "="*50 + "\n\n"
                
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
            
            enhanced_query = f"""Answer the following question in a well-structured format:

• Use bullet points for lists
• Use numbered steps for procedures
• Write clear paragraphs
• Include examples when relevant

Question: {query}"""
            
            response = model.generate_content(enhanced_query)
            print("✅ Gemini AI response received")
            print("="*60)
            print("🎉 Gemini search completed successfully!")
            print("="*60 + "\n")
            disclaimer = "\n\n" + "⚠️ " + "="*50 + "\n"
            disclaimer += "📍 IMPORTANT: This information was NOT found in your uploaded documents.\n"
            disclaimer += "🌐 This answer is generated using Gemini AI with web search capabilities.\n"
            disclaimer += "="*50 + "\n\n"
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
                    {"role": "system", "content": "You are a helpful assistant. Always structure your responses with bullet points, numbered lists, and clear paragraphs for better readability."},
                    {"role": "user", "content": query}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            print("✅ OpenAI response received")
            print("="*60)
            print("🎉 OpenAI search completed successfully!")
            print("="*60 + "\n")
            disclaimer = "\n\n" + "⚠️ " + "="*50 + "\n"
            disclaimer += "📍 IMPORTANT: This information was NOT found in your uploaded documents.\n"
            disclaimer += "🌐 This answer is generated using OpenAI GPT with general knowledge.\n"
            disclaimer += "="*50 + "\n\n"
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

    # If ONLINE mode is enabled, search the web directly
    if not OFFLINE_ONLY:
        print("🌐 ONLINE MODE: Searching the web directly...")
        print("#"*60)
        response = search_online(query)
        return jsonify({"response": response})

    # OFFLINE MODE: Check if we have documents
    try:
        all_docs = collection.get(limit=1)
        has_documents = len(all_docs.get('ids', [])) > 0
        print(f"📚 Documents in database: {'Yes' if has_documents else 'No'}")
    except:
        has_documents = False
        print("⚠️ Could not check document database")

    # If no documents and offline mode
    if not has_documents:
        print("❌ No documents uploaded and in OFFLINE mode")
        print("#"*60 + "\n")
        return jsonify({"response": "No documents uploaded. Please upload documents to get started."})

    # Query the vector database
    print("🔍 Searching in uploaded documents...")
    print("⏳ Generating query embedding...")
    results = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=5
    )
    print("✅ Vector database search completed")

    context = "\n".join(results.get("documents", [[]])[0]) if results.get("documents") else ""

    # Check relevance
    distances = results.get('distances', [[]])[0] if results.get('distances') else []
    is_relevant = False
    
    print("📊 Analyzing relevance of search results...")
    if distances and len(distances) > 0:
        best_distance = min(distances)
        is_relevant = best_distance < 0.8
        print(f"   Best match distance: {best_distance:.3f}")
        print(f"   Relevance threshold: 0.8")
        print(f"   Result: {'✅ RELEVANT' if is_relevant else '❌ NOT RELEVANT'}")

    # If not relevant in offline mode
    if not context or not is_relevant:
        print("\n❌ No relevant information found in uploaded documents")
        print("🔒 Current Mode: OFFLINE - Cannot search online")
        print("#"*60 + "\n")
        return jsonify({"response": "No relevant information found in uploaded documents."})

    # Generate answer from local documents
    print("\n" + "*"*60)
    print("✅ Using information from uploaded documents")
    print("*"*60)
    print("🤖 Generating answer using local LLaMA model...")
    response = generate_answer(context, query)
    print("✅ Answer generated successfully from local documents")
    print("#"*60 + "\n")
    
    # Add clear indication that this IS from uploaded documents
    source_indicator = "📚 Source: Your Uploaded Documents/Manual\n" + "="*50 + "\n\n"
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
