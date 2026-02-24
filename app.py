from flask import Flask, request, render_template, jsonify
import os
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
from PIL import Image
import io
import google.generativeai as genai
import platform

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
    """Extract images from PDF and perform OCR with proper error handling."""
    doc = fitz.open(pdf_path)
    image_texts = []
    total_images = 0
    successful_ocr = 0
    
    try:
        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            images = page.get_images(full=True)
            total_images += len(images)
            
            for img in images:
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Perform OCR
                    text = pytesseract.image_to_string(image)
                    if text.strip():  # Only add non-empty text
                        image_texts.append(text)
                        successful_ocr += 1
                except Exception as img_error:
                    print(f"  ⚠ OCR failed for image on page {page_number + 1}: {str(img_error)}")
                    continue
        
        print(f"  📊 OCR Stats: {total_images} images found, {successful_ocr} successfully processed")
        if total_images > 0 and successful_ocr == 0:
            print("  ⚠ WARNING: No text extracted from images. Check Tesseract installation.")
        
        return {"text": "\n".join(image_texts), "images_found": total_images, "images_processed": successful_ocr}
    except Exception as e:
        print(f"  ❌ OCR Error: {str(e)}")
        return {"text": "", "images_found": 0, "images_processed": 0, "error": str(e)}

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

    # Extract text from PDF
    print("  📄 Extracting text from PDF...")
    text = parse_pdf_text(path)
    text_length = len(text.strip())
    print(f"  ✓ Extracted {text_length} characters of text")
    
    # Extract images and perform OCR
    print("  🖼️ Extracting images and performing OCR...")
    ocr_result = extract_images_and_ocr(path)
    image_text = ocr_result.get("text", "")
    ocr_length = len(image_text.strip())
    print(f"  ✓ Extracted {ocr_length} characters from OCR")
    
    # Show preview of OCR text if any
    if ocr_length > 0:
        preview = image_text[:200].replace('\n', ' ').strip()
        print(f"  📋 OCR Preview: {preview}...")
    
    combined_text = text + "\n" + image_text

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(combined_text)

    print(f"PDF split into {len(chunks)} chunks. Adding to vector DB...")
    for idx, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Processing chunks"):
        # Store in Vector Database
        collection.add(
            documents=[chunk],
            embeddings=[embed_text(chunk)],
            ids=[f"{file.filename}_{idx}"]
        )
    
    print(f"Upload complete: {file.filename}")
    
    # Prepare detailed response
    response_data = {
        "status": "success",
        "message": "PDF uploaded and processed",
        "mode": "offline" if OFFLINE_ONLY else "online",
        "stats": {
            "text_characters": text_length,
            "ocr_characters": ocr_length,
            "images_found": ocr_result.get("images_found", 0),
            "images_processed": ocr_result.get("images_processed", 0),
            "total_chunks": len(chunks)
        }
    }
    
    # Add warning if OCR failed
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
    prompt = f"""You are a helpful AI assistant. Answer the question based on the following document content.

IMPORTANT INSTRUCTIONS:
- Provide a COMPREHENSIVE and DETAILED answer using ALL relevant information from the document
- Include specific details, examples, features, and explanations found in the context
- For resume/CV questions: Extract names, titles, education, experience, etc. directly
- For document questions: Provide exact information without paraphrasing
- Use simple dashes (-) for bullet points when listing items
- Use numbers (1. 2. 3.) for sequential steps or procedures
- Write clear, informative paragraphs separated by blank lines
- Start with a direct answer, then provide thorough details
- Do NOT use special symbols like • or ** or other formatting marks
- Keep the language professional, clear, and informative

Document Content:
{context}

Question: {question}

Answer (provide a thorough, comprehensive, and detailed response using all relevant information):"""

    try:
        output = llm(prompt, max_tokens=2000, temperature=0.7, stop=["Question:", "Manual Content:"])
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
        
        # Check if we can fallback to online search
        has_gemini = bool(os.getenv("GEMINI_API_KEY"))
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        can_search_online = has_gemini or has_openai or SEARCH_ENGINE == "duckduckgo"
        
        if can_search_online:
            print("🌐 No documents available - Falling back to ONLINE SEARCH")
            print(f"   Using search engine: {SEARCH_ENGINE}")
            print("#"*60)
            
            online_response = search_online(query)
            fallback_note = "\nNote: No documents were uploaded, so this information was generated from online sources.\n\n"
            return jsonify({"response": fallback_note + online_response})
        else:
            print("#"*60 + "\n")
            return jsonify({"response": "No documents uploaded. Please upload documents to get started."})

    # Query the vector database
    print("🔍 Searching in uploaded documents...")
    print("⏳ Generating query embedding...")
    # Similarity Search
    results = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=20  # Increased to 20 to catch more potential matches
    )
    print("✅ Vector database search completed")

    # Get all results
    all_chunks = results.get("documents", [[]])[0] if results.get("documents") else []
    all_ids = results.get("ids", [[]])[0] if results.get("ids") else []
    all_distances = results.get('distances', [[]])[0] if results.get('distances') else []
    
    if not all_chunks:
        context = ""
        context_chunks = []
        distances = []
        best_file = "Unknown"
    else:
        # Group chunks by source file
        file_groups = {}
        for chunk_id, chunk, distance in zip(all_ids, all_chunks, all_distances):
            # Extract filename from chunk ID (format: filename_chunknum)
            filename = "_".join(chunk_id.split("_")[:-1])
            if filename not in file_groups:
                file_groups[filename] = []
            file_groups[filename].append({
                'id': chunk_id,
                'chunk': chunk,
                'distance': distance
            })
        
        # Find the file with the best (lowest) average distance in top results
        file_scores = {}
        for filename, items in file_groups.items():
            # Use average of top 3 distances from this file
            top_distances = sorted([item['distance'] for item in items])[:3]
            file_scores[filename] = sum(top_distances) / len(top_distances)
        
        # Get the best matching file
        best_file = min(file_scores, key=file_scores.get)
        best_file_score = file_scores[best_file]
        
        print(f"\n📁 File relevance scores (lower = better):")
        for filename, score in sorted(file_scores.items(), key=lambda x: x[1])[:3]:
            print(f"   • {filename}: {score:.4f}")
        print(f"\n✅ Selected file: {best_file}")
        
        # Use only chunks from the best matching file
        selected_items = sorted(file_groups[best_file], key=lambda x: x['distance'])[:5]
        context_chunks = [item['chunk'] for item in selected_items]
        distances = [item['distance'] for item in selected_items]
        context = "\n\n---\n\n".join(context_chunks)

    # Check relevance
    is_relevant = False
    
    print("\n📊 Analyzing relevance of search results...")
    print(f"   Found {len(context_chunks)} chunks from selected file")
    
    # Show previews of top 5 matches with ALL scores
    if context_chunks and distances:
        print(f"   Top {len(context_chunks)} matches from {best_file}:")
        for i in range(len(context_chunks)):
            preview = context_chunks[i][:150].replace('\n', ' ')
            print(f"   [{i+1}] Distance: {distances[i]:.4f} | Preview: {preview}...")
    
    if distances and len(distances) > 0:
        best_distance = min(distances)
        # Further relaxed threshold from 1.2 to 1.5
        # Semantic search can have high distances for factual queries
        # e.g., "What is X's profession?" vs "X Computer Engineer" = high distance
        is_relevant = best_distance < 1.5
        print(f"   Best match distance: {best_distance:.4f}")
        print(f"   Relevance threshold: 1.5 (very relaxed for better recall)")
        print(f"   Result: {'✅ RELEVANT' if is_relevant else '❌ NOT RELEVANT (try rephrasing)'}")

    # If not relevant in offline mode
    if not context or not is_relevant:
        print("\n❌ No relevant information found in uploaded documents")
        if distances and len(distances) > 0:
            print(f"   Closest match was {best_distance:.4f} (threshold: 1.5)")
        
        # Check if we can fallback to online search
        has_gemini = bool(os.getenv("GEMINI_API_KEY"))
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        can_search_online = has_gemini or has_openai or SEARCH_ENGINE == "duckduckgo"
        
        if can_search_online:
            print("🌐 No relevant documents found - Falling back to ONLINE SEARCH")
            print(f"   Using search engine: {SEARCH_ENGINE}")
            print("#"*60)
            
            online_response = search_online(query)
            fallback_note = "\nNote: This information was generated online because we couldn't find it in your uploaded documents.\n\n"
            return jsonify({"response": fallback_note + online_response})
        else:
            print("   • Try using keywords directly (e.g., use 'engineer' instead of 'profession')")
            print("   • Try asking differently (e.g., 'Tell me about X' instead of 'What is X')")
            print("   • The semantic embedding might not capture the relationship")
            print("🔒 Current Mode: OFFLINE - Cannot search online (no API keys configured)")
            print("#"*60 + "\n")
            return jsonify({"response": "No relevant information found in uploaded documents. Try rephrasing your question with more specific keywords from the document."})

    # Generate answer from local documents
    print("\n" + "*"*60)
    print(f"✅ Using information from: {best_file}")
    print("*"*60)
    print("🤖 Generating answer using local LLaMA model...")
    response = generate_answer(context, query)
    print("✅ Answer generated successfully from local documents")
    print("#"*60 + "\n")
    
    # Add clear indication of source file
    source_indicator = f"Source: {best_file}\n\n"
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
