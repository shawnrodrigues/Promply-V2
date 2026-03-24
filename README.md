# 🚀 Promply-V2 — Installation & Setup Guide

Promply-V2 is an AI-powered document search and OCR application, supporting NVIDIA GPU acceleration, vector search, and the latest LLM APIs.

---

## ✅ Prerequisites

Before installing, ensure the following are installed on your system:

| Requirement | Purpose |
|------------|---------|
| **Python 3.11** | Required runtime |
| **NVIDIA GPU + CUDA Toolkit (12.X recommended)** | GPU acceleration |
| **Visual Studio Build Tools** | For building Python dependencies |
| **Tesseract OCR** | Extract text from PDFs/images |

Optional model downloads:   
📁 [Models (Google Drive Link 1)](https://drive.google.com/drive/folders/1HOnDhHKFnIaD96kj-8zbLoGCrBK4Pklf?usp=drive_link)

---

## ⚙️ Installation Steps

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/shawnrodrigues/Promply-V2.git
cd Promply-V2
```

### 2️⃣ Create & Activate Virtual Environment

```bash
py -3.11 -m venv myenv
myenv\Scripts\activate
```

### 3️⃣ Install PyTorch with CUDA Support

```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If GPU installation fails, install CPU-only:

```bash
pip install torch torchvision torchaudio
```

### 4️⃣ Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Folder Creation (Required)

Create folders if they don’t exist:

```bash
mkdir uploads models vector_store
mkdir uploads\images
```

---

## 🖼️ Tesseract OCR Setup

### Install Tesseract
Download from:  
🔗 https://github.com/UB-Mannheim/tesseract/wiki

### Add to PATH (Windows PowerShell)

```bash
setx PATH "%PATH%;C:\Program Files\Tesseract-OCR"
```

### Verify Installation

```bash
tesseract --version
```

---

## 🔑 Environment Variables

Create a file named `.env` in the project root:

```ini
# Google Search
GOOGLE_API_KEY=
GOOGLE_CX=

# Gemini API
GEMINI_API_KEY=

# OpenAI
OPENAI_API_KEY=
```

---

## ▶️ Run Promply-V2

```bash
python app.py
```

Then open:

```
http://localhost:5000
```

---

## � Evaluation & Metrics (OCR Quality Assessment)

Promply-V2 includes an offline evaluation framework to measure OCR accuracy and extraction quality.

### Quick Start Evaluation

```bash
python evaluation/run_evaluation.py
```

This scans all PDFs in `uploads/` and generates an **examiner-ready dashboard**:

```bash
start evaluation/output/examiner_report_latest.html
```

### What the Evaluation Shows

| Metric | Description |
|--------|-------------|
| **Documents** | Total PDFs processed |
| **Pages** | Total pages scanned |
| **Scan Coverage %** | Percentage of documents with scanned (image) content |
| **Extraction Coverage %** | Percentage of documents with any extractable text |
| **OCR Success Rate %** | Percentage of scanned images successfully converted to text |
| **OCR Contribution %** | How much of extracted text came from OCR vs. native PDF |
| **Answer Accuracy %** | Character Error Rate (when ground truth is provided) |
| **Processing Time** | Time per document in seconds |

### Enable True Accuracy Metrics (WER/CER)

1. Create `evaluation/input/ground_truth.csv`:

```csv
file_name,expected_text
document_01.pdf,"Exact expected transcription here"
document_02.pdf,"Another document's expected text"
```

2. Run with ground truth:

```bash
python evaluation/run_evaluation.py --ground-truth evaluation/input/ground_truth.csv
```

3. View results:

```bash
start evaluation/output/examiner_report_latest.html
```

### Archive Previous Runs

By default, only the latest reports are kept. To also archive timestamped copies:

```bash
python evaluation/run_evaluation.py --archive
```

### Tools & Technologies Used

The evaluation framework leverages:

- **pdfminer.six** — PDF text extraction
- **PyMuPDF (fitz)** — Embedded image extraction from PDFs
- **Pillow** — Image preprocessing (grayscale, denoise, contrast, threshold)
- **Tesseract OCR** — Local optical character recognition
- **Levenshtein Distance** — Accuracy calculation (WER/CER metrics)

All processing is **local and offline** — no external APIs or paid services required.

---

## �📸 UI Preview

### Project Structure
```
PROMPLY-V2
├── data/                              # Data storage
├── Images/                            # UI screenshots
├── interface/                         # Next.js frontend
├── models/                            # LLM models (Llama, Mistral)
├── uploads/                           # User-uploaded PDFs
│   └── images/                        # Extracted images
├── vector_store/                      # ChromaDB embeddings
├── evaluation/                        # OCR Evaluation Kit ⭐ NEW
│   ├── src/
│   │   └── run_evaluation.py          # Main evaluation engine
│   ├── run_evaluation.py              # Launcher script
│   ├── README.md                      # Evaluation docs
│   ├── EXAMINER_PACK.md               # Presentation guide
│   ├── input/
│   │   └── ground_truth_template.csv  # Ground truth labels (optional)
│   └── output/
│       ├── examiner_report_latest.html  # Visual dashboard
│       ├── examiner_report_latest.md    # Markdown report
│       ├── metrics_latest.csv           # Raw metrics table
│       └── summary_latest.json          # KPI summary
├── app.py                             # Flask backend
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

![Contents](Images/Contents.png)

Landing Page

![landing Page](<Images/Landing UI.png>)

Uploading UI

![Uploading Page](<Images/Upload UI.png>)

Progress Bar UI

![Progress UI](<Images/UI Progress.png>)

Chat-Box UI

![Chat-Box UI](<Images/Chat-Box UI.png>)

---

## 🔁 Environment Reset (If Required)

```bash
deactivate
Remove-Item -Recurse -Force myenv
py -3.11 -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
```

---

## ✅ You’re Ready!

You now have Promply-V2 successfully running! 🚀  
Have fun exploring the power of AI search & OCR! 😄
