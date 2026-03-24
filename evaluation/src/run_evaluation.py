#!/usr/bin/env python3
"""
Promply-V2 Evaluation Framework
Offline OCR accuracy measurement and reporting

Scans PDFs in uploads/, extracts text from native layers + scanned images,
runs Tesseract OCR, calculates metrics, and generates examiner reports.
"""

import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageFilter, ImageOps
import pytesseract
from pdfminer.high_level import extract_text
import fitz  # PyMuPDF


def configure_tesseract_for_windows():
    """Auto-detect and configure Tesseract on Windows."""
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in candidates:
        if os.path.exists(path):
            pytesseract.pytesseract.pytesseract_cmd = path
            return True
    raise FileNotFoundError(
        "Tesseract not found in expected locations. "
        "Download from: https://github.com/UB-Mannheim/tesseract/wiki"
    )


def preprocess_image_for_ocr(image_path):
    """
    Preprocess scanned image for optimal OCR recognition.
    
    Steps:
    1. Grayscale conversion
    2. Median denoise
    3. Autocontrast
    4. Binary threshold
    5. Mode validation
    """
    try:
        img = Image.open(image_path).convert('L')  # Grayscale
        img = img.filter(ImageFilter.MedianFilter(size=3))  # Denoise
        img = ImageOps.autocontrast(img, cutoff=1)  # Boost contrast
        threshold = 150
        img = img.point(lambda x: 0 if x < threshold else 255, '1')  # Binary
        if img.mode != 'L':
            img = img.convert('L')
        return img
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None


def extract_ocr_from_pdf_images(pdf_path, output_temp_dir):
    """Extract OCR text from scanned images in PDF."""
    ocr_text_list = []
    images_found = 0
    images_processed = 0
    
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            image_list = page.get_images()
            if image_list:
                images_found += len(image_list)
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            temp_path = os.path.join(output_temp_dir, f"page_{page_num}_img_{img_index}.png")
                            pix.save(temp_path)
                            
                            preprocessed = preprocess_image_for_ocr(temp_path)
                            if preprocessed:
                                text = pytesseract.image_to_string(preprocessed, lang='eng', config='--psm 6')
                                if text.strip():
                                    ocr_text_list.append(text)
                                    images_processed += 1
                            
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                    except Exception as e:
                        pass
        doc.close()
    except Exception as e:
        pass
    
    return "\n".join(ocr_text_list), images_found, images_processed


def levenshtein_distance(s1, s2):
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def word_error_rate(expected, extracted):
    """Calculate WER between expected and extracted text."""
    expected_words = expected.split()
    extracted_words = extracted.split()
    
    if len(expected_words) == 0:
        return None
    
    distance = levenshtein_distance(" ".join(expected_words), " ".join(extracted_words))
    wer = (distance / len(expected_words)) * 100
    return round(wer, 2)


def character_error_rate(expected, extracted):
    """Calculate CER between expected and extracted text."""
    if len(expected) == 0:
        return None
    
    distance = levenshtein_distance(expected, extracted)
    cer = (distance / len(expected)) * 100
    return round(cer, 2)


def evaluate_documents(uploads_dir, ground_truth_data=None):
    """Process all PDFs and calculate metrics."""
    metrics_list = []
    doc_id = 0
    
    pdf_files = sorted([f for f in os.listdir(uploads_dir) if f.lower().endswith('.pdf')])
    
    for pdf_file in pdf_files:
        doc_id += 1
        doc_id_str = f"D{doc_id:02d}"
        pdf_path = os.path.join(uploads_dir, pdf_file)
        start_time = time.time()
        
        try:
            # Native text extraction
            try:
                native_text = extract_text(pdf_path)
            except:
                native_text = ""
            
            # OCR from scanned images
            temp_dir = os.path.join(uploads_dir, '.temp_ocr')
            os.makedirs(temp_dir, exist_ok=True)
            ocr_text, images_found, images_processed = extract_ocr_from_pdf_images(pdf_path, temp_dir)
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass
            
            # Count pages
            doc = fitz.open(pdf_path)
            pages = len(doc)
            doc.close()
            
            # Calculate metrics
            native_chars = len(native_text) if native_text else 0
            ocr_chars = len(ocr_text) if ocr_text else 0
            total_chars = native_chars + ocr_chars
            
            ocr_success_rate = (images_processed / images_found * 100) if images_found > 0 else None
            ocr_contribution = (ocr_chars / total_chars * 100) if total_chars > 0 else 0.0
            
            processing_time = round(time.time() - start_time, 2)
            
            # Ground truth accuracy (if provided)
            wer = None
            cer = None
            if ground_truth_data and pdf_file in ground_truth_data:
                expected_text = ground_truth_data[pdf_file]
                extracted_combined = native_text + " " + ocr_text
                wer = word_error_rate(expected_text, extracted_combined)
                cer = character_error_rate(expected_text, extracted_combined)
            
            metrics = {
                'doc_id': doc_id_str,
                'file_name': pdf_file,
                'pages': pages,
                'images_found': images_found,
                'images_processed': images_processed if images_found > 0 else 0,
                'ocr_success_rate_pct': round(ocr_success_rate, 2) if ocr_success_rate is not None else 'N/A',
                'native_chars': native_chars,
                'ocr_chars': ocr_chars,
                'total_chars': total_chars,
                'ocr_contribution_pct': round(ocr_contribution, 2),
                'wer_pct': wer,
                'cer_pct': cer,
                'processing_time_sec': processing_time,
                'native_text': native_text[:200],
                'ocr_text': ocr_text[:200],
            }
            
            metrics_list.append(metrics)
        
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            continue
    
    return metrics_list


def summarize(metrics_list):
    """Aggregate metrics across all documents."""
    if not metrics_list:
        return {}
    
    count = len(metrics_list)
    pages = sum(m['pages'] for m in metrics_list)
    images = sum(m['images_found'] for m in metrics_list)
    
    # Extraction coverage
    docs_with_text = sum(1 for m in metrics_list if m['total_chars'] > 0)
    extraction_coverage = round((docs_with_text / count) * 100, 2) if count else 0.0
    
    # OCR performance (only for docs with images)
    ocr_success_rates = [m['ocr_success_rate_pct'] for m in metrics_list 
                         if isinstance(m['ocr_success_rate_pct'], (int, float))]
    avg_ocr_success = round(sum(ocr_success_rates) / len(ocr_success_rates), 2) if ocr_success_rates else 0.0
    
    # Text contribution
    total_native = sum(m['native_chars'] for m in metrics_list)
    total_ocr = sum(m['ocr_chars'] for m in metrics_list)
    total = total_native + total_ocr
    ocr_contribution = round((total_ocr / total * 100), 2) if total > 0 else 0.0
    
    # CER/WER averages (if ground truth exists)
    cers = [m['cer_pct'] for m in metrics_list if m['cer_pct'] is not None]
    avg_cer = round(sum(cers) / len(cers), 2) if cers else None
    
    # Answer accuracy
    transcription_accuracy = round(max(0.0, 100.0 - avg_cer), 2) if avg_cer is not None else None
    
    # Answer confidence proxy (when no ground truth)
    proxy_ocr = avg_ocr_success if avg_ocr_success is not None else 85.0
    answer_confidence_proxy = round((0.45 * extraction_coverage) + (0.35 * proxy_ocr) + (0.20 * 70.0), 2)
    
    # Timing
    total_time = sum(m['processing_time_sec'] for m in metrics_list)
    avg_time = round(total_time / count, 2)
    
    return {
        'documents': count,
        'pages': pages,
        'images': images,
        'extraction_coverage_pct': extraction_coverage,
        'ocr_success_rate_pct': avg_ocr_success,
        'ocr_contribution_pct': ocr_contribution,
        'answer_accuracy_pct': transcription_accuracy,
        'answer_confidence_proxy_pct': answer_confidence_proxy,
        'total_processing_time_sec': round(total_time, 2),
        'avg_time_per_doc_sec': avg_time,
    }


def build_html_report(summary, metrics_list, output_file):
    """Generate HTML dashboard report."""
    html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Promply-V2 Evaluation Report</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --line: #dbe3ef;
      --blue: #2563eb;
      --teal: #0f766e;
      --amber: #b45309;
      --violet: #4338ca;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", Tahoma, sans-serif; background: var(--bg); color: var(--ink); }}
    .wrap {{ max-width: 1240px; margin: 28px auto 40px; padding: 0 18px; }}
    .hero {{ background: linear-gradient(120deg, #0b3b8f, #0f766e); color: #fff; padding: 24px; border-radius: 16px; box-shadow: var(--shadow); }}
    .hero h1 {{ margin: 0 0 6px; font-size: 1.8rem; }}
    .hero p {{ margin: 0; opacity: 0.95; }}
    .kpi-grid {{ margin-top: 16px; display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }}
    .kpi-card {{ background: #fff; color: var(--ink); border-radius: 12px; padding: 12px; border: 1px solid var(--line); }}
    .kpi-label {{ font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .kpi-value {{ margin-top: 2px; font-size: 1.2rem; font-weight: 700; }}
    .summary {{ margin-top: 14px; display: flex; gap: 16px; flex-wrap: wrap; color: #e2e8f0; font-size: 0.92rem; }}
    .section {{ margin-top: 18px; background: var(--panel); border: 1px solid var(--line); border-radius: 14px; box-shadow: var(--shadow); padding: 16px; }}
    .section h2 {{ margin: 0 0 8px; font-size: 1.08rem; }}
    .hint {{ margin: 0 0 12px; color: var(--muted); font-size: 0.92rem; }}
    .bar-chart {{ display: grid; gap: 8px; }}
    .bar-row {{ display: grid; grid-template-columns: 56px 1fr 76px; gap: 8px; align-items: center; }}
    .bar-label {{ font-weight: 700; color: var(--muted); }}
    .bar-track {{ height: 14px; border-radius: 999px; background: #e2e8f0; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 999px; }}
    .blue {{ background: var(--blue); }}
    .teal {{ background: var(--teal); }}
    .amber {{ background: var(--amber); }}
    .violet {{ background: var(--violet); }}
    .bar-value {{ text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }}
    .mix {{ border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }}
    .mix-track {{ display: flex; height: 26px; }}
    .mix-native {{ background: var(--violet); }}
    .mix-ocr {{ background: var(--teal); }}
    .mix-labels {{ display: flex; justify-content: space-between; gap: 12px; padding: 10px; font-size: 0.92rem; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }}
    th {{ font-size: 0.8rem; text-transform: uppercase; color: var(--muted); letter-spacing: 0.03em; }}
    tr:hover {{ background: #f9fafb; }}
  </style>
</head>
<body>
  <div class='wrap'>
    <div class='hero'>
      <h1>Promply-V2 Evaluation Report</h1>
      <p>OCR accuracy assessment — automatic metrics generation</p>
      <div class='kpi-grid'>
        <div class='kpi-card'>
          <div class='kpi-label'>Documents</div>
          <div class='kpi-value'>{summary['documents']}</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-label'>Pages</div>
          <div class='kpi-value'>{summary['pages']}</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-label'>Images</div>
          <div class='kpi-value'>{summary['images']}</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-label'>Extraction Coverage %</div>
          <div class='kpi-value'>{summary['extraction_coverage_pct']}</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-label'>OCR Success %</div>
          <div class='kpi-value'>{summary['ocr_success_rate_pct']}</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-label'>OCR Contribution %</div>
          <div class='kpi-value'>{summary['ocr_contribution_pct']}</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-label'>Answer Accuracy %</div>
          <div class='kpi-value'>{summary['answer_accuracy_pct'] or 'N/A'}</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-label'>Confidence Proxy %</div>
          <div class='kpi-value'>{summary['answer_confidence_proxy_pct']}</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-label'>Total Time</div>
          <div class='kpi-value'>{summary['total_processing_time_sec']}s</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-label'>Avg Time/Doc</div>
          <div class='kpi-value'>{summary['avg_time_per_doc_sec']}s</div>
        </div>
      </div>
    </div>

    <section class='section'>
      <h2>Page Distribution</h2>
      <div class='bar-chart'>
"""
    
    # Top 10 docs by pages
    top_docs_pages = sorted(metrics_list, key=lambda x: x['pages'], reverse=True)[:10]
    max_pages = max([d['pages'] for d in top_docs_pages]) or 1
    
    for doc in top_docs_pages:
        pct = (doc['pages'] / max_pages * 100) if max_pages else 0
        html += f"""        <div class='bar-row'>
          <div class='bar-label'>{doc['doc_id']}</div>
          <div class='bar-track'><div class='bar-fill blue' style='width: {pct}%'></div></div>
          <div class='bar-value'>{doc['pages']}</div>
        </div>
"""
    
    html += f"""      </div>
    </section>

    <section class='section'>
      <h2>OCR Success Rate</h2>
      <div class='bar-chart'>
"""
    
    # Top 10 docs by OCR success
    ocr_success_docs = [d for d in metrics_list if isinstance(d['ocr_success_rate_pct'], (int, float))]
    top_docs_ocr = sorted(ocr_success_docs, key=lambda x: x['ocr_success_rate_pct'], reverse=True)[:10]
    
    for doc in top_docs_ocr:
        val = doc['ocr_success_rate_pct']
        html += f"""        <div class='bar-row'>
          <div class='bar-label'>{doc['doc_id']}</div>
          <div class='bar-track'><div class='bar-fill teal' style='width: {val}%'></div></div>
          <div class='bar-value'>{val}%</div>
        </div>
"""
    
    # Native vs OCR mix
    native_pct = 100 - summary['ocr_contribution_pct']
    ocr_pct = summary['ocr_contribution_pct']
    
    html += f"""      </div>
    </section>

    <section class='section'>
      <h2>Text Source Mix</h2>
      <div class='mix'>
        <div class='mix-track'>
          <div class='mix-native' style='flex: {native_pct}'></div>
          <div class='mix-ocr' style='flex: {ocr_pct}'></div>
        </div>
        <div class='mix-labels'>
          <span>Native PDF: {native_pct}%</span>
          <span>OCR: {ocr_pct}%</span>
        </div>
      </div>
    </section>

    <section class='section'>
      <h2>OCR Contribution %</h2>
      <div class='bar-chart'>
"""
    
    # Top 10 docs by OCR contribution
    top_docs_contrib = sorted(metrics_list, key=lambda x: x['ocr_contribution_pct'], reverse=True)[:10]
    max_contrib = max([d['ocr_contribution_pct'] for d in top_docs_contrib]) or 1
    
    for doc in top_docs_contrib:
        val = doc['ocr_contribution_pct']
        pct = (val / max_contrib * 100) if max_contrib else 0
        html += f"""        <div class='bar-row'>
          <div class='bar-label'>{doc['doc_id']}</div>
          <div class='bar-track'><div class='bar-fill amber' style='width: {pct}%'></div></div>
          <div class='bar-value'>{val}%</div>
        </div>
"""
    
    html += """      </div>
    </section>

    <section class='section'>
      <h2>Processing Time (s)</h2>
      <div class='bar-chart'>
"""
    
    # Top 10 docs by processing time
    top_docs_time = sorted(metrics_list, key=lambda x: x['processing_time_sec'], reverse=True)[:10]
    max_time = max([d['processing_time_sec'] for d in top_docs_time]) or 1
    
    for doc in top_docs_time:
        val = doc['processing_time_sec']
        pct = (val / max_time * 100) if max_time else 0
        html += f"""        <div class='bar-row'>
          <div class='bar-label'>{doc['doc_id']}</div>
          <div class='bar-track'><div class='bar-fill violet' style='width: {pct}%'></div></div>
          <div class='bar-value'>{val}s</div>
        </div>
"""
    
    html += """      </div>
    </section>

    <section class='section'>
      <h2>Document Metrics</h2>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>File Name</th>
            <th>Pages</th>
            <th>Images</th>
            <th>Processed</th>
            <th>OCR Success %</th>
            <th>OCR Contribution %</th>
            <th>Time (s)</th>
            <th>WER %</th>
            <th>CER %</th>
          </tr>
        </thead>
        <tbody>
"""
    
    for doc in metrics_list:
        wer_val = doc['wer_pct'] if doc['wer_pct'] is not None else 'N/A'
        cer_val = doc['cer_pct'] if doc['cer_pct'] is not None else 'N/A'
        ocr_success = doc['ocr_success_rate_pct'] if isinstance(doc['ocr_success_rate_pct'], (int, float)) else 'N/A'
        
        html += f"""          <tr>
            <td>{doc['doc_id']}</td>
            <td title='{doc['file_name']}'>{doc['file_name']}</td>
            <td>{doc['pages']}</td>
            <td>{doc['images_found']}</td>
            <td>{doc['images_processed']}</td>
            <td>{ocr_success}</td>
            <td>{doc['ocr_contribution_pct']}</td>
            <td>{doc['processing_time_sec']}</td>
            <td>{wer_val}</td>
            <td>{cer_val}</td>
          </tr>
"""
    
    html += """        </tbody>
      </table>
    </section>

    <section class='section'>
      <h2>Document Legend</h2>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>File Name</th>
          </tr>
        </thead>
        <tbody>
"""
    
    for doc in metrics_list:
        html += f"""          <tr>
            <td>{doc['doc_id']}</td>
            <td>{doc['file_name']}</td>
          </tr>
"""
    
    html += """        </tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)


def build_markdown_report(summary, metrics_list, output_file):
    """Generate Markdown report."""
    md = f"""# Promply-V2 Evaluation Report

## Summary

- **Documents Processed:** {summary['documents']}
- **Total Pages:** {summary['pages']}
- **Extraction Coverage:** {summary['extraction_coverage_pct']}%
- **OCR Success Rate:** {summary['ocr_success_rate_pct']}%
- **OCR Contribution:** {summary['ocr_contribution_pct']}%
- **Answer Accuracy:** {summary['answer_accuracy_pct'] or 'N/A'}%
- **Confidence Proxy:** {summary['answer_confidence_proxy_pct']}%
- **Total Time:** {summary['total_processing_time_sec']}s

## Document Metrics

| ID | File | Pages | Images | Processed | OCR Success % | Contribution % | Time (s) | WER % | CER % |
|----|------|-------|--------|-----------|---------------|-----------------|---------|-------|-------|
"""
    
    for doc in metrics_list:
        wer_val = doc['wer_pct'] if doc['wer_pct'] is not None else 'N/A'
        cer_val = doc['cer_pct'] if doc['cer_pct'] is not None else 'N/A'
        ocr_success = doc['ocr_success_rate_pct'] if isinstance(doc['ocr_success_rate_pct'], (int, float)) else 'N/A'
        
        md += f"| {doc['doc_id']} | {doc['file_name']} | {doc['pages']} | {doc['images_found']} | {doc['images_processed']} | {ocr_success} | {doc['ocr_contribution_pct']} | {doc['processing_time_sec']} | {wer_val} | {cer_val} |\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md)


def write_outputs(summary, metrics_list, output_dir):
    """Write all output files (latest-only by default)."""
    os.makedirs(output_dir, exist_ok=True)
    
    # HTML report
    html_latest = os.path.join(output_dir, 'examiner_report_latest.html')
    build_html_report(summary, metrics_list, html_latest)
    print(f"✓ HTML Report: {html_latest}")
    
    # Markdown report
    md_latest = os.path.join(output_dir, 'examiner_report_latest.md')
    build_markdown_report(summary, metrics_list, md_latest)
    print(f"✓ Markdown Report: {md_latest}")
    
    # CSV metrics
    csv_latest = os.path.join(output_dir, 'metrics_latest.csv')
    with open(csv_latest, 'w', newline='', encoding='utf-8') as f:
        if metrics_list:
            writer = csv.DictWriter(f, fieldnames=metrics_list[0].keys())
            writer.writeheader()
            writer.writerows(metrics_list)
    print(f"✓ Metrics CSV: {csv_latest}")
    
    # JSON summary
    json_latest = os.path.join(output_dir, 'summary_latest.json')
    with open(json_latest, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary JSON: {json_latest}")


def load_ground_truth(csv_path):
    """Load ground truth data from CSV."""
    data = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'file_name' in row and 'expected_text' in row:
                    data[row['file_name']] = row['expected_text']
    except Exception as e:
        print(f"Warning: Could not load ground truth: {e}")
    return data


def main():
    parser = argparse.ArgumentParser(description='Promply-V2 OCR Evaluation Framework')
    parser.add_argument('--ground-truth', type=str, help='Path to ground truth CSV file')
    parser.add_argument('--archive', action='store_true', help='Archive timestamped results')
    args = parser.parse_args()
    
    # Configure Tesseract
    configure_tesseract_for_windows()
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    uploads_dir = project_root / 'uploads'
    output_dir = project_root / 'evaluation' / 'output'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth if provided
    ground_truth_data = {}
    if args.ground_truth:
        ground_truth_data = load_ground_truth(args.ground_truth)
    
    # Evaluate documents
    print("Scanning PDFs in uploads/...")
    metrics_list = evaluate_documents(str(uploads_dir), ground_truth_data)
    
    if not metrics_list:
        print("No PDFs found in uploads/")
        return
    
    print(f"Processed {len(metrics_list)} documents")
    
    # Summarize
    summary = summarize(metrics_list)
    
    # Write outputs
    write_outputs(summary, metrics_list, str(output_dir))
    
    print("\n✓ Evaluation complete!")
    print(f"Open: {output_dir}/examiner_report_latest.html")


if __name__ == '__main__':
    main()
