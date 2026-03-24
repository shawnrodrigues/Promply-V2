# Evaluation Kit (Examiner Ready)

This folder generates offline metrics and examiner-ready reports from your current project pipeline.

## What You Can Show to an Examiner

- OCR Success Rate by document
- OCR Contribution by document
- Pages scanned per document
- Total files, total pages, total images found/processed
- Processing time per document
- WER (Word Error Rate) by document (when ground truth is provided)
- CER (Character Error Rate) by document (when ground truth is provided)
- Methodology section (what was done and how metrics were computed)

## Files

- `src/run_evaluation.py`: main evaluation code.
- `run_evaluation.py`: convenience launcher.
- `README.md`: this file.
- `EXAMINER_PACK.md`: presentation guide.
- `input/ground_truth_template.csv`: template for true text labels.
- `output/`: generated reports and metrics.

## Quick Start

From project root:

```powershell
python evaluation/run_evaluation.py
```

This scans `uploads/*.pdf` and writes:

- `evaluation/output/metrics_latest.csv`
- `evaluation/output/summary_latest.json`
- `evaluation/output/examiner_report_latest.md`
- `evaluation/output/examiner_report_latest.html`

To also keep timestamped snapshots, add `--archive`:

```powershell
python evaluation/run_evaluation.py --archive
```

By default, old timestamped files are auto-cleaned so the output folder stays tidy.

## Enable True Accuracy (WER/CER)

Create `evaluation/input/ground_truth.csv` using this format:

```csv
file_name,expected_text
my_notes_01.pdf,"Exact expected transcription text here"
my_notes_02.pdf,"Exact expected transcription text here"
```

Then run:

```powershell
python evaluation/run_evaluation.py --ground-truth evaluation/input/ground_truth.csv
```

Open the polished visual dashboard in your browser:

```powershell
start evaluation/output/examiner_report_latest.html
```

## Suggested KPI Targets

- OCR Success Rate: above 85% on scanned docs
- OCR Contribution: higher is expected for scanned image-heavy docs, lower is expected for text-native PDFs
- WER: below 25% for neat handwriting, below 15% for printed docs
- CER: below 12% for neat handwriting, below 8% for printed docs

## Notes

- All evaluation is local and offline.
- No paid OCR API is required.
- Accuracy improves when you add preprocessing and handwriting OCR models.
