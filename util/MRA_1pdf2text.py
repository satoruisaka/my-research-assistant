"""
MRA_1pdf2text.py

The 1st step in the MRA data preparation pipeline.

PDF to Text (markdown) Conversion using OCR when needed.
- Scans PDFs in 'pdfs_in/' for extractable text.
- Applies OCR via OCRmyPDF to scanned PDFs.
- Converts PDFs to Markdown using PyMuPDF4LLM.
- Logs failures to 'failed_files.log'.

Dependencies:
  pip install -U ocrmypdf PyMuPDF opencv-python PyMuPDF4LLM
  sudo apt install tesseract-ocr pngquant ghostscript
  (pip install -U pymupdf-layout) # Not used in this version

Usage:
  Place PDFs in 'pdfs_in/' and run:
    python PDF2Text_pipeline3.py
  Output Markdown files will be saved in 'pdfs_md/'.
"""
import os
import subprocess
import fitz  # PyMuPDF
from pymupdf4llm import to_markdown  # PyMuPDF4LLM

# Paths
INPUT_DIR = "pdfs_in"
OCR_DIR = "pdfs_ocr"
MD_DIR = "pdfs_md"
LOG_PATH = "failed_files.log"
os.makedirs(OCR_DIR, exist_ok=True)
os.makedirs(MD_DIR, exist_ok=True)

def log_failure(filename, reason):
    """Append a failed filename and reason to the log file."""
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"{filename}: {reason}\n")

def has_text_layer(pdf_path):
    """Check if a PDF has extractable text using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text("text").strip()
            if text:
                return True
        return False
    except Exception as e:
        log_failure(os.path.basename(pdf_path), f"text check failed: {e}")
        return False

def run_ocrmypdf(input_path, output_path):
    """Run OCRmyPDF on a scanned PDF."""
    try:
        subprocess.run([
            "ocrmypdf",
            "--skip-text",
            "--optimize", "3",
            "--output-type", "pdfa",
            input_path,
            output_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        log_failure(os.path.basename(input_path), f"OCR failed: {e}")
        raise

def convert_to_markdown(pdf_path, md_path):
    """Convert PDF to Markdown using PyMuPDF4LLM."""
    try:
        doc = fitz.open(pdf_path)
        md_text = to_markdown(doc)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)
    except Exception as e:
        log_failure(os.path.basename(pdf_path), f"Markdown conversion failed: {e}")
        raise

def process_pdfs():
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith(".pdf"):
            continue

        input_path = os.path.join(INPUT_DIR, filename)
        ocr_path = os.path.join(OCR_DIR, filename)
        md_path = os.path.join(MD_DIR, filename.replace(".pdf", ".md"))

        try:
            if has_text_layer(input_path):
                print(f"[SKIP OCR] {filename} already has text.")
                if input_path != ocr_path:
                    subprocess.run(["cp", input_path, ocr_path])
            else:
                print(f"[RUN OCR] {filename} has no text, running OCRmyPDF...")
                run_ocrmypdf(input_path, ocr_path)

            print(f"[MARKDOWN] Converting {filename} → {md_path}")
            convert_to_markdown(ocr_path, md_path)

        except Exception as e:
            print(f"[SKIP FILE] {filename} failed: {e}")
            continue

if __name__ == "__main__":
    # Clear previous log
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    process_pdfs()
    print("✅ Pipeline complete. All PDFs converted to Markdown in pdfs_md/")
    print("📄 Failed files (if any) are logged in failed_files.log")
    