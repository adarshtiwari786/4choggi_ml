import os
import re
import json
import requests
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import fitz  # PyMuPDF


SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
THRESHOLD = 0.4


def ensure_data_dirs(*dirs):
    """Create required folders if missing."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def download_pdf_from_url(url: str, save_path: Path) -> Path:
    """Download a PDF from a given URL (Google Cloud, etc.)"""
    print(f"[+] Downloading PDF from {url}")
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"[✓] Saved to {save_path}")
        return save_path
    except Exception as e:
        print(f"[Error] Failed to download {url}: {e}")
        return None


def extract_text_from_pdf(filepath: Path) -> str:
    """Extract text from a PDF using pdfplumber or PyMuPDF fallback."""
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception:
        try:
            with fitz.open(filepath) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            print(f"[Error] Could not read {filepath}: {e}")
            return ""

    return re.sub(r"\s+", " ", text.strip())


def compute_similarity(jd_text: str, resumes: dict, threshold: float = THRESHOLD):
    """Compute semantic similarity between JD and resumes."""
    model = SentenceTransformer(SENTENCE_MODEL_NAME)
    jd_emb = model.encode([jd_text])
    results = []

    for name, text in resumes.items():
        emb = model.encode([text])
        sim = cosine_similarity(jd_emb, emb)[0][0]
        results.append((name, sim))

    df = pd.DataFrame(results, columns=["Candidate", "Similarity"])
    df = df.sort_values("Similarity", ascending=False)
    filtered = df[df["Similarity"] >= threshold]
    return filtered


def save_filtered_results(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save filtered matches to CSV."""
    out_path = output_dir / "filtered_matches.csv"
    df.to_csv(out_path, index=False)
    print(f"[✓] Results saved to {out_path}")
    return out_path
