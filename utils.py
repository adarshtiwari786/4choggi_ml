import os
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import fitz


# Configuration

DATA_DIR = Path("data")
JD_FILE = DATA_DIR / "job_description.pdf"
RESUMES_DIR = DATA_DIR / "resumes"
GITHUB_INSIGHTS_FILE = DATA_DIR / "github_insights.json"

SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
THRESHOLD = 0.4

# PDF Extraction

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

    text = re.sub(r"\s+", " ", text.strip())
    return text

# Document Loading

def load_documents():
    """Load JD, all resumes, and optional GitHub insights."""
    jd_text = ""
    resumes = {}
    github_insights = {}

    if JD_FILE.exists():
        jd_text = extract_text_from_pdf(JD_FILE)
    else:
        print(f"[Warning] Job description not found at {JD_FILE}")

    if RESUMES_DIR.exists():
        for f in RESUMES_DIR.glob("*.pdf"):
            resumes[f.name] = extract_text_from_pdf(f)
    else:
        print(f"[Warning] Resume folder not found at {RESUMES_DIR}")

    if GITHUB_INSIGHTS_FILE.exists():
        try:
            with open(GITHUB_INSIGHTS_FILE, "r") as f:
                github_insights = json.load(f)
        except Exception as e:
            print(f"[Warning] Could not load GitHub insights: {e}")

    print(f"[✓] JD text length: {len(jd_text)}")
    print(f"[✓] Resumes loaded: {len(resumes)}")
    return jd_text, resumes, github_insights


# Keyword Extraction

def extract_keywords(text: str):
    """Extract unique non-stopword keywords from text."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    stop = {"the", "and", "for", "with", "this", "that", "from", "your", "have", "are", "was", "you"}
    return list(set(w for w in words if w not in stop))

# Semantic Similarity

def compute_similarity(jd_text: str, resumes: dict, github_insights: dict = None, threshold: float = THRESHOLD):
    """Compute similarity between JD and resumes (optionally merging GitHub insights)."""
    github_insights = github_insights or {}
    model = SentenceTransformer(SENTENCE_MODEL_NAME)

    # Encode JD once
    jd_emb = model.encode([jd_text])
    results = []

    for name, text in resumes.items():
        github_text = github_insights.get(name.replace(".pdf", ""), "")
        combined_text = text + " " + github_text
        emb = model.encode([combined_text])
        sim = cosine_similarity(jd_emb, emb)[0][0]
        results.append((name, sim))

    df = pd.DataFrame(results, columns=["Candidate", "Similarity"]).sort_values("Similarity", ascending=False)
    return df

# Save Filtered Results

def save_filtered_results(df: pd.DataFrame, threshold: float = THRESHOLD, output_path: str = "filtered_matches.csv"):
    """Filter matches above threshold and save to CSV."""
    filtered = df[df["Similarity"] >= threshold]
    filtered.to_csv(output_path, index=False)
    print(f"[✓] Filtered results saved to {output_path}")
    return filtered
