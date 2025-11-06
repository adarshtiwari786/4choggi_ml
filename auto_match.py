"""
Auto Resume Matching Client
---------------------------
Runs end-to-end: upload JD + resumes, run matching, download results.
"""

import os
import requests
import json
from pathlib import Path

# Configuration
BASE_URL = "http://127.0.0.1:5000"
DATA_DIR = Path("data")
JD_FILE = DATA_DIR / "job_description.pdf"
RESUME_DIR = DATA_DIR / "resumes"
THRESHOLD = 0.4

def upload_jd():
    print(f"[+] Uploading JD: {JD_FILE.name}")
    with open(JD_FILE, "rb") as f:
        res = requests.post(f"{BASE_URL}/upload_jd", files={"file": f})
    print("   ↳", res.json())
    return res.json().get("path")

def upload_resumes():
    files = [("files", open(f, "rb")) for f in RESUME_DIR.glob("*.pdf")]
    if not files:
        print("No resumes found in", RESUME_DIR)
        return []
    print(f"[+] Uploading {len(files)} resumes...")
    res = requests.post(f"{BASE_URL}/upload_resumes", files=files)
    for f in files: f[1].close()
    print("   ↳", res.json())
    return res.json().get("paths", [])

def run_matching(jd_path=None):
    print("[+] Running matching...")
    payload = {"threshold": THRESHOLD}
    if jd_path: payload["jd_path"] = jd_path
    res = requests.post(f"{BASE_URL}/match", json=payload)
    result = res.json()
    print(f"   ↳ Top {len(result['matches'])} matches")
    for m in result["matches"][:5]:
        print(f"   - {m['Candidate']}: {m['Similarity']:.2f}")
    return result

def download_csv(csv_path):
    print("[+] Downloading CSV...")
    out_path = Path("filtered_matches.csv")
    r = requests.get(f"{BASE_URL}/download", params={"path": csv_path})
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"[✓] Results saved to {out_path.resolve()}")

def main():
    print("🚀 Auto Resume Matcher Started")
    if not JD_FILE.exists():
        print("❌ Missing JD file:", JD_FILE)
        return
    if not RESUME_DIR.exists():
        print("❌ Missing resume folder:", RESUME_DIR)
        return

    jd_path = upload_jd()
    upload_resumes()
    result = run_matching(jd_path)
    csv_path = result.get("csv")
    if csv_path:
        download_csv(csv_path)
    print("\n✅ Matching complete!")

if __name__ == "__main__":
    main()
