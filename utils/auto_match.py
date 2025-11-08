import requests
import json

BASE_URL = "http://127.0.0.1:5000"

# 🔗 Put your Google Cloud URLs here
JD_URL = "https://storage.googleapis.com/your_bucket/job_description.pdf"
RESUME_URLS = [
    "https://storage.googleapis.com/your_bucket/resume1.pdf",
    "https://storage.googleapis.com/your_bucket/resume2.pdf"
]


def main():
    print("🚀 Starting auto-match process...")

    # Step 1: Upload JD + resumes from URLs
    payload = {"jd_url": JD_URL, "resume_urls": RESUME_URLS}
    res = requests.post(f"{BASE_URL}/upload_from_url", json=payload)
    data = res.json()
    print("✅ Upload complete:", json.dumps(data, indent=2))

    # Step 2: Run matching
    res = requests.post(f"{BASE_URL}/match")
    result = res.json()
    print("\n✅ Matching done. Top candidates:")
    for m in result["top_matches"][:5]:
        print(f" - {m['Candidate']}: {m['Similarity']:.2f}")

    # Step 3: Download CSV
    csv_path = result["csv_path"]
    csv_data = requests.get(f"{BASE_URL}/download", params={"path": csv_path})
    with open("final_results.csv", "wb") as f:
        f.write(csv_data.content)
    print(f"\n📂 Results saved to final_results.csv")


if __name__ == "__main__":
    main()
