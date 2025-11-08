from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer, util

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Similarity threshold (default = 0.6)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.6))


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Resume Filtering API with Scores ✅"})


@app.route("/filter_resumes", methods=["POST"])
def filter_resumes():
    """
    Input JSON Example:
    {
      "job": {
        "title": "Full Stack Developer",
        "expLevel": "Mid-level",
        "location": "Bangalore, India",
        "requiredSkills": ["JavaScript", "Node.js", "Express.js", "MongoDB", "React"],
        "jobDescription": "We are looking for..."
      },
      "resume": [
        {"summary": "...", "_id": "r1"},
        {"summary": "...", "_id": "r2"}
      ]
    }

    Output JSON Example:
    [
      {"summary": "...", "_id": "r2", "similarity_score": 0.84},
      {"summary": "...", "_id": "r1", "similarity_score": 0.78}
    ]
    """

    try:
        data = request.get_json()
        job = data.get("job", {})
        resumes = data.get("resume", [])

        if not job or not resumes:
            return jsonify({"error": "Missing 'job' or 'resume' data"}), 400

        # Combine job fields into one text block
        job_text = f"{job.get('title', '')} {job.get('expLevel', '')} {job.get('location', '')} " \
                   f"{' '.join(job.get('requiredSkills', []))} {job.get('jobDescription', '')}"

        # Encode the job description
        job_emb = model.encode(job_text, convert_to_tensor=True)

        qualified_resumes = []

        # Evaluate each resume in the list
        for info in resumes:
            summary = info.get("summary", "").strip()
            _id = info.get("_id", "")

            if not summary:
                continue

            resume_emb = model.encode(summary, convert_to_tensor=True)
            score = float(util.pytorch_cos_sim(job_emb, resume_emb).item())

            # Only include resumes that meet or exceed threshold
            if score >= SIMILARITY_THRESHOLD:
                qualified_resumes.append({
                    "summary": summary,
                    "_id": _id,
                    "similarity_score": round(score, 3)
                })

        # Sort by similarity score (descending)
        qualified_resumes.sort(key=lambda x: x["similarity_score"], reverse=True)

        return jsonify(qualified_resumes)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
