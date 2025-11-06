import os
import io
import uuid
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from utils import extract_text_from_pdf, compute_similarity, ensure_data_dirs

# Load env (optional)
from dotenv import load_dotenv
load_dotenv()

DATA_DIR = Path(os.getenv('DATA_DIR', 'data'))
RESUMES_DIR = DATA_DIR / 'resumes'
JD_DIR = DATA_DIR / 'jd'
OUTPUT_DIR = DATA_DIR / 'output'

ensure_data_dirs(DATA_DIR, RESUMES_DIR, JD_DIR, OUTPUT_DIR)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Health
@app.route('/', methods=['GET'])
def hello():
    return jsonify({'status': 'ok', 'message': 'Resume matcher running'})

# Upload JD (file)
@app.route('/upload_jd', methods=['POST'])
def upload_jd():
    if 'file' not in request.files and 'url' not in request.json:
        return jsonify({'error': 'Provide multipart file under `file` or a JSON with `url`'}), 400

    if 'file' in request.files:
        f = request.files['file']
        filename = secure_filename(f.filename)
        target = JD_DIR / filename
        f.save(target)
        return jsonify({'jd_path': str(target)}), 201

    data = request.get_json() or {}
    url = data.get('url')
    if url:
        import requests
        r = requests.get(url)
        if r.status_code != 200:
            return jsonify({'error': 'Failed to download file from url'}), 400
        filename = f'jd_{uuid.uuid4().hex}.pdf'
        target = JD_DIR / filename
        with open(target, 'wb') as fh:
            fh.write(r.content)
        return jsonify({'jd_path': str(target)}), 201

    return jsonify({'error': 'No valid JD provided'}), 400

# Upload resume(s) — accept multiple files
@app.route('/upload_resumes', methods=['POST'])
def upload_resumes():
    if 'files' not in request.files:
        return jsonify({'error': 'Send files under field name `files` (multipart/form-data)'}), 400

    files = request.files.getlist('files')
    saved = []
    for f in files:
        filename = secure_filename(f.filename)
        target = RESUMES_DIR / filename
        f.save(target)
        saved.append(str(target))

    return jsonify({'saved': saved}), 201

# Run matching
@app.route('/match', methods=['POST'])
def match():
    """Run matching. POST JSON with:
        {
          "jd_path": "data/jd/yourfile.pdf",  # OR omit to pick the latest JD
          "threshold": 0.4  # optional
        }
    """
    payload = request.get_json() or {}
    jd_path = payload.get('jd_path')
    threshold = float(payload.get('threshold', os.getenv('THRESHOLD', 0.4)))

    # choose JD
    if jd_path:
        jd_file = Path(jd_path)
        if not jd_file.exists():
            return jsonify({'error': 'Provided jd_path does not exist'}), 400
    else:
        jds = sorted(JD_DIR.glob('*.pdf'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not jds:
            return jsonify({'error': 'No JD uploaded yet'}), 400
        jd_file = jds[0]

    # read resumes
    resumes = {f.name: str(f) for f in RESUMES_DIR.glob('*.pdf')}
    if not resumes:
        return jsonify({'error': 'No resumes uploaded yet'}), 400

    # compute
    df = compute_similarity(str(jd_file), resumes, threshold=float(threshold))

    # save CSV
    out_csv = OUTPUT_DIR / f'filtered_matches_{uuid.uuid4().hex}.csv'
    df.to_csv(out_csv, index=False)

    # prepare response
    top = df.head(50).to_dict(orient='records')
    return jsonify({'matches': top, 'csv': str(out_csv)}), 200

# Download CSV
@app.route('/download', methods=['GET'])
def download():
    path = request.args.get('path')
    if not path:
        return jsonify({'error': 'Provide ?path=/full/path/to.csv'}), 400
    p = Path(path)
    if not p.exists():
        return jsonify({'error': 'File not found'}), 404
    return send_file(str(p), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))