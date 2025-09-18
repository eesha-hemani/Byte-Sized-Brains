import os, io, tempfile, datetime, re, traceback, logging, nltk
from flask import Flask, request, jsonify, send_file, send_from_directory
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
import pdfplumber
from docx import Document as DocxDocument
from langdetect import detect, DetectorFactory, LangDetectException
from googletrans import Translator as GTranslator
from flask_cors import CORS
from dotenv import load_dotenv
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# --- Initial Setup ---
load_dotenv()
DetectorFactory.seed = 0

# --- NLTK Downloader ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK data packages ('punkt', 'stopwords')...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("Download complete.")

# ---------- CONFIGURATION ----------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "kmrl_docs")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "../frontend"))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="", template_folder=FRONTEND_DIR)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- DATABASE CONNECTION ----------
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    fs = gridfs.GridFS(db)
    files_col = db["files"]
    logger.info("Successfully connected to MongoDB.")
except Exception as e:
    logger.error(f"Could not connect to MongoDB: {e}")
    exit(1)

# ---------- ROBUST SUMMARIZER ----------
def generate_robust_summary(text, sentences_count=5):
    if not text or not isinstance(text, str): return "No text to summarize."
    try:
        sentences = sent_tokenize(text)
        if len(sentences) <= sentences_count:
            return "\n".join([f"* {s.strip()}" for s in sentences])

        stop_words = set(stopwords.words('english'))
        words = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
        if not words: return "No summarizable content."

        word_freq = Counter(words)
        max_freq = word_freq.most_common(1)[0][1]
        for word in word_freq: word_freq[word] /= max_freq

        sentence_scores = {sent: sum(word_freq.get(w.lower(), 0) for w in word_tokenize(sent) if w.isalpha()) for sent in sentences}
        summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:sentences_count]
        return "\n".join([f"* {s.strip()}" for s in summary_sentences])
    except Exception:
        logger.error(f"Summarization failed: {traceback.format_exc()}")
        return "Could not generate summary due to an internal error."

# ---------- HELPER FUNCTIONS ----------
def oid_to_str(o): return str(o) if o is not None else None
def save_file_to_gridfs(file_bytes, filename, content_type): return fs.put(file_bytes, filename=filename, contentType=content_type)

def insert_metadata(filename, gridfs_id, content_type, uploaded_by, **kwargs):
    return files_col.insert_one({
        "filename": filename, "gridfs_id": gridfs_id, "content_type": content_type,
        "uploaded_by": uploaded_by, "uploaded_at": datetime.datetime.utcnow(), **kwargs
    }).inserted_id

def extract_text(file_bytes, filename, filetype_hint=None):
    ext = os.path.splitext(filename.lower())[1]
    text = ""
    try:
        if ext == ".pdf" or "pdf" in (filetype_hint or ""):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text = "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
        elif ext == ".docx" or "word" in (filetype_hint or ""):
            doc = DocxDocument(io.BytesIO(file_bytes))
            text = "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())
        elif ext in [".txt", ".text"]:
            text = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Text extraction failed for {filename}: {e}")
    return text.strip()

def is_malayalam(text):
    if not text: return False
    try: return detect(text) == 'ml'
    except LangDetectException: return False

def translate_ml_to_en(text):
    if not text: return ""
    try: return GTranslator().translate(text, src='ml', dest='en').text
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text

# ---------- CORE PROCESSING PIPELINE ----------
def process_and_store(file_bytes, filename, filetype, uploaded_by="anonymous"):
    grid_id = save_file_to_gridfs(file_bytes, filename, filetype)
    meta_id = insert_metadata(filename, grid_id, filetype, uploaded_by)

    extracted_text = extract_text(file_bytes, filename, filetype)
    if not extracted_text:
        files_col.update_one({"_id": meta_id}, {"$set": {"status": "error_no_text"}})
        return {"original_meta_id": oid_to_str(meta_id), "summary_meta_id": oid_to_str(meta_id), "translated": False}

    if is_malayalam(extracted_text):
        translated_text = translate_ml_to_en(extracted_text)
        trans_filename = os.path.splitext(filename)[0] + "_translated.txt"
        trans_grid_id = save_file_to_gridfs(translated_text.encode("utf-8"), trans_filename, "text/plain")
        trans_meta_id = insert_metadata(trans_filename, trans_grid_id, "text/plain", "system", parent_id=meta_id, language="en")
        
        files_col.update_one({"_id": meta_id}, {"$set": {"status": "translated", "language": "ml"}})
        text_to_process = translated_text
        final_meta_id = trans_meta_id
    else:
        files_col.update_one({"_id": meta_id}, {"$set": {"status": "processed", "language": "en"}})
        text_to_process = extracted_text
        final_meta_id = meta_id

    tags, _ = classify_by_rules(text_to_process)
    files_col.update_one({"_id": final_meta_id}, {"$set": {
        "tags": tags, "category": tags[0] if tags else 'miscellaneous'
    }})
    
    return {
        "original_meta_id": oid_to_str(meta_id),
        "summary_meta_id": oid_to_str(final_meta_id),
        "translated": final_meta_id != meta_id,
        "translation_meta_id": oid_to_str(trans_meta_id) if 'trans_meta_id' in locals() else None,
        "tags": tags, "primary_tag": tags[0] if tags else 'miscellaneous'
    }

# ---------- CLASSIFICATION ----------
CATEGORIES = {
    'invoices': ['invoice', 'bill', 'payment', 'amount', 'total', 'due', 'rs', 'tax', 'gst', 'receipt'],
    'safety_reports': ['safety', 'audit', 'risk', 'hazard', 'compliance', 'incident', 'accident', 'inspection'],
    'urgent': ['urgent', 'immediate', 'asap', 'critical', 'emergency', 'priority', 'deadline'],
    'engineering_drawings': ['drawing', 'blueprint', 'cad', 'dimension', 'specification', 'technical', 'design'],
}
def classify_by_rules(text):
    if not text: return ['miscellaneous'], {}
    text_lower = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in text_lower) for cat, kws in CATEGORIES.items()}
    tags = sorted([cat for cat, s in scores.items() if s > 0], key=scores.get, reverse=True)
    return tags or ['miscellaneous'], scores

# ---------- API ROUTES ----------
@app.route('/', defaults={'path': 'home.html'})
@app.route('/<path:path>')
def serve_frontend(path):
    return send_from_directory(FRONTEND_DIR, path)

@app.route("/upload", methods=["POST"])
def upload_route():
    if "file" not in request.files: return jsonify({"error": "no file part"}), 400
    f = request.files["file"]
    try:
        res = process_and_store(f.read(), f.filename, f.mimetype)
        return jsonify(res), 201
    except Exception:
        logger.error(f"Upload failed: {traceback.format_exc()}")
        return jsonify({"error": "processing failed"}), 500

@app.route("/summary/<meta_id>", methods=["GET"])
def get_summary_route(meta_id):
    try:
        meta = files_col.find_one({"_id": ObjectId(meta_id)})
        if not meta: return jsonify({"error": "not_found"}), 404
        
        grid_id = meta.get("gridfs_id")
        if not grid_id or not fs.exists(grid_id): return jsonify({"error": "file not in storage"}), 404

        text = fs.get(grid_id).read().decode('utf-8', 'ignore')
        summary = generate_robust_summary(text)
        return jsonify({"summary": summary})
    except Exception:
        logger.error(f"Summary route failed for {meta_id}: {traceback.format_exc()}")
        return jsonify({"error": "summary generation failed"}), 500

@app.route("/files", methods=["GET"])
def list_files_route():
    out = []
    for r in files_col.find({"parent_id": None}).sort("uploaded_at", -1):
        item = { "meta_id": oid_to_str(r["_id"]), "filename": r.get("filename"),
                 "uploaded_at": r.get("uploaded_at").isoformat(), "category": "processing...",
                 "has_translation": False }
        
        child = files_col.find_one({"parent_id": r["_id"]})
        if child:
            item["category"] = child.get("category", "miscellaneous")
            item["has_translation"] = True
            item["translation_meta_id"] = oid_to_str(child["_id"])
        elif r.get("language") != "ml":
             item["category"] = r.get("category", "miscellaneous")
        out.append(item)
    return jsonify(out)

@app.route("/search-by-tag/<tag>", methods=["GET"])
def search_by_tag_route(tag):
    try:
        parent_ids_from_children = set()
        child_docs_with_tag = files_col.find({"tags": tag, "parent_id": {"$ne": None}})
        for child in child_docs_with_tag:
            if child.get("parent_id"):
                parent_ids_from_children.add(child["parent_id"])

        original_docs_with_tag_cursor = files_col.find({"tags": tag, "parent_id": None})
        original_ids = {doc["_id"] for doc in original_docs_with_tag_cursor}
        
        all_relevant_ids = list(parent_ids_from_children.union(original_ids))

        if not all_relevant_ids:
            return jsonify({"documents": []})

        documents = list(files_col.find({"_id": {"$in": all_relevant_ids}}).sort("uploaded_at", -1))
        
        output_docs = []
        for r in documents:
            item = { "meta_id": oid_to_str(r["_id"]), "filename": r.get("filename"),
                     "uploaded_at": r.get("uploaded_at").isoformat(), "category": "processing...",
                     "has_translation": False }
            
            child = files_col.find_one({"parent_id": r["_id"]})
            if child:
                item["category"] = child.get("category", "miscellaneous")
                item["has_translation"] = True
                item["translation_meta_id"] = oid_to_str(child["_id"])
            elif r.get("language") != "ml":
                if tag in r.get("tags", []):
                    item["category"] = r.get("category", "miscellaneous")
            output_docs.append(item)
            
        return jsonify({"documents": output_docs})
    except Exception:
        logger.error(f"Search by tag failed for {tag}: {traceback.format_exc()}")
        return jsonify({"error": "search failed"}), 500

@app.route("/download/<meta_id>", methods=["GET"])
def download_route(meta_id):
    try:
        meta = files_col.find_one({"_id": ObjectId(meta_id)})
        if not meta: return jsonify({"error": "not_found"}), 404
        gf = fs.get(meta["gridfs_id"])
        return send_file(io.BytesIO(gf.read()), mimetype=meta.get("content_type"), as_attachment=True, download_name=meta.get("filename"))
    except Exception:
        logger.error(f"Download failed for {meta_id}: {traceback.format_exc()}")
        return jsonify({"error": "download failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)

