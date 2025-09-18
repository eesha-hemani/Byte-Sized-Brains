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
# This ensures the necessary data packages are available on startup.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)
    print("Download complete.")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK 'stopwords' corpus not found. Downloading...")
    nltk.download('stopwords', quiet=True)
    print("Download complete.")

# ---------- CONFIGURATION ----------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "kmrl_docs")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "../frontend"))

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path="",
    template_folder=FRONTEND_DIR
)
CORS(app, supports_credentials=True)

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

# ---------- NEW ROBUST SUMMARIZER ----------
def generate_robust_summary(text, sentences_count=5):
    if not text or not isinstance(text, str):
        return "No text content to summarize."

    try:
        sentences = sent_tokenize(text)
        
        # Guardrail: If the document is short, return the whole text.
        if len(sentences) <= sentences_count:
            return "\n".join([f"* {s.strip()}" for s in sentences])

        stop_words = set(stopwords.words('english'))
        words = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stop_words]
        
        if not words:
            return "Document contains no summarizable content."

        word_freq = Counter(words)
        max_freq = word_freq.most_common(1)[0][1]
        
        # Normalize frequencies
        for word in word_freq:
            word_freq[word] = (word_freq[word] / max_freq)

        # Score sentences based on word frequencies
        sentence_scores = {}
        for sent in sentences:
            sent_words = [word.lower() for word in word_tokenize(sent) if word.isalpha()]
            score = sum(word_freq.get(word, 0) for word in sent_words)
            if len(sent_words) > 0:
                 sentence_scores[sent] = score

        # Get the top sentences
        sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
        summary_sentences = sorted_sentences[:sentences_count]
        
        summary = "\n".join([f"* {s.strip()}" for s in summary_sentences])
        return summary

    except Exception as e:
        logger.error(f"Robust summarization failed: {traceback.format_exc()}")
        return "Could not generate summary due to an internal processing error."


# ---------- HELPER FUNCTIONS ----------
def oid_to_str(o): return str(o) if o is not None else None
def save_file_to_gridfs(file_bytes, filename, content_type): return fs.put(file_bytes, filename=filename, contentType=content_type)

def insert_metadata(filename, gridfs_id, content_type, uploaded_by, **kwargs):
    doc = {
        "filename": filename, "gridfs_id": gridfs_id, "content_type": content_type,
        "uploaded_by": uploaded_by, "uploaded_at": datetime.datetime.utcnow(),
        **kwargs
    }
    return files_col.insert_one(doc).inserted_id

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
        return ""
    return text.strip()

def is_malayalam(text):
    if not text: return False
    try:
        return detect(text) == 'ml'
    except LangDetectException:
        return False

def translate_ml_to_en(text):
    if not text: return ""
    try:
        return GTranslator().translate(text, src='ml', dest='en').text
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text

# ---------- CORE PROCESSING PIPELINE ----------
def process_and_store(file_bytes, filename, filetype, uploaded_by="anonymous"):
    # 1. Save original file and create metadata record
    grid_id = save_file_to_gridfs(file_bytes, filename, filetype)
    meta_id = insert_metadata(filename, grid_id, filetype, uploaded_by, source_type="upload")

    # 2. Extract text from the document
    extracted_text = extract_text(file_bytes, filename, filetype)
    if not extracted_text:
        files_col.update_one({"_id": meta_id}, {"$set": {"translation_status": "no_extractable_text"}})
        return {"original_meta_id": oid_to_str(meta_id), "translated": False}

    # 3. Check for Malayalam and translate if necessary
    if is_malayalam(extracted_text):
        translated_text = translate_ml_to_en(extracted_text)
        trans_filename = os.path.splitext(filename)[0] + "_translated.txt"
        
        # THIS IS WHERE THE TRANSLATED FILE IS SAVED
        trans_grid_id = save_file_to_gridfs(translated_text.encode("utf-8"), trans_filename, "text/plain")
        trans_meta_id = insert_metadata(trans_filename, trans_grid_id, "text/plain", "system", parent_id=meta_id, language="en")
        
        files_col.update_one({"_id": meta_id}, {"$set": {"translation_status": "translated", "language": "ml"}})
        text_to_classify = translated_text
        final_meta_id = trans_meta_id
    else:
        files_col.update_one({"_id": meta_id}, {"$set": {"translation_status": "not_needed", "language": "en"}})
        text_to_classify = extracted_text
        final_meta_id = meta_id

    # 4. Classify the final English text
    tags, _ = classify_by_rules(text_to_classify)
    files_col.update_one({"_id": final_meta_id}, {"$set": {
        "tags": tags, "category": tags[0] if tags else 'miscellaneous',
        "classification_status": "completed"
    }})
    
    return {
        "original_meta_id": oid_to_str(meta_id),
        "translated": final_meta_id != meta_id,
        "translation_meta_id": oid_to_str(final_meta_id) if final_meta_id != meta_id else None,
        "tags": tags, "primary_tag": tags[0] if tags else 'miscellaneous'
    }

# ---------- RULES-BASED CLASSIFICATION ----------
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
    tags = sorted([cat for cat, score in scores.items() if score > 0], key=lambda t: scores[t], reverse=True)
    return tags if tags else ['miscellaneous'], scores

# ---------- API ROUTES ----------
@app.route('/', defaults={'path': 'home.html'})
@app.route('/<path:path>')
def serve_frontend(path):
    return send_from_directory(FRONTEND_DIR, path)

@app.route("/upload", methods=["POST"])
def upload_route():
    if "file" not in request.files: return jsonify({"error": "no file part"}), 400
    f = request.files["file"]
    if f.filename == "": return jsonify({"error": "no selected file"}), 400
    try:
        res = process_and_store(f.read(), f.filename, f.mimetype)
        return jsonify(res), 201
    except Exception as e:
        logger.error(f"Upload failed: {traceback.format_exc()}")
        return jsonify({"error": "processing failed"}), 500

@app.route("/summary/<meta_id>", methods=["GET"])
def get_summary_route(meta_id):
    try:
        meta = files_col.find_one({"_id": ObjectId(meta_id)})
        if not meta: return jsonify({"error": "not_found"}), 404
        
        # If original was ML, get summary from the translated child
        if meta.get("translation_status") == "translated":
            child = files_col.find_one({"parent_id": meta["_id"]})
            if child: meta = child

        grid_id = meta.get("gridfs_id")
        if not grid_id or not fs.exists(grid_id): return jsonify({"error": "file not in storage"}), 404

        text = fs.get(grid_id).read().decode('utf-8', 'ignore')
        summary = generate_robust_summary(text)
        return jsonify({"summary": summary})
    except Exception as e:
        logger.error(f"Summary route failed for {meta_id}: {traceback.format_exc()}")
        return jsonify({"error": "summary generation failed"}), 500

@app.route("/files", methods=["GET"])
def list_files_route():
    out = []
    for r in files_col.find({"parent_id": None}).sort("uploaded_at", -1):
        item = {
            "meta_id": oid_to_str(r["_id"]),
            "filename": r.get("filename"),
            "uploaded_at": r.get("uploaded_at").isoformat() if r.get("uploaded_at") else None,
            "category": "processing...",
            "has_translation": False
        }
        if r.get("translation_status") == "translated":
            child = files_col.find_one({"parent_id": r["_id"]})
            if child:
                item["category"] = child.get("category", "miscellaneous")
                item["has_translation"] = True
                item["translation_meta_id"] = oid_to_str(child["_id"])
        elif r.get("language") == "en":
             item["category"] = r.get("category", "miscellaneous")
        out.append(item)
    return jsonify(out)

@app.route("/download/<meta_id>", methods=["GET"])
def download_route(meta_id):
    try:
        meta = files_col.find_one({"_id": ObjectId(meta_id)})
        if not meta: return jsonify({"error": "not_found"}), 404
        gf = fs.get(meta["gridfs_id"])
        return send_file(io.BytesIO(gf.read()), mimetype=meta.get("content_type"), as_attachment=True, download_name=meta.get("filename"))
    except Exception as e:
        logger.error(f"Download failed for {meta_id}: {e}")
        return jsonify({"error": "download failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)

