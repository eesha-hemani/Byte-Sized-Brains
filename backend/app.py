import os, io, tempfile, datetime, re, traceback, logging
from flask import Flask, request, jsonify, send_file, send_from_directory
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
import requests
import pdfplumber
from docx import Document as DocxDocument
from langdetect import detect, DetectorFactory, LangDetectException
from googletrans import Translator as GTranslator
from flask_cors import CORS
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Make langdetect deterministic
DetectorFactory.seed = 0

# ---------- CONFIG ----------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "kmrl_docs")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        logging.error(f"Failed to configure Gemini API: {e}")
        GEMINI_API_KEY = None # Disable if configuration fails

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Correctly point to the 'frontend' directory which is a sibling of the 'backend' directory.
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

# ---------- Connect to MongoDB & GridFS ----------
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    fs = gridfs.GridFS(db)
    files_col = db["files"]
    logger.info("Successfully connected to MongoDB.")
except Exception as e:
    logger.error(f"Could not connect to MongoDB: {e}")
    exit(1)


@app.route('/', defaults={'path': 'home.html'})
@app.route('/<path:path>')
def serve_frontend(path):
    # Check if the requested path exists in the frontend directory
    if not os.path.exists(os.path.join(FRONTEND_DIR, path)):
        # If not, return a 404 error
        return jsonify({"error": "not_found"}), 404
    return send_from_directory(FRONTEND_DIR, path)

# ---------- Helpers ----------
def oid_to_str(o):
    return str(o) if o is not None else None

def save_file_to_gridfs(file_bytes, filename, content_type):
    return fs.put(file_bytes, filename=filename, contentType=content_type)

def insert_metadata(filename, gridfs_id, content_type, uploaded_by, source_type="upload", source_url=None, parent_id=None, language=None, translation_status=None):
    doc = {
        "filename": filename,
        "gridfs_id": gridfs_id,
        "content_type": content_type,
        "uploaded_by": uploaded_by,
        "uploaded_at": datetime.datetime.utcnow(),
        "source_type": source_type,
        "source_url": source_url,
        "parent_id": parent_id,
        "language": language,
        "translation_status": translation_status
    }
    return files_col.insert_one(doc).inserted_id

def update_metadata(meta_id, patch: dict):
    files_col.update_one({"_id": ObjectId(meta_id)}, {"$set": patch})

def extract_text_if_digital(file_bytes, filename, filetype_hint=None):
    ext = os.path.splitext(filename.lower())[1]
    text = ""
    try:
        if ext == ".pdf" or (filetype_hint and "pdf" in filetype_hint):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text = "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
        elif ext == ".docx" or (filetype_hint and "word" in filetype_hint):
            doc = DocxDocument(io.BytesIO(file_bytes))
            text = "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())
        elif ext in [".txt", ".text"]:
            text = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Text extraction failed for {filename}: {str(e)}")
        return "", None
        
    text = text.strip()
    lang = None
    if text:
        try:
            lang = detect(text)
        except LangDetectException:
            lang = 'en'
    
    return text, lang

def is_malayalam(text, detected_lang):
    if not text: return False
    if detected_lang == 'ml': return True
    return any('\u0D00' <= char <= '\u0D7F' for char in text)

def translate_ml_to_en(text):
    if not text: return ""
    translator = GTranslator()
    try:
        return translator.translate(text, src='ml', dest='en').text
    except Exception as e:
        logger.warning(f"Googletrans translation failed: {e}")
        return text

# ---------- AI Summary Generation ----------
def generate_summary(text):
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set.")
        return "Summary generation is not configured. API Key is missing."
    if not text:
        return "No text content to summarize."
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Provide a brief, professional summary of the following document. Use bullet points for key items:\n\n---\n\n{text}"
        response = model.generate_content(prompt)
        return response.text
    except google_exceptions.PermissionDenied as e:
        logger.error(f"Gemini API Permission Denied: {e}")
        return "Could not generate summary: The API key is invalid or the 'Generative Language API' is not enabled in your Google Cloud project."
    except google_exceptions.InvalidArgument as e:
        logger.error(f"Gemini API Invalid Argument: {e}")
        return "Could not generate summary: The AI service received an invalid request. This might be a content safety issue."
    except Exception as e:
        logger.error(f"Gemini summary generation failed: {traceback.format_exc()}")
        if 'api key not valid' in str(e).lower():
             return "Could not generate summary: Your API Key is not valid. Please check it in your .env file."
        return "Could not generate summary: An unexpected error occurred with the AI service."

# ---------- Processing pipeline ----------
def process_and_store(file_bytes, filename, filetype, uploaded_by="anonymous"):
    grid_id = save_file_to_gridfs(file_bytes, filename, filetype)
    meta_id = insert_metadata(filename, grid_id, filetype, uploaded_by)
    
    extracted, detected_lang = extract_text_if_digital(file_bytes, filename, filetype)
    
    if not extracted:
        update_metadata(meta_id, {"translation_status": "no_extractable_text"})
        return {"original_meta_id": oid_to_str(meta_id), "translated": False, "reason": "no_extractable_text"}
    
    if not is_malayalam(extracted, detected_lang):
        tags, _ = classify_by_rules(extracted)
        update_document_with_tags(meta_id, tags)
        update_metadata(meta_id, {"translation_status": "not_needed", "language": detected_lang or 'en'})
        return {"original_meta_id": oid_to_str(meta_id), "translated": False, "tags": tags, "primary_tag": tags[0] if tags else 'miscellaneous'}

    translated_text = translate_ml_to_en(extracted)
    trans_filename = os.path.splitext(filename)[0] + "_translated.txt"
    trans_grid_id = save_file_to_gridfs(translated_text.encode("utf-8"), trans_filename, "text/plain")
    trans_meta_id = insert_metadata(trans_filename, trans_grid_id, "text/plain", "system_translator", parent_id=meta_id, language="en", translation_status="translated")
    
    tags, _ = classify_by_rules(translated_text)
    update_document_with_tags(trans_meta_id, tags)
    update_metadata(meta_id, {"translation_status": "translated", "language": detected_lang})
    
    return {"original_meta_id": oid_to_str(meta_id), "translated": True, "translation_meta_id": oid_to_str(trans_meta_id), "tags": tags, "primary_tag": tags[0] if tags else 'miscellaneous'}

# ---------- Rules-Based Classification ----------
CATEGORIES = {
    'invoices': ['invoice', 'bill', 'payment', 'amount', 'total', 'due', 'rs', 'tax', 'gst', 'receipt', 'quotation'],
    'safety_reports': ['safety', 'audit', 'risk', 'hazard', 'compliance', 'incident', 'accident', 'inspection', 'hse'],
    'urgent': ['urgent', 'immediate', 'asap', 'critical', 'emergency', 'priority', 'deadline'],
    'engineering_drawings': ['drawing', 'blueprint', 'cad', 'dimension', 'specification', 'technical', 'design', 'engineering', 'schematics'],
}

def classify_by_rules(text):
    if not text: return ['miscellaneous'], {}
    text_lower = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in text_lower) for cat, kws in CATEGORIES.items()}
    
    tags = [cat for cat, score in scores.items() if score > 0]
    tags.sort(key=lambda t: scores.get(t, 0), reverse=True)
    if not tags: tags = ['miscellaneous']
    
    return tags, scores

def update_document_with_tags(meta_id, tags):
    files_col.update_one(
        {"_id": ObjectId(meta_id)},
        {"$set": {
            "category": tags[0] if tags else 'miscellaneous',
            "tags": tags,
            "classification_status": "completed",
            "classified_at": datetime.datetime.utcnow()
        }}
    )

# ---------- ROUTES ----------
@app.route("/upload", methods=["POST"])
def upload_route():
    if "file" not in request.files: return jsonify({"error": "no file part"}), 400
    f = request.files["file"]
    if f.filename == "": return jsonify({"error": "no selected file"}), 400
    
    try:
        res = process_and_store(f.read(), f.filename, f.mimetype)
        return jsonify(res), 201
    except Exception as e:
        logger.error(f"Upload processing failed: {traceback.format_exc()}")
        return jsonify({"error": "processing_failed", "details": str(e)}), 500

@app.route("/summary/<meta_id>", methods=["GET"])
def get_summary_route(meta_id):
    if not GEMINI_API_KEY:
        return jsonify({"error": "summary_unavailable", "message": "The AI summarization service is not configured on the server."}), 503

    try:
        meta = files_col.find_one({"_id": ObjectId(meta_id)})
        if not meta: return jsonify({"error": "not_found"}), 404

        if meta.get("translation_status") == "translated":
            translated_doc = files_col.find_one({"parent_id": meta["_id"]})
            if translated_doc: meta = translated_doc
        
        grid_id = meta.get("gridfs_id")
        if not grid_id or not fs.exists(grid_id):
            return jsonify({"error": "file_not_found_in_gridfs"}), 404

        gf = fs.get(grid_id)
        text = gf.read().decode('utf-8', errors='ignore')
        
        summary = generate_summary(text)
        # Check if the summary is an error message from our function
        if "Could not generate summary" in summary or "is not configured" in summary:
            return jsonify({"error": "summary_failed", "message": summary}), 500

        return jsonify({"summary": summary})
        
    except Exception as e:
        logger.error(f"Summary generation route failed for {meta_id}: {traceback.format_exc()}")
        return jsonify({"error": "summary_failed", "details": str(e)}), 500

@app.route("/files", methods=["GET"])
def list_files_enhanced():
    out = []
    for r in files_col.find({"parent_id": None}).sort("uploaded_at", -1).limit(200):
        category = r.get("category")
        if not category and r.get("translation_status") == "translated":
            child_doc = files_col.find_one({"parent_id": r["_id"]})
            if child_doc: category = child_doc.get("category", "miscellaneous")

        out.append({
            "meta_id": oid_to_str(r["_id"]),
            "filename": r.get("filename"),
            "uploaded_at": r.get("uploaded_at").isoformat() if r.get("uploaded_at") else None,
            "category": category or "miscellaneous",
        })
    return jsonify(out)

@app.route("/download/<meta_id>", methods=["GET"])
def download(meta_id):
    try:
        meta = files_col.find_one({"_id": ObjectId(meta_id)})
        if not meta: return jsonify({"error": "not_found"}), 404
        gf = fs.get(meta["gridfs_id"])
        return send_file(io.BytesIO(gf.read()), mimetype=meta.get("content_type"), as_attachment=True, download_name=meta.get("filename"))
    except Exception as e:
        logger.error(f"Download failed for {meta_id}: {e}")
        return jsonify({"error": "download_failed", "details": str(e)}), 500

@app.route("/reprocess/<meta_id>", methods=["POST"])
def reprocess(meta_id):
    try:
        meta = files_col.find_one({"_id": ObjectId(meta_id)})
        if not meta: return jsonify({"error": "not_found"}), 404
        
        child_doc = files_col.find_one_and_delete({"parent_id": ObjectId(meta_id)})
        if child_doc and child_doc.get("gridfs_id"): fs.delete(child_doc["gridfs_id"])

        gf = fs.get(meta["gridfs_id"])
        res = process_and_store(gf.read(), meta["filename"], meta.get("content_type"))
        return jsonify(res)
    except Exception as e:
        logger.error(f"Reprocessing failed for {meta_id}: {traceback.format_exc()}")
        return jsonify({"error": "reprocessing_failed", "details": str(e)}), 500

@app.route("/search-by-tag/<tag>", methods=["GET"])
def search_by_tag(tag):
    try:
        query = {"tags": tag}
        tagged_docs_cursor = files_col.find(query)
        
        parent_ids = {doc.get("parent_id") for doc in tagged_docs_cursor if doc.get("parent_id")}
        original_ids = {doc["_id"] for doc in files_col.find({**query, "parent_id": None})}
        
        all_relevant_ids = list(parent_ids.union(original_ids))
        if not all_relevant_ids: return jsonify({"documents": []})

        documents = list(files_col.find({"_id": {"$in": all_relevant_ids}}).sort("uploaded_at", -1))
        
        output_docs = []
        for doc in documents:
            category = doc.get("category")
            if not category and doc.get("translation_status") == "translated":
                child = files_col.find_one({"parent_id": doc["_id"]})
                category = child.get("category", "miscellaneous") if child else "miscellaneous"
            
            output_docs.append({
                "meta_id": str(doc["_id"]),
                "filename": doc.get("filename", "unknown"),
                "uploaded_at": doc.get("uploaded_at").isoformat(),
                "category": category or "miscellaneous",
            })
        
        return jsonify({"documents": output_docs})
    except Exception as e:
        logger.error(f"Search by tag failed for {tag}: {traceback.format_exc()}")
        return jsonify({"error": "search_failed", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)

