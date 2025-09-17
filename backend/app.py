import os, io, tempfile, datetime, re, traceback, logging
from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
import requests
import pdfplumber
from docx import Document as DocxDocument
from langdetect import detect, DetectorFactory, LangDetectException
from googletrans import Translator as GTranslator

# Make langdetect deterministic
DetectorFactory.seed = 0

# ---------- CONFIG ----------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "kmrl_docs")
USE_GCLOUD = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Connect to MongoDB & GridFS ----------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)
files_col = db["files"]  # metadata

# ---------- Translators ----------
gtranslator = GTranslator()
gcloud_client = None
if USE_GCLOUD:
    try:
        from google.cloud import translate_v2 as gcloud_translate
        gcloud_client = gcloud_translate.Client()
        logger.info("Google Cloud Translate enabled.")
    except Exception as e:
        gcloud_client = None
        logger.warning(f"Could not init Google Cloud Translate: {str(e)}; will use googletrans fallback.")

# ---------- Helpers ----------
def oid_to_str(o):
    return str(o) if o is not None else None

def save_file_to_gridfs(file_bytes, filename, content_type):
    grid_id = fs.put(file_bytes, filename=filename, contentType=content_type)
    return grid_id

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
    res = files_col.insert_one(doc)
    return res.inserted_id

def get_metadata(meta_id):
    return files_col.find_one({"_id": ObjectId(meta_id)})

def update_metadata(meta_id, patch: dict):
    files_col.update_one({"_id": ObjectId(meta_id)}, {"$set": patch})

# ---------- Download helper (Google Drive and SharePoint) ----------
def download_url_to_bytes(url, timeout=30):
    # Google Drive handling
    gd = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if not gd:
        gd = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if gd:
        fid = gd.group(1)
        dl = f"https://drive.google.com/uc?export=download&id={fid}"
        r = requests.get(dl, timeout=timeout)
        if r.status_code == 200:
            filename = f"drive_{fid}"
            return r.content, filename, r.headers.get("content-type", "application/octet-stream")
        else:
            raise Exception(f"Drive download failed: {r.status_code}")
    
    # SharePoint handling (simplified - may need adjustments for your SharePoint setup)
    if "sharepoint.com" in url:
        # For SharePoint, we need to handle authentication
        # This is a basic implementation - you may need to adjust based on your auth method
        headers = {}
        # Add authentication headers if needed (e.g., Bearer token)
        # headers['Authorization'] = f'Bearer {sharepoint_token}'
        
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            raise Exception(f"SharePoint download failed: {r.status_code}")
        
        # Try to get filename from Content-Disposition header
        content_disposition = r.headers.get('Content-Disposition', '')
        filename_match = re.search(r'filename="([^"]+)"', content_disposition)
        if filename_match:
            filename = filename_match.group(1)
        else:
            filename = url.split("/")[-1].split("?")[0] or "sharepoint_download.bin"
            
        return r.content, filename, r.headers.get("content-type", "application/octet-stream")
    
    # Generic URL handling
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        raise Exception(f"Download failed: {r.status_code}")
    
    # Try to get filename from Content-Disposition header
    content_disposition = r.headers.get('Content-Disposition', '')
    filename_match = re.search(r'filename="([^"]+)"', content_disposition)
    if filename_match:
        filename = filename_match.group(1)
    else:
        filename = url.split("/")[-1].split("?")[0] or "downloaded.bin"
    
    return r.content, filename, r.headers.get("content-type", "application/octet-stream")

# ---------- Extract digital text ONLY (no OCR) ----------
def extract_text_if_digital(file_bytes, filename, filetype_hint=None):
    ext = os.path.splitext(filename.lower())[1]
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        tf.write(file_bytes)
        tf.flush()
        tf.close()
        
        text = ""
        if ext == ".pdf" or (filetype_hint and "pdf" in filetype_hint):
            try:
                with pdfplumber.open(tf.name) as pdf:
                    for p in pdf.pages:
                        t = p.extract_text()
                        if t: 
                            text += t + "\n"
            except Exception as e:
                logger.warning(f"PDF extraction failed: {str(e)}")
                text = ""
        elif ext == ".docx" or (filetype_hint and "word" in (filetype_hint or "")):
            try:
                doc = DocxDocument(tf.name)
                paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
                text = "\n".join(paras)
            except Exception as e:
                logger.warning(f"DOCX extraction failed: {str(e)}")
                text = ""
        elif ext in [".txt", ".text"]:
            try:
                text = file_bytes.decode("utf-8", errors="ignore")
            except Exception as e:
                logger.warning(f"TXT extraction failed: {str(e)}")
                text = ""
        else:
            # Try PDF as fallback for unknown file types
            try:
                with pdfplumber.open(tf.name) as pdf:
                    for p in pdf.pages:
                        t = p.extract_text()
                        if t: 
                            text += t + "\n"
            except Exception:
                text = ""
        
        text = text.strip()
        lang = None
        if text:
            try:
                lang = detect(text)
            except LangDetectException:
                # If langdetect fails, try to determine language based on characters
                if any('\u0D00' <= ch <= '\u0D7F' for ch in text):
                    lang = 'ml'
                else:
                    lang = 'en'  # Default to English if we can't detect
            except Exception as e:
                logger.warning(f"Language detection failed: {str(e)}")
                lang = None
        
        return text, lang
    finally:
        try: 
            os.unlink(tf.name)
        except: 
            pass

# ---------- Malayalam detection ----------
def is_malayalam(text, detected_lang):
    if not text: 
        return False
    
    # First, trust the language detection if it says Malayalam
    if detected_lang == 'ml':
        return True
    
    # If language detection is uncertain, check for Malayalam characters
    malayalam_chars = sum(1 for ch in text if '\u0D00' <= ch <= '\u0D7F')
    total_chars = len(text)
    
    # If more than 10% of characters are Malayalam, consider it Malayalam
    if total_chars > 0 and malayalam_chars / total_chars > 0.1:
        return True
        
    return False

# ---------- Translation ----------
def translate_ml_to_en(text):
    if not text: 
        return ""
    
    # Split text into chunks to avoid API limits
    max_chunk_size = 5000  # Adjust based on API limits
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    translated_chunks = []
    
    for chunk in chunks:
        if gcloud_client:
            try:
                res = gcloud_client.translate(chunk, source_language='ml', target_language='en')
                translated_chunks.append(res.get('translatedText', chunk))
                continue
            except Exception as e:
                logger.warning(f"Google Cloud translation failed: {str(e)}")
        
        try:
            translated = gtranslator.translate(chunk, src='ml', dest='en').text
            translated_chunks.append(translated)
        except Exception as e:
            logger.warning(f"Googletrans translation failed: {str(e)}")
            translated_chunks.append(chunk)  # Fallback to original text
    
    return " ".join(translated_chunks)

# ---------- Processing pipeline ----------
def process_and_store(file_bytes, filename, filetype, uploaded_by, source_type='upload', source_url=None, force_translate=False):
    # First, save the original file
    grid_id = save_file_to_gridfs(file_bytes, filename, filetype)
    meta_id = insert_metadata(filename, grid_id, filetype, uploaded_by, source_type, source_url, None, None, None)
    
    # Extract text and detect language
    extracted, detected_lang = extract_text_if_digital(file_bytes, filename, filetype)
    
    if not extracted:
        update_metadata(meta_id, {"translation_status": "no_extractable_text", "language": None})
        return {"original_meta_id": oid_to_str(meta_id), "translated": False, "reason": "no_extractable_text"}
    
    # Check if the content is Malayalam
    mal = is_malayalam(extracted, detected_lang)
    
    
    if not mal:
        # NEW: CLASSIFY NON-MALAYALAM FILES
        tags, confidence_scores = classify_by_rules(extracted)
        update_document_with_tags(meta_id, tags, confidence_scores)
        
        update_metadata(meta_id, {"translation_status": "not_needed", "language": detected_lang or 'en'})
        return {
            "original_meta_id": oid_to_str(meta_id), 
            "translated": False, 
            "reason": "not_malayalam", 
            "language": detected_lang,
            "tags": tags,  # NEW: Return tags
            "primary_tag": tags[0] if tags else 'miscellaneous'
        }
    

    # If Malayalam, create translation
    base_txt = os.path.splitext(filename)[0] + ".txt"
    existing = files_col.find_one({"parent_id": meta_id, "filename": base_txt})
    
    if existing and not force_translate:
        update_metadata(meta_id, {"translation_status": "translated", "language": detected_lang})
        return {
            "original_meta_id": oid_to_str(meta_id), 
            "translated": True, 
            "translation_meta_id": oid_to_str(existing["_id"]), 
            "note": "already_exists"
        }
    
    try:
        translated_text = translate_ml_to_en(extracted)
    except Exception as e:
        update_metadata(meta_id, {"translation_status": "failed", "language": detected_lang})
        return {"original_meta_id": oid_to_str(meta_id), "translated": False, "reason": "translation_failed", "error": str(e)}
    
    # Save the translation
    translated_bytes = translated_text.encode("utf-8")
    trans_grid_id = save_file_to_gridfs(translated_bytes, base_txt, "text/plain")
    trans_meta_id = insert_metadata(
        base_txt, trans_grid_id, "text/plain", "system_translator", 
        "translation", None, meta_id, "en", "translated"
    )

    
    # NEW: CLASSIFY THE TRANSLATED ENGLISH TEXT
    tags, confidence_scores = classify_by_rules(translated_text)
    update_document_with_tags(trans_meta_id, tags, confidence_scores)
    
    update_metadata(meta_id, {"translation_status": "translated", "language": detected_lang})
    
    return {
        "original_meta_id": oid_to_str(meta_id), 
        "translated": True, 
        "translation_meta_id": oid_to_str(trans_meta_id),
        "tags": tags,  # NEW: Return tags in response
        "primary_tag": tags[0] if tags else 'miscellaneous'
    }

# ---------- Rules-Based Classification ----------
CATEGORIES = {
    'invoices': [
        'invoice', 'bill', 'payment', 'amount', 'total', 'due', 'balance', 
        '$', 'rs', 'usd', 'rupees', 'credit', 'debit', 'tax', 'gst', 'vat',
        'amount due', 'payment received', 'financial', 'transaction', 'receipt',
        'quotation', 'estimate', 'charges', 'fee', 'cost', 'price', 'payment terms'
    ],
    'safety_reports': [
        'safety', 'audit', 'risk', 'hazard', 'compliance', 'regulation', 
        'standard', 'incident', 'accident', 'injury', 'prevention', 'inspection',
        'emergency', 'protocol', 'guideline', 'safety measure', 'risk assessment',
        'hse', 'health safety environment', 'safety procedure', 'safety protocol',
        'safety inspection', 'safety audit', 'risk management', 'hazard analysis'
    ],
    'urgent': [
        'urgent', 'immediate', 'asap', 'critical', 'emergency', 'attention', 
        'important', 'priority', 'time sensitive', 'deadline', 'expedite',
        'rush', 'quick', 'without delay', 'prompt action', 'urgently', 'as soon as possible',
        'time critical', 'high priority', 'urgent attention', 'immediate action'
    ],
    'engineering_drawings': [
        'drawing', 'blueprint', 'cad', 'dimension', 'tolerance', 'specification', 
        'technical', 'design', 'engineering', 'schematics', 'diagram', 'plan',
        'layout', 'mechanical', 'electrical', 'civil', 'architecture', 'drafting',
        'technical drawing', 'engineering diagram', 'construction drawing', 'assembly drawing',
        'detail drawing', 'fabrication drawing', 'installation drawing'
    ],
    'miscellaneous': []  # Catch-all category
}

def classify_by_rules(text):
    """Classify text using keyword rules and return multiple tags"""
    if not text:
        return ['miscellaneous'], {'miscellaneous': 100.0}
    
    text_lower = text.lower()
    category_scores = {}
    
    for category, keywords in CATEGORIES.items():
        if category == 'miscellaneous':
            continue
            
        score = 0
        keyword_matches = []
        
        for keyword in keywords:
            # Count occurrences of each keyword
            occurrences = text_lower.count(keyword.lower())
            if occurrences > 0:
                # Base score for finding the keyword + bonus for multiple occurrences
                score += 10 + (occurrences * 5)
                keyword_matches.append(f"{keyword}({occurrences})")
        
        if score > 0:
            category_scores[category] = score
    
    # Apply threshold and get final tags
    threshold = 15.0  # Minimum score to include a tag
    final_tags = [tag for tag, score in category_scores.items() if score >= threshold]
    
    # Calculate confidence scores (normalized to 0-100)
    confidence_scores = {}
    if category_scores:
        max_score = max(category_scores.values()) if category_scores else 1
        for tag, score in category_scores.items():
            confidence_scores[tag] = min(100, (score / max_score) * 100)
    
    # Always include miscellaneous if no other tags meet threshold
    if not final_tags:
        final_tags = ['miscellaneous']
        confidence_scores['miscellaneous'] = 100.0
    
    # Sort by confidence score (highest first)
    final_tags.sort(key=lambda x: confidence_scores[x], reverse=True)
    
    return final_tags, confidence_scores

def update_document_with_tags(meta_id, tags, confidence_scores):
    """Update document with classification tags"""
    try:
        files_col.update_one(
            {"_id": ObjectId(meta_id)},
            {"$set": {
                "category": tags[0] if tags else 'miscellaneous',
                "tags": tags,
                "confidence_scores": confidence_scores,
                "classification_method": "rules_based",
                "classification_status": "completed",
                "classified_at": datetime.datetime.utcnow()
            }}
        )
        return True
    except Exception as e:
        logger.error(f"Failed to update document with tags: {e}")
        return False
    

# ---------- ROUTES ----------
@app.route("/upload", methods=["POST"])
def upload_route():
    uploaded_by = request.form.get("uploaded_by", "anonymous")
    force_translate = request.form.get("force_translate", "false").lower() == "true"
    
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "no selected file"}), 400
        
    filename = f.filename or "uploaded.bin"
    file_bytes = f.read()
    filetype = f.mimetype or "application/octet-stream"
    
    try:
        res = process_and_store(file_bytes, filename, filetype, uploaded_by, "upload", None, force_translate)
        return jsonify(res), 201
    except Exception as e:
        logger.error(f"Upload processing failed: {traceback.format_exc()}")
        return jsonify({"error": "processing_failed", "details": str(e)}), 500

@app.route("/ingest_link", methods=["POST"])
def ingest_link_route():
    url = request.form.get("url")
    if not url: 
        return jsonify({"error": "no url provided"}), 400
        
    uploaded_by = request.form.get("uploaded_by", "anonymous")
    force_translate = request.form.get("force_translate", "false").lower() == "true"
    
    try:
        file_bytes, filename, filetype = download_url_to_bytes(url)
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return jsonify({"error": "download_failed", "details": str(e)}), 400
    
    try:
        res = process_and_store(file_bytes, filename, filetype, uploaded_by, "link", url, force_translate)
        return jsonify(res), 201
    except Exception as e:
        logger.error(f"Link processing failed: {traceback.format_exc()}")
        return jsonify({"error": "processing_failed", "details": str(e)}), 500

@app.route("/files", methods=["GET"])
def list_files():
    out = []
    for r in files_col.find().sort("uploaded_at", -1).limit(200):
        out.append({
            "meta_id": oid_to_str(r["_id"]),
            "filename": r.get("filename"),
            "content_type": r.get("content_type"),
            "uploaded_by": r.get("uploaded_by"),
            "uploaded_at": r.get("uploaded_at").isoformat() if r.get("uploaded_at") else None,
            "source_type": r.get("source_type"),
            "source_url": r.get("source_url"),
            "parent_id": oid_to_str(r.get("parent_id")),
            "language": r.get("language"),
            "translation_status": r.get("translation_status")
        })
    return jsonify(out)

@app.route("/download/<meta_id>", methods=["GET"])
def download(meta_id):
    try:
        meta = files_col.find_one({"_id": ObjectId(meta_id)})
        if not meta: 
            return jsonify({"error": "not_found"}), 404
            
        grid_id = meta.get("gridfs_id")
        if not grid_id: 
            return jsonify({"error": "gridfs_id_missing"}), 500
            
        gf = fs.get(grid_id)
        data = gf.read()
        
        return send_file(
            io.BytesIO(data), 
            mimetype=meta.get("content_type") or "application/octet-stream", 
            as_attachment=True, 
            download_name=meta.get("filename")
        )
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return jsonify({"error": "download_failed", "details": str(e)}), 500

@app.route("/reprocess/<meta_id>", methods=["POST"])
def reprocess(meta_id):
    force = request.args.get("force", "false").lower() == "true"
    
    try:
        meta = files_col.find_one({"_id": ObjectId(meta_id)})
        if not meta: 
            return jsonify({"error": "not_found"}), 404
            
        gf = fs.get(meta["gridfs_id"])
        file_bytes = gf.read()
        
        res = process_and_store(
            file_bytes, meta["filename"], meta.get("content_type"), 
            meta.get("uploaded_by"), meta.get("source_type"), 
            meta.get("source_url"), force
        )
        
        return jsonify(res)
    except Exception as e:
        logger.error(f"Reprocessing failed: {traceback.format_exc()}")
        return jsonify({"error": "processing_failed", "details": str(e)}), 500

@app.route("/ocr_text/<meta_id>", methods=["POST"])
def ocr_text_route(meta_id):
    try:
        payload = request.get_json(force=True)
        text = payload.get("ocr_text", "")
        
        if not text: 
            return jsonify({"error": "no ocr_text provided"}), 400
            
        force = bool(payload.get("force", False))
        
        try:
            lang = detect(text)
        except LangDetectException:
            # If langdetect fails, check for Malayalam characters
            if any('\u0D00' <= ch <= '\u0D7F' for ch in text):
                lang = 'ml'
            else:
                lang = 'en'
        except Exception:
            lang = None
        
        mal = (lang == 'ml') or any('\u0D00' <= ch <= '\u0D7F' for ch in text)
        
        if not mal: 
            return jsonify({"translated": False, "reason": "not_malayalam", "detected_lang": lang})
        
        parent_meta = files_col.find_one({"_id": ObjectId(meta_id)})
        if not parent_meta: 
            return jsonify({"error": "parent_not_found"}), 404
            
        base_txt = os.path.splitext(parent_meta["filename"])[0] + ".txt"
        existing = files_col.find_one({"parent_id": parent_meta["_id"], "filename": base_txt})
        
        if existing and not force:
            return jsonify({
                "translated": True, 
                "translation_meta_id": oid_to_str(existing["_id"]), 
                "note": "already_exists"
            })
        
        translated = translate_ml_to_en(text)
        tbytes = translated.encode("utf-8")
        tgrid = save_file_to_gridfs(tbytes, base_txt, "text/plain")
        
        tmeta_id = insert_metadata(
            base_txt, tgrid, "text/plain", 
            payload.get("uploader", "ocr_user"), "translation", 
            None, parent_meta["_id"], "en", "translated"
        )
        
        update_metadata(parent_meta["_id"], {"translation_status": "translated"})
        
        return jsonify({"translated": True, "translation_meta_id": oid_to_str(tmeta_id)})
    except Exception as e:
        logger.error(f"OCR translation failed: {traceback.format_exc()}")
        return jsonify({"error": "ocr_translation_failed", "details": str(e)}), 500
    
# ---------- Classification Routes ----------
@app.route("/classify/<meta_id>", methods=["POST"])
def classify_document(meta_id):
    """Classify a specific document using rules"""
    try:
        # Get document metadata
        doc = files_col.find_one({"_id": ObjectId(meta_id)})
        if not doc:
            return jsonify({"error": "Document not found"}), 404
        
        # Get text content (you'll need to implement GridFS retrieval)
        # For now, this is a placeholder - implement based on your GridFS setup
        text = "Implement GridFS text retrieval here"
        
        if not text:
            return jsonify({"error": "Could not retrieve document text"}), 400
        
        # Perform classification
        tags, confidence_scores = classify_by_rules(text)
        
        # Update database
        success = update_document_with_tags(meta_id, tags, confidence_scores)
        
        if success:
            return jsonify({
                "success": True,
                "meta_id": meta_id,
                "tags": tags,
                "confidence_scores": confidence_scores,
                "classification_method": "rules_based"
            }), 200
        else:
            return jsonify({"error": "Failed to update database"}), 500
            
    except Exception as e:
        return jsonify({"error": f"Classification error: {str(e)}"}), 500

@app.route("/search-by-tag/<tag>", methods=["GET"])
def search_by_tag(tag):
    """Search for documents with a specific tag"""
    try:
        documents = list(files_col.find({
            "tags": tag,
            "classification_status": "completed"
        }).sort("classified_at", -1))
        
        return jsonify({
            "success": True,
            "tag": tag,
            "count": len(documents),
            "documents": [{
                "meta_id": str(doc["_id"]),
                "filename": doc.get("filename", "unknown"),
                "confidence": doc.get("confidence_scores", {}).get(tag, 0),
                "classified_at": doc.get("classified_at")
            } for doc in documents]
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/classification-status", methods=["GET"])
def classification_status():
    """Get classification statistics"""
    try:
        total_files = files_col.count_documents({})
        classified_files = files_col.count_documents({"classification_status": "completed"})
        
        # Get tag distribution
        pipeline = [
            {"$match": {"classification_status": "completed"}},
            {"$unwind": "$tags"},
            {"$group": {"_id": "$tags", "count": {"$sum": 1}}}
        ]
        tag_stats = list(files_col.aggregate(pipeline))
        
        return jsonify({
            "success": True,
            "total_files": total_files,
            "classified_files": classified_files,
            "completion_rate": (classified_files / total_files * 100) if total_files > 0 else 0,
            "tag_distribution": {stat["_id"]: stat["count"] for stat in tag_stats}
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)