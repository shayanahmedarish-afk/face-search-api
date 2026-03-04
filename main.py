import os
import json
import gc
import numpy as np
import face_recognition
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from contextlib import asynccontextmanager

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
DB_PATH        = "face_database.json"
TOLERANCE      = 0.55
MAX_IMAGE_SIZE = (800, 800)

face_db = []

# ----------------------------------------------------------
# LOAD DATABASE INTO MEMORY (runs once on startup)
# ----------------------------------------------------------
def load_database():
    global face_db
    if not os.path.exists(DB_PATH):
        print("WARNING: face_database.json not found!")
        return

    with open(DB_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    face_db = []
    for product in raw:
        encodings = [np.array(face["encoding"]) for face in product.get("faces", [])]
        if encodings:
            face_db.append({
                "product_id":   product["product_id"],
                "product_name": product["product_name"],
                "product_url":  product["product_url"],
                "encodings":    encodings,
            })

    gc.collect()
    print(f"Database loaded: {len(face_db)} models found.")

# ----------------------------------------------------------
# APP STARTUP & SHUTDOWN
# ----------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_database()
    yield
    face_db.clear()

app = FastAPI(
    title="Visual Face Search API",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from your WordPress site
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://modelsaler.com", "https://www.modelsaler.com"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ----------------------------------------------------------
# ROUTES
# ----------------------------------------------------------

@app.get("/")
def root():
    return {
        "status":        "running",
        "models_loaded": len(face_db),
        "message":       "Visual Face Search API is live!"
    }


@app.get("/health")
def health():
    return {"status": "ok", "models": len(face_db)}


@app.post("/search")
async def search_face(file: UploadFile = File(...)):
    """
    Upload an image → returns matching model info
    """

    # 1. Check file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    # 2. Read image (max 5MB)
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be under 5MB.")

    # 3. Process image (memory-safe)
    try:
        pil_img = Image.open(BytesIO(contents))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pil_img.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
        img_array = np.array(pil_img)
        del pil_img, contents
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process image.")

    # 4. Detect face in uploaded image
    try:
        locations = face_recognition.face_locations(img_array, model="hog")
        if not locations:
            del img_array
            gc.collect()
            return {"found": False, "message": "No face detected in the uploaded image."}

        query_encodings = face_recognition.face_encodings(img_array, locations)
        del img_array
    except Exception:
        raise HTTPException(status_code=500, detail="Face detection failed.")

    # 5. Compare with database
    best_match    = None
    best_distance = 1.0

    for product in face_db:
        for q_enc in query_encodings:
            distances = face_recognition.face_distance(product["encodings"], q_enc)
            min_dist  = float(np.min(distances))
            if min_dist < TOLERANCE and min_dist < best_distance:
                best_distance = min_dist
                best_match    = product

    gc.collect()

    # 6. Return result
    if best_match:
        confidence = round((1 - best_distance) * 100, 1)
        return {
            "found":        True,
            "product_id":   best_match["product_id"],
            "product_name": best_match["product_name"],
            "product_url":  best_match["product_url"],
            "confidence":   confidence,
        }

    return {"found": False, "message": "No matching model found."}


@app.post("/reload-db")
def reload_db(secret: str = ""):
    """Reload face_database.json without restarting the server"""
    if secret != os.getenv("RELOAD_SECRET", "changeme"):
        raise HTTPException(status_code=403, detail="Unauthorized.")
    load_database()
    return {"status": "ok", "models": len(face_db)}
