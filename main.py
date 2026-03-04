import os
import json
import gc
import time
import requests
import numpy as np
import face_recognition
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from contextlib import asynccontextmanager

# ----------------------------------------------------------
# CONFIG — Railway Environment Variables থেকে নেবে
# ----------------------------------------------------------
DB_PATH            = "face_database.json"
TOLERANCE          = 0.55
MAX_IMAGE_SIZE     = (800, 800)
WP_BASE_URL        = os.getenv("WP_BASE_URL",        "https://modelsaler.com")
WC_CONSUMER_KEY    = os.getenv("WC_CONSUMER_KEY",    "")
WC_CONSUMER_SECRET = os.getenv("WC_CONSUMER_SECRET", "")
BUILD_SECRET       = os.getenv("BUILD_SECRET",       "build123")

face_db      = []
build_status = {"running": False, "progress": "", "done": False, "error": ""}

# ----------------------------------------------------------
# LOAD DATABASE INTO MEMORY
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
        faces = product.get("faces", [])
        encodings = [np.array(face["encoding"]) for face in faces]
        
        # ফিক্স: ডেটাবেস থেকে ইমেজের লিংকটা বের করে আনা হচ্ছে
        image_url = faces[0]["source_image"] if faces and "source_image" in faces[0] else ""

        if encodings:
            face_db.append({
                "product_id":   product["product_id"],
                "product_name": product["product_name"],
                "product_url":  product["product_url"],
                "image_url":    image_url, # <-- ইমেজের লিংক মেমরিতে সেভ করা হলো
                "encodings":    encodings,
            })
    gc.collect()
    print(f"Database loaded: {len(face_db)} models found.")

# ----------------------------------------------------------
# BUILD DATABASE ON RAILWAY SERVER
# ----------------------------------------------------------
def build_database_task():
    global build_status, face_db
    build_status = {"running": True, "progress": "Starting...", "done": False, "error": ""}
    try:
        # Step 1: Fetch all WooCommerce products
        build_status["progress"] = "Fetching products from WooCommerce..."
        all_products = []
        page = 1
        while True:
            params = {
                "consumer_key":    WC_CONSUMER_KEY,
                "consumer_secret": WC_CONSUMER_SECRET,
                "per_page": 20,
                "page":     page,
                "status":   "publish",
                "_fields":  "id,name,slug,permalink,images",
            }
            resp = requests.get(
                f"{WP_BASE_URL}/wp-json/wc/v3/products",
                params=params, timeout=30
            )
            resp.raise_for_status()
            products = resp.json()
            if not products:
                break
            all_products.extend(products)
            build_status["progress"] = f"Fetched {len(all_products)} products..."
            page += 1
            time.sleep(0.3)

        total = len(all_products)
        build_status["progress"] = f"Found {total} products. Building face encodings..."

        # Step 2: Load existing to support resume
        existing = []
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        processed_ids = {e["product_id"] for e in existing}
        database = list(existing)

        # Step 3: Process each product
        for i, product in enumerate(all_products):
            pid  = product["id"]
            purl = product["permalink"]
            imgs = product.get("images", [])

            build_status["progress"] = f"Processing {i+1}/{total}: {product['name']}"

            if pid in processed_ids or not imgs:
                continue

            product_faces = []
            for img_data in imgs:
                img_url = img_data.get("src", "")
                if not img_url:
                    continue
                try:
                    r = requests.get(img_url, timeout=20)
                    r.raise_for_status()
                    pil = Image.open(BytesIO(r.content))
                    if pil.mode != "RGB":
                        pil = pil.convert("RGB")
                    pil.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
                    arr = np.array(pil)
                    del pil
                    locs = face_recognition.face_locations(arr, model="hog")
                    if locs:
                        for enc in face_recognition.face_encodings(arr, locs):
                            product_faces.append({"source_image": img_url, "encoding": enc.tolist()})
                    del arr
                    gc.collect()
                except Exception:
                    continue
                time.sleep(0.2)

            database.append({
                "product_id":   pid,
                "product_name": product["name"],
                "product_slug": product["slug"],
                "product_url":  purl,
                "face_count":   len(product_faces),
                "faces":        product_faces
            })

            # Auto-save every 10 products
            if len(database) % 10 == 0:
                with open(DB_PATH, "w", encoding="utf-8") as f:
                    json.dump(database, f, ensure_ascii=False)

        # Step 4: Final save & reload
        with open(DB_PATH, "w", encoding="utf-8") as f:
            json.dump(database, f, ensure_ascii=False)
        load_database()

        build_status = {
            "running": False,
            "progress": f"Done! {total} products processed. {len(face_db)} models with faces loaded.",
            "done":  True,
            "error": ""
        }

    except Exception as e:
        build_status = {"running": False, "progress": "Failed.", "done": False, "error": str(e)}

# ----------------------------------------------------------
# APP
# ----------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_database()
    yield
    face_db.clear()

app = FastAPI(title="Visual Face Search API", version="2.0.0", lifespan=lifespan)

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
    return {"status": "running", "models_loaded": len(face_db), "message": "Visual Face Search API is live!"}

@app.get("/health")
def health():
    return {"status": "ok", "models": len(face_db)}

@app.get("/build-db")
def start_build(secret: str, background_tasks: BackgroundTasks):
    """Browser থেকে একবার call করলেই Railway নিজে database বানাবে"""
    if secret != BUILD_SECRET:
        raise HTTPException(status_code=403, detail="Wrong secret key.")
    if build_status["running"]:
        return {"status": "already_running", "progress": build_status["progress"]}
    if not WC_CONSUMER_KEY:
        raise HTTPException(status_code=500, detail="WC_CONSUMER_KEY not set in Variables.")
    background_tasks.add_task(build_database_task)
    return {"status": "started", "message": "Building... check /build-status for progress"}

@app.get("/build-status")
def get_build_status():
    """Database তৈরির progress দেখুন"""
    return {
        "running":       build_status["running"],
        "progress":      build_status["progress"],
        "done":          build_status["done"],
        "error":         build_status["error"],
        "models_loaded": len(face_db),
    }

@app.post("/search")
async def search_face(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be under 5MB.")
    try:
        pil_img = Image.open(BytesIO(contents))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pil_img.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
        img_array = np.array(pil_img)
        del pil_img, contents
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process image.")
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

    if best_match:
        confidence = round((1 - best_distance) * 100, 1)
        return {
            "found":        True,
            "product_id":   best_match["product_id"],
            "product_name": best_match["product_name"],
            "product_url":  best_match["product_url"],
            "image_url":    best_match.get("image_url", ""), # <-- ফিক্স: ইমেজের লিংক রেজাল্টে পাঠানো হচ্ছে
            "confidence":   confidence,
        }
    return {"found": False, "message": "No matching model found."}

@app.post("/reload-db")
def reload_db(secret: str = ""):
    if secret != os.getenv("RELOAD_SECRET", "changeme"):
        raise HTTPException(status_code=403, detail="Unauthorized.")
    load_database()
    return {"status": "ok", "models": len(face_db)}
