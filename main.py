from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import cv2
import numpy as np
import requests
import json
from io import BytesIO
import threading

try:
    import qrcode
    QR_CODE_AVAILABLE = True
except ImportError:
    QR_CODE_AVAILABLE = False
    print("WARNING: qrcode module not available. Install with: pip install qrcode[pil]")

from insightface.app import FaceAnalysis
from cloudinary_delete import delete_asset

app = FastAPI(title="Smart Photo Sharing AI Backend")

# ================= CORS MIDDLEWARE =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- FACE ENGINE LAZY LOADER ----------------
face_app = None
face_app_lock = threading.Lock()

def get_face_app():
    global face_app
    with face_app_lock:
        if face_app is None:
            print("Loading InsightFace model (buffalo_s)...")
            # Use 'buffalo_s' (Small) model for lower memory usage (Crucial for Render Free Tier)
            face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=-1, det_size=(320, 320))
            print("InsightFace model loaded successfully.")
    return face_app

# Startup Event to verify imports and Log Memory (Optional but helpful)
@app.on_event("startup")
async def startup_event():
    print("Backend starting up... Checking libraries.")
    try:
        import cv2
        print(f"OpenCV Version: {cv2.__version__}")
        import onnxruntime
        print(f"ONNX Runtime Version: {onnxruntime.__version__}")
    except Exception as e:
        print(f"Startup Import Error: {e}")

# ================= REQUEST MODELS =================

class EnrollRequest(BaseModel):
    imageUrl: str
    uid: str


class MatchRequest(BaseModel):
    imageUrl: str
    embeddings: list  # [{"uid": "...", "embedding": [...]}]


class CloudinaryDeleteRequest(BaseModel):
    publicId: str
    resourceType: str = "image"


class QRCodeRequest(BaseModel):
    eventId: str


# ================= HEALTH =================

@app.get("/health")
def health():
    return {"status": "ok"}


# ================= QR CODE GENERATION =================

@app.post("/qr/generate")
async def generate_qr(req: QRCodeRequest):
    """Generate QR code for event joining"""
    if not QR_CODE_AVAILABLE:
        return {"success": False, "error": "QR code generation not available. Install with: pip install qrcode[pil]"}
    
    try:
        # Create QR data
        qr_data = {
            "type": "JOIN_EVENT",
            "eventId": req.eventId
        }
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=2,
        )
        qr.add_data(json.dumps(qr_data))
        qr.make(fit=True)
        
        # Create image
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to bytes
        img_io = BytesIO()
        img.save(img_io, format='PNG')
        img_io.seek(0)
        
        return StreamingResponse(img_io, media_type="image/png")
    
    except Exception as e:
        return {"success": False, "error": str(e)}


# ================= FACE ENROLL =================

@app.post("/face/enroll")
async def enroll_face(req: EnrollRequest):
    print(f"Enroll request received for UID: {req.uid}")
    try:
        print(f"Downloading image: {req.imageUrl}")
        response = requests.get(req.imageUrl)
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        print("Detecting faces...")
        faces = get_face_app().get(img)

        if not faces:
            return {"success": False, "message": "No face detected"}

        # Take first detected face
        embedding = faces[0].embedding.tolist()

        return {
            "success": True,
            "uid": req.uid,
            "embedding": embedding,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ================= FACE MATCH =================

@app.post("/face/match")
async def match_face(req: MatchRequest):
    try:
        response = requests.get(req.imageUrl)
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        faces = get_face_app().get(img)

        if not faces:
            return {"success": True, "matches": []}

        results = []

        for face in faces:
            face_emb = face.embedding

            for user in req.embeddings:
                db_emb = np.array(user["embedding"], dtype=np.float32)

                sim = np.dot(face_emb, db_emb) / (
                    np.linalg.norm(face_emb) * np.linalg.norm(db_emb)
                )

                # Similarity threshold
                if sim > 0.45:
                    results.append({
                        "uid": user["uid"],
                        "score": float(sim)
                    })

        return {"success": True, "matches": results}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ================= CLOUDINARY DELETE =================

@app.post("/cloudinary/delete")
def cloudinary_delete(req: CloudinaryDeleteRequest):
    try:
        result = delete_asset(req.publicId, req.resourceType)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
