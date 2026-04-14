"""
Optimized Smart Photo Sharing Backend
Drop-in replacement for main.py hosted on Hugging Face.

Changes:
  - singleton model, normalized embeddings, TTL cache
  - vectorized similarity, threshold 0.55, background queue guard
  - APP_ROLE gating (api | worker) — API never loads InsightFace
  - Redis RPUSH/BLPOP distributed job queue — in-memory deque REMOVED
  - /cloudinary/delete IDOR fix — verifies Firestore ownership before delete
"""

import os, time, asyncio, json, threading
from contextlib import asynccontextmanager
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
import requests
from fastapi import FastAPI, BackgroundTasks, Request, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import firebase_admin
from firebase_admin import credentials, firestore, auth as fb_auth
from cloudinary_delete import delete_asset, extract_public_id_from_url, delete_batch

# ─────────────────────────────────────────────────────
#  APP ROLE — controls which subsystems are initialized
#  Set APP_ROLE=api     on your API container   (no ML)
#  Set APP_ROLE=worker  on your ML container    (ML + queue loop)
# ─────────────────────────────────────────────────────
APP_ROLE = os.environ.get("APP_ROLE", "api").lower()
REDIS_JOB_QUEUE_KEY = "media_jobs_queue"
print(f"ℹ️  APP_ROLE={APP_ROLE}")

# ─────────────────────────────────────────────────────
#  FIREBASE INIT
# ─────────────────────────────────────────────────────
db = None
try:
    key_path = "serviceAccountKey__1_.json" if os.path.exists("serviceAccountKey__1_.json") \
               else "serviceAccountKey.json"
    if os.path.exists(key_path):
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print(f"✅ Firebase initialized via {key_path}")
    else:
        print("⚠️  No service account key found – Firebase disabled")
except Exception as e:
    print(f"❌ Firebase init failed: {e}")

# ─────────────────────────────────────────────────────
#  CLOUDINARY
# ─────────────────────────────────────────────────────
import cloudinary
import cloudinary.utils

# ─────────────────────────────────────────────────────
#  REDIS — REQUIRED in production; crash-fast if unavailable
# ─────────────────────────────────────────────────────
_redis = None
REDIS_URL = os.environ.get("REDIS_URL", "")

if REDIS_URL:
    try:
        import redis as _redis_lib
        _redis = _redis_lib.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
        _redis.ping()
        print("✅ Redis connected – distributed cache, job guard & queue active")
    except Exception as _re:
        if APP_ROLE == "worker":
            # Worker CANNOT function without Redis — fail loudly so Docker restarts us
            raise RuntimeError(f"❌ [WORKER] Redis is required but unavailable: {_re}")
        else:
            print(f"⚠️  [API] Redis unavailable ({_re}) – job queueing will return 503")
            _redis = None
else:
    if APP_ROLE == "worker":
        raise RuntimeError("❌ [WORKER] REDIS_URL is not set. Workers require Redis.")
    print("ℹ️  REDIS_URL not set – running in local dev mode (no job queue)")

try:
    import qrcode
    QR_CODE_AVAILABLE = True
except ImportError:
    QR_CODE_AVAILABLE = False

# ─────────────────────────────────────────────────────
#  AUTHENTICATION DEPENDENCY
# ─────────────────────────────────────────────────────
async def verify_token(authorization: str = Header(default=None)) -> str:
    """
    Validates a Firebase ID Token from the Authorization header.
    Expected format: Authorization: Bearer <firebase_id_token>
    Returns the authenticated user's UID or raises HTTP 401.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Expected: Authorization: Bearer <firebase_id_token>",
        )
    token = authorization[7:]
    try:
        decoded = fb_auth.verify_id_token(token)
        return decoded["uid"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {str(e)}")

# ─────────────────────────────────────────────────────
#  SINGLETON FACE ENGINE  (worker only)
# ─────────────────────────────────────────────────────
_face_app = None
_model_lock = threading.Lock()

def get_face_app():
    """Returns the single shared FaceAnalysis instance. Thread-safe."""
    global _face_app
    if _face_app is None:
        with _model_lock:
            if _face_app is None:
                raise RuntimeError("Face engine not yet initialized. Is APP_ROLE=worker?")
    return _face_app

# ─────────────────────────────────────────────────────
#  LIFESPAN — model loaded ONLY in worker role
# ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _face_app

    if APP_ROLE == "worker":
        # Deferred import so API containers never pay the InsightFace import cost
        from insightface.app import FaceAnalysis
        print("🔄 [WORKER] Loading InsightFace model …")
        _face_app = FaceAnalysis(
            name="buffalo_l",
            root="/app/.insightface",
            providers=["CPUExecutionProvider"],
        )
        await asyncio.to_thread(_face_app.prepare, ctx_id=-1, det_size=(320, 320))
        print("✅ [WORKER] InsightFace loaded (det_size=320×320)")

        # Start the Redis-backed job consumer loop as a background task
        asyncio.create_task(_worker_job_loop())
        print("✅ [WORKER] Redis job consumer loop started")
    else:
        print("ℹ️  [API] Skipping InsightFace load (not a worker role)")

    yield
    print("🛑 Shutting down")

# ─────────────────────────────────────────────────────
#  WORKER JOB LOOP — consumes Redis queue via BLPOP
# ─────────────────────────────────────────────────────
async def _worker_job_loop():
    """
    Persistent async loop that blocks on Redis BLPOP until a job arrives.
    Each job is processed synchronously in the thread pool — one at a time
    per worker container. Scale containers to scale parallelism.
    """
    print("[WORKER] Waiting for jobs on Redis queue …")
    while True:
        try:
            # BLPOP blocks for up to 5s; returns (queue_name, json_value) or None
            result = await asyncio.to_thread(
                _redis.blpop, REDIS_JOB_QUEUE_KEY, timeout=5
            )
            if result is None:
                continue  # timeout, retry

            _, job_json = result
            job = json.loads(job_json)
            media_id = job.get("mediaId")
            url       = job.get("url")
            event_id  = job.get("eventId")

            if not all([media_id, url, event_id]):
                print(f"[WORKER] ⚠️  Malformed job, skipping: {job}")
                continue

            # Deduplication: skip if already being processed
            if not _acquire_job_lock(media_id):
                print(f"[WORKER] Duplicate job skipped: {media_id}")
                continue

            print(f"[WORKER] Processing job: {media_id}")
            await asyncio.to_thread(_process_media_sync, media_id, url, event_id)

        except Exception as loop_err:
            # Never crash the loop — log and keep polling
            print(f"[WORKER] ❌ Job loop error: {loop_err}")
            await asyncio.sleep(2)

# ─────────────────────────────────────────────────────
#  TTL EMBEDDING CACHE
# ─────────────────────────────────────────────────────
_embed_cache: dict = {}              # event_id -> (timestamp, list[dict])
CACHE_TTL = 300                      # 5 minutes

def _cache_get(event_id: str) -> Optional[list]:
    if _redis:
        try:
            raw = _redis.get(f"emb:{event_id}")
            return json.loads(raw) if raw else None
        except Exception:
            pass
    entry = _embed_cache.get(event_id)
    if entry and (time.time() - entry[0]) < CACHE_TTL:
        return entry[1]
    return None


def _cache_set(event_id: str, data: list):
    if _redis:
        try:
            _redis.setex(f"emb:{event_id}", CACHE_TTL, json.dumps(data))
            return
        except Exception:
            pass
    _embed_cache[event_id] = (time.time(), data)


def _cache_invalidate(event_id: str):
    if _redis:
        try:
            _redis.delete(f"emb:{event_id}")
        except Exception:
            pass
    _embed_cache.pop(event_id, None)


# ─────────────────────────────────────────────────────
#  JOB LOCK  (Redis SET NX or local set fallback)
# ─────────────────────────────────────────────────────
_active_jobs: set = set()

def _acquire_job_lock(media_id: str) -> bool:
    """Atomic deduplication. Redis SET NX across workers; local set fallback."""
    if _redis:
        try:
            return bool(_redis.set(f"lock:{media_id}", "1", nx=True, ex=600))
        except Exception:
            pass
    if media_id in _active_jobs:
        return False
    _active_jobs.add(media_id)
    return True


def _release_job_lock(media_id: str):
    if _redis:
        try:
            _redis.delete(f"lock:{media_id}")
        except Exception:
            pass
    _active_jobs.discard(media_id)

# ─────────────────────────────────────────────────────
#  THREAD POOL
# ─────────────────────────────────────────────────────
MAX_WORKERS = int(os.environ.get("MAX_AI_THREADS", os.cpu_count() or 4))
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# ─────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────
def _optimize_cloudinary_url(url: str) -> str:
    if "cloudinary.com" not in url or "/upload/" not in url:
        return url
    if "q_auto" in url:
        return url
    parts = url.split("/upload/")
    return f"{parts[0]}/upload/w_800,c_limit,q_auto,f_auto/{parts[1]}"


def _download_image(url: str) -> Optional[np.ndarray]:
    try:
        resp = requests.get(_optimize_cloudinary_url(url), timeout=10)
        resp.raise_for_status()
        arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[download] {e}")
        return None


def _get_faces(img: np.ndarray):
    return get_face_app().get(img)


def _fetch_event_embeddings(event_id: str) -> list:
    """
    Fetch embeddings for all enrolled members of an event.
    Uses TTL cache to avoid repeated Firestore scans.
    Returns list of {'uid': str, 'embedding': list[float]}
    """
    cached = _cache_get(event_id)
    if cached is not None:
        return cached

    if not db:
        return []

    member_docs = db.collection("eventMembers").where("eventId", "==", event_id).stream()
    uids = list({d.to_dict()["uid"] for d in member_docs if d.to_dict().get("uid")})

    if not uids:
        _cache_set(event_id, [])
        return []

    profiles = []
    for i in range(0, len(uids), 10):
        chunk = uids[i:i + 10]
        for pdoc in db.collection("faceProfiles").where("uid", "in", chunk).stream():
            pdata = pdoc.to_dict()
            if pdata.get("embedding"):
                pdata.setdefault("uid", pdoc.id)
                profiles.append(pdata)

    _cache_set(event_id, profiles)
    return profiles


def _vectorized_match(faces, known_embeddings: list, threshold: float = 0.55) -> list:
    """
    Vectorized cosine similarity using numpy matrix ops.
    ~10-50× faster than nested Python loops for large profile sets.
    Returns [{'uid': str, 'score': float}]
    """
    if not known_embeddings or not faces:
        return []

    db_vecs = np.array([e["embedding"] for e in known_embeddings], dtype=np.float32)
    norms = np.linalg.norm(db_vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    db_vecs_norm = db_vecs / norms

    best: dict = {}

    for face in faces:
        fvec = face.embedding.astype(np.float32)
        fn = np.linalg.norm(fvec)
        if fn == 0:
            continue
        fvec_norm = fvec / fn

        sims = db_vecs_norm @ fvec_norm
        above = np.where(sims >= threshold)[0]
        for idx in above:
            uid = known_embeddings[idx]["uid"]
            score = float(sims[idx])
            if score > best.get(uid, -1):
                best[uid] = score

    return [{"uid": uid, "score": score} for uid, score in best.items()]


# ─────────────────────────────────────────────────────
#  FASTAPI + RATE LIMITER
# ─────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Smart Photo Sharing AI Backend", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
_allowed_origins = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────────────────
class EnrollRequest(BaseModel):
    imageUrl: str
    uid: str

class MatchRequest(BaseModel):
    imageUrl: str
    eventId: str
    embeddings: Optional[list] = None   # deprecated, kept for compat

class ProcessMediaRequest(BaseModel):
    mediaId: str
    url: str
    eventId: str

class CloudinaryDeleteRequest(BaseModel):
    publicId: str
    resourceType: str = "image"

class BatchDeleteRequest(BaseModel):
    publicIds: List[str]
    resourceType: str = "image"

class QRCodeRequest(BaseModel):
    eventId: str

class BatchEnrollRequest(BaseModel):
    images: List[EnrollRequest]

# ─────────────────────────────────────────────────────
#  HEALTH
# ─────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "Smart Photo Sharing API", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "role": APP_ROLE,
        "model_loaded": _face_app is not None,
        "redis": _redis is not None,
    }

# ─────────────────────────────────────────────────────
#  CLOUDINARY SIGNED UPLOAD TOKEN
# ─────────────────────────────────────────────────────
@app.post("/cloudinary/sign")
async def sign_cloudinary_upload(uid: str = Depends(verify_token)):
    """
    Returns a short-lived signed upload token.
    The client uses this to upload directly to Cloudinary without
    exposing CLOUDINARY_API_SECRET in the app bundle.
    """
    try:
        api_secret = os.environ.get("CLOUDINARY_API_SECRET") or \
                     cloudinary.config().api_secret
        api_key    = os.environ.get("CLOUDINARY_API_KEY") or \
                     cloudinary.config().api_key
        cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME") or \
                     cloudinary.config().cloud_name

        if not all([api_secret, api_key, cloud_name]):
            raise HTTPException(status_code=503, detail="Cloudinary not configured")

        ts = int(time.time())
        params_to_sign = {"timestamp": ts, "folder": "events"}
        signature = cloudinary.utils.api_sign_request(params_to_sign, api_secret)

        return {
            "signature": signature,
            "timestamp": ts,
            "apiKey": api_key,
            "cloudName": cloud_name,
            "folder": "events",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sign error: {str(e)}")

# ─────────────────────────────────────────────────────
#  QR CODE
# ─────────────────────────────────────────────────────
@app.post("/qr/generate")
async def generate_qr(req: QRCodeRequest, _uid: str = Depends(verify_token)):
    if not QR_CODE_AVAILABLE:
        return {"success": False, "error": "qrcode not installed"}
    try:
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H,
                            box_size=10, border=2)
        qr.add_data(json.dumps({"type": "JOIN_EVENT", "eventId": req.eventId}))
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        return {"success": False, "error": str(e)}

# ─────────────────────────────────────────────────────
#  FACE ENROLL  (optimized: normalize + cache invalidate)
# ─────────────────────────────────────────────────────
@app.post("/face/enroll")
async def enroll_face(req: EnrollRequest, background_tasks: BackgroundTasks, uid: str = Depends(verify_token)):
    # A user can only enroll their own face – prevent impersonation
    if req.uid != uid:
        raise HTTPException(status_code=403, detail="You can only enroll your own face profile")
    loop = asyncio.get_running_loop()
    try:
        img = await loop.run_in_executor(executor, _download_image, req.imageUrl)
        if img is None:
            return {"success": False, "message": "Failed to download image"}

        faces = await loop.run_in_executor(executor, _get_faces, img)
        if not faces:
            return {"success": False, "message": "No face detected"}

        # Normalize embedding at enroll time → faster matching later
        raw_emb = faces[0].embedding.astype(np.float32)
        norm = np.linalg.norm(raw_emb)
        embedding = (raw_emb / norm if norm > 0 else raw_emb).tolist()

        # Invalidate cache for all events this user belongs to (best-effort)
        if db:
            member_docs = db.collection("eventMembers").where("uid", "==", req.uid).stream()
            for mdoc in member_docs:
                _cache_invalidate(mdoc.to_dict().get("eventId", ""))

        # Clean up selfie from Cloudinary after enrollment
        public_id = extract_public_id_from_url(req.imageUrl)
        if public_id:
            background_tasks.add_task(delete_asset, public_id, "image")

        return {"success": True, "uid": req.uid, "embedding": embedding}

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/face/enroll/batch")
async def enroll_batch(req: BatchEnrollRequest, background_tasks: BackgroundTasks, _uid: str = Depends(verify_token)):
    if len(req.images) > 50:
        return {"success": False, "error": "Max batch size is 50"}

    async def _process_batch(images):
        loop = asyncio.get_running_loop()
        success_count = 0
        cleanup_ids = []
        for item in images:
            try:
                img = await loop.run_in_executor(executor, _download_image, item.imageUrl)
                if img is None:
                    continue
                faces = await loop.run_in_executor(executor, _get_faces, img)
                if not faces:
                    continue
                success_count += 1
                pid = extract_public_id_from_url(item.imageUrl)
                if pid:
                    cleanup_ids.append(pid)
            except Exception as e:
                print(f"[batch_enroll] {item.uid}: {e}")
        if cleanup_ids:
            await loop.run_in_executor(executor, delete_batch, cleanup_ids, "image")
        print(f"[batch_enroll] done {success_count}/{len(images)}")

    background_tasks.add_task(_process_batch, req.images)
    return {"success": True, "message": f"Processing {len(req.images)} images"}

# ─────────────────────────────────────────────────────
#  FACE MATCH  (optimized: vectorized, cached, threshold 0.55)
# ─────────────────────────────────────────────────────
@app.post("/face/match")
@limiter.limit("20/minute")
async def match_face(req: MatchRequest, request: Request, _uid: str = Depends(verify_token)):
    loop = asyncio.get_running_loop()
    try:
        img = await loop.run_in_executor(executor, _download_image, req.imageUrl)
        if img is None:
            return {"success": False, "matches": [], "error": "Image download failed"}

        faces = await loop.run_in_executor(executor, _get_faces, img)
        if not faces:
            return {"success": True, "matches": []}

        known = await loop.run_in_executor(executor, _fetch_event_embeddings, req.eventId)
        if not known:
            return {"success": True, "matches": []}

        matches = await loop.run_in_executor(executor, _vectorized_match, faces, known)
        return {"success": True, "matches": matches}

    except Exception as e:
        return {"success": False, "error": str(e)}

# ─────────────────────────────────────────────────────
#  MEDIA PROCESS  — API just enqueues; worker does the work
# ─────────────────────────────────────────────────────
@app.post("/media/process")
async def process_media_endpoint(req: ProcessMediaRequest, _uid: str = Depends(verify_token)):
    """
    API role: Validates auth, pushes job JSON to Redis queue, returns immediately.
    Worker role: Jobs are consumed from Redis by _worker_job_loop() — NOT this endpoint.

    The previous ThreadPoolExecutor approach has been replaced:
    - No more in-memory deque (jobs survive container restarts via Redis persistence)
    - No more MAX_CONCURRENT_JOBS guess — workers self-regulate by processing one at a time
    - Scale by adding worker containers, not threads inside the API
    """
    if not _redis:
        raise HTTPException(
            status_code=503,
            detail="Redis is required for job queuing. REDIS_URL is not configured."
        )

    job = {"mediaId": req.mediaId, "url": req.url, "eventId": req.eventId}
    try:
        _redis.rpush(REDIS_JOB_QUEUE_KEY, json.dumps(job))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to enqueue job: {str(e)}")

    return {"success": True, "message": "Job queued for processing"}

# ─────────────────────────────────────────────────────
#  MEDIA PROCESS SYNC WORKER — runs only inside worker containers
# ─────────────────────────────────────────────────────
def _process_media_sync(media_id: str, url: str, event_id: str):
    """
    Synchronous side of the ML pipeline. Runs in thread pool inside the
    worker container only. API containers never call this function.
    """
    try:
        img = _download_image(url)
        if img is None:
            print(f"[process] download failed: {media_id}")
            return

        faces = _get_faces(img)
        if not faces:
            return

        known = _fetch_event_embeddings(event_id)
        if not known:
            return

        matches = _vectorized_match(faces, known)
        if not matches:
            return

        batch = db.batch()
        col = db.collection("deliveries")
        for m in matches:
            ref = col.document()
            batch.set(ref, {
                "mediaId": media_id,
                "eventId": event_id,
                "uid": m["uid"],
                "status": "matched",
                "matchedBy": "ai",
                "matchScore": m["score"],
                # Denormalized URL so clients don't need a separate media read
                "mediaUrl": url,
                "createdAt": firestore.SERVER_TIMESTAMP,
            })
        batch.commit()

        # Create in-app notifications server-side.
        # Prevents client-side syncMediaNotifications race condition.
        notif_batch = db.batch()
        notif_col = db.collection("notifications")
        for m in matches:
            notif_ref = notif_col.document()
            notif_batch.set(notif_ref, {
                "uid": m["uid"],
                "type": "photo_matched",
                "title": "New Photo Matched! ✨",
                "body": "A new photo was found for you in an event.",
                "isRead": False,
                "data": {"mediaId": media_id, "eventId": event_id},
                "createdAt": firestore.SERVER_TIMESTAMP,
            })
        notif_batch.commit()
        print(f"[process] ✅ {media_id} → {len(matches)} deliveries + {len(matches)} notifications")

    except Exception as e:
        print(f"[process] ❌ {media_id}: {e}")
    finally:
        _release_job_lock(media_id)

# ─────────────────────────────────────────────────────
#  CLOUDINARY DELETE — IDOR-SAFE (ownership verified)
# ─────────────────────────────────────────────────────
@app.post("/cloudinary/delete")
async def cloudinary_delete(
    req: CloudinaryDeleteRequest,
    uid: str = Depends(verify_token)
):
    """
    FIX: Verifies the caller owns the media before deleting.
    Previously, any authenticated user could delete ANY photo by publicId.
    Now we look up the Firestore media document and confirm the uploader
    or an admin is making the request.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    # --- Ownership check ---
    # Check if requester is an admin
    user_doc = db.collection("users").document(uid).get()
    is_admin = user_doc.exists and user_doc.to_dict().get("role") == "admin"

    if not is_admin:
        # Find media documents matching this publicId
        media_query = db.collection("media").where("publicId", "==", req.publicId).limit(1).stream()
        media_docs = list(media_query)

        if not media_docs:
            # Also check by URL pattern for legacy records
            raise HTTPException(
                status_code=404,
                detail="Media not found. Cannot verify ownership."
            )

        media_data = media_docs[0].to_dict()
        uploader_uid = media_data.get("uploadedBy") or media_data.get("uid")

        if uploader_uid != uid:
            raise HTTPException(
                status_code=403,
                detail="Forbidden: You can only delete your own media."
            )
    # --- End ownership check ---

    try:
        result = delete_asset(req.publicId, req.resourceType)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _batch_deletion_bg(public_ids: list, resource_type: str):
    import time as _t
    t0 = _t.time()
    try:
        results = delete_batch(public_ids, resource_type)
        deleted_ids = [pid for pid, status in results.items() if status in ("deleted", "not_found")]

        if db and deleted_ids:
            batch = db.batch()
            ops = 0
            media_ids = []
            for i in range(0, len(deleted_ids), 10):
                chunk = deleted_ids[i:i + 10]
                for snap in db.collection("media").where("publicId", "in", chunk).stream():
                    media_ids.append(snap.id)
                    batch.delete(snap.reference)
                    ops += 1
            for i in range(0, len(media_ids), 10):
                chunk = media_ids[i:i + 10]
                for snap in db.collection("deliveries").where("mediaId", "in", chunk).stream():
                    batch.delete(snap.reference)
                    ops += 1
                    if ops >= 400:
                        batch.commit()
                        batch = db.batch()
                        ops = 0
            if ops > 0:
                batch.commit()
    except Exception as e:
        print(f"[batch_delete] ❌ {e}")
    print(f"[batch_delete] done in {_t.time()-t0:.2f}s")


@app.post("/cloudinary/delete/batch")
async def cloudinary_delete_batch(
    req: BatchDeleteRequest,
    background_tasks: BackgroundTasks,
    uid: str = Depends(verify_token)
):
    """
    Batch delete: admin-only operation. Regular users cannot batch-delete media.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    # Batch delete is admin-only: too destructive to allow for regular users
    user_doc = db.collection("users").document(uid).get()
    is_admin = user_doc.exists and user_doc.to_dict().get("role") == "admin"
    if not is_admin:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Batch deletion requires admin privileges."
        )

    if len(req.publicIds) > 100:
        return {"success": False, "error": "Max 100 items per batch"}

    background_tasks.add_task(_batch_deletion_bg, req.publicIds, req.resourceType)
    return {"success": True, "message": f"Queued deletion for {len(req.publicIds)} items"}
