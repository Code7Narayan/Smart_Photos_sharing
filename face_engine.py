# ARCHIVED — face_engine.py
#
# This file is the original class-based face engine and is NO LONGER USED.
# main.py implements the face engine as a singleton directly (see get_face_app()
# and _process_media_sync) using:
#   - det_size=(320, 320) instead of (640, 640) for lower memory + faster inference
#   - vectorized numpy matmul for batch similarity instead of nested Python loops
#   - TTL caching (now Redis-backed) for event embeddings
#
# Keeping this file for reference only. Do NOT import FaceEngine from here.
# Safe to delete once confident in production.
#
# Original code below:
# ─────────────────────────────────────────────────────────────────────────────

# import numpy as np
# import cv2
# import requests
# from insightface.app import FaceAnalysis
#
#
# class FaceEngine:
#     def __init__(self):
#         self.app = FaceAnalysis(name="buffalo_l")
#         self.app.prepare(ctx_id=0, det_size=(640, 640))
#
#     def _download_image(self, url: str):
#         resp = requests.get(url)
#         if resp.status_code != 200:
#             raise Exception("Unable to download image")
#         img_arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
#         img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
#         return img
#
#     def get_embeddings_from_url(self, url: str):
#         img = self._download_image(url)
#         faces = self.app.get(img)
#         if len(faces) == 0:
#             return []
#         return [f.embedding.tolist() for f in faces]
#
#     def cosine_similarity(self, a, b):
#         a = np.array(a)
#         b = np.array(b)
#         return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
#
#     def match_one(self, target_embedding, known_embeddings: list):
#         best_uid = None
#         best_score = -1
#         for item in known_embeddings:
#             uid = item["uid"]
#             emb = item["embedding"]
#             score = self.cosine_similarity(target_embedding, emb)
#             if score > best_score:
#                 best_score = score
#                 best_uid = uid
#         return best_uid, best_score
#
#     def match_many(self, face_embeddings: list, known_embeddings: list, min_similarity: float = 0.55):
#         matches = []
#         for emb in face_embeddings:
#             uid, score = self.match_one(emb, known_embeddings)
#             if uid and score >= min_similarity:
#                 matches.append({"uid": uid, "score": score})
#         unique = {}
#         for m in matches:
#             uid = m["uid"]
#             if uid not in unique or m["score"] > unique[uid]["score"]:
#                 unique[uid] = m
#         return list(unique.values())
