"""
Album Cover Identifier — CLIP + FAISS edition
==============================================
1. Encode the query image with CLIP           →  get a visual vector
2. Search the FAISS index for nearest match   →  find closest cover
3. Fetch that cover from Cover Art Archive    →  ORB visual confirmation

No text needed — works on purely visual content including textless covers.

Setup:
    pip install transformers torch faiss-cpu opencv-python requests Pillow numpy

Usage:
    # First build the index (run once):
    python build_index.py --limit 500
    python build_index.py --artist "Coil" --append
    python build_index.py --artist "The Beatles" --append

    # Then identify any cover:
    python album_matcher.py test_image.png
    python album_matcher.py test_image.png --top 5
"""

import argparse
import io
import os
import json

import cv2
import faiss
import numpy as np
import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ── Constants ─────────────────────────────────────────────────────────────────

CLIP_MODEL      = "openai/clip-vit-base-patch32"
INDEX_FILE      = "covers.index"
META_FILE       = "covers.json"
ORB             = cv2.ORB_create(nfeatures=2000)
TARGET_SIZE     = (300, 300)
VECTOR_DIM      = 512
# CLIP cosine similarity threshold (0-1). Lower = accept weaker matches.
CLIP_THRESHOLD  = 0.70
MB_HEADERS      = {"User-Agent": "AlbumCoverMatcher/1.0 (your@email.com)"}

# ── CLIP ──────────────────────────────────────────────────────────────────────

_clip_processor = None
_clip_model     = None

def load_clip():
    global _clip_processor, _clip_model
    if _clip_processor is None:
        print("Loading CLIP model (downloads ~600MB on first run)...")
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        _clip_model     = CLIPModel.from_pretrained(CLIP_MODEL)
        _clip_model.eval()
        print("CLIP ready.\n")
    return _clip_processor, _clip_model


def encode_image(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a normalised CLIP vector."""
    import torch
    processor, model = load_clip()
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    vec = features.pooler_output.detach().numpy().astype("float32").reshape(1, -1)
    vec /= np.linalg.norm(vec) + 1e-9   # L2 normalise → cosine via dot product
    return vec

# ── FAISS search ──────────────────────────────────────────────────────────────

def search_index(query_vec: np.ndarray, top_k: int = 5) -> list:
    """
    Search the FAISS index for the top_k nearest covers.
    Returns list of {"title", "artist", "mbid", "score"} dicts.
    """
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(
            f"Index file {INDEX_FILE!r} not found.\n"
            "Run build_index.py first to create it."
        )

    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE) as f:
        meta = json.load(f)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:   # FAISS returns -1 for empty slots
            continue
        item = meta[idx].copy()
        item["clip_score"] = float(score)
        results.append(item)

    return results

# ── Cover Art Archive ─────────────────────────────────────────────────────────

def get_cover_url(mbid: str) -> str | None:
    try:
        resp = requests.get(
            f"https://coverartarchive.org/release-group/{mbid}",
            headers={**MB_HEADERS, "Accept": "application/json"},
            timeout=10,
            allow_redirects=True,
        )
        if resp.status_code != 200:
            return None
        for image in resp.json().get("images", []):
            if image.get("front"):
                thumbs = image.get("thumbnails", {})
                return thumbs.get("500") or thumbs.get("large") or image["image"]
    except (requests.RequestException, ValueError):
        pass
    return None


def image_from_url(url: str):
    """Download a URL and return a grayscale OpenCV image."""
    try:
        resp = requests.get(
            url,
            headers={**MB_HEADERS, "Accept": "image/*"},
            timeout=15,
            allow_redirects=True,
        )
        if resp.status_code != 200:
            return None
        arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    except requests.RequestException:
        return None

# ── ORB confirmation ──────────────────────────────────────────────────────────

def preprocess(img):
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(img)


def orb_score(query_img, db_img) -> int:
    q, d = preprocess(query_img), preprocess(db_img)
    _, des1 = ORB.detectAndCompute(q, None)
    _, des2 = ORB.detectAndCompute(d, None)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0
    raw  = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if len((m, n)) == 2 and m.distance < 0.75 * n.distance]
    return len(good)

# ── Main pipeline ─────────────────────────────────────────────────────────────

def identify_album(image_path: str, top_k: int = 5) -> dict | None:
    # 1. Load and encode the query image with CLIP
    print("→ Encoding query image with CLIP...")
    query_pil = Image.open(image_path).convert("RGB")
    query_vec = encode_image(query_pil)

    # 2. Search the FAISS index
    print("→ Searching index...")
    candidates = search_index(query_vec, top_k=top_k)

    if not candidates:
        print("  No candidates found in index.")
        return None

    print(f"  Top {len(candidates)} CLIP matches:")
    for c in candidates:
        print(f"    {c['clip_score']:.3f}  {c['title']} by {c['artist']}")

    # Filter by CLIP threshold — anything below is likely wrong
    strong = [c for c in candidates if c["clip_score"] >= CLIP_THRESHOLD]
    if not strong:
        print(f"\n  No candidates above CLIP threshold ({CLIP_THRESHOLD}).")
        print("  Closest match was:", candidates[0]["title"], "by", candidates[0]["artist"])
        print("  → Add more covers to the index: python build_index.py --artist \"Artist Name\"")
        return None

    # 3. ORB confirmation on top CLIP candidates
    print("\n→ Confirming with ORB feature matching...")
    query_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    best, best_orb = None, -1

    for item in strong:
        cover_url = get_cover_url(item["mbid"])
        if not cover_url:
            print(f"  ✗ No cover art: {item['title']}")
            continue

        db_img = image_from_url(cover_url)
        if db_img is None:
            print(f"  ✗ Download failed: {item['title']}")
            continue

        score = orb_score(query_cv, db_img)
        print(f"  • {item['title']} by {item['artist']}  →  ORB: {score}  CLIP: {item['clip_score']:.3f}")

        if score > best_orb:
            best_orb = score
            best = item

    # If ORB couldn't confirm, trust the top CLIP match
    if best is None:
        best = strong[0]
        best["confirmed_by"] = "CLIP only"
    else:
        best["confirmed_by"] = "CLIP + ORB"
        best["orb_score"]    = best_orb

    return best

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify an album cover using CLIP + FAISS + ORB."
    )
    parser.add_argument("image", help="Path to the album cover image (jpg, png, webp)")
    parser.add_argument("--top",  type=int, default=5,
                        help="Number of CLIP candidates to check with ORB (default 5)")
    args = parser.parse_args()

    result = identify_album(args.image, top_k=args.top)

    print("\n── Result ──────────────────────────────────────────")
    if result:
        print(f"  Title      : {result['title']}")
        print(f"  Artist     : {result['artist']}")
        print(f"  MBID       : {result['mbid']}")
        print(f"  CLIP score : {result['clip_score']:.3f}")
        print(f"  Confirmed  : {result['confirmed_by']}")
    else:
        print("  Could not identify the album.")
        print("  → Add more covers to the index: python build_index.py --artist \"Artist Name\"")
