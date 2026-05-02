"""
Album Cover Identifier
======================
1. Send the image to Hugging Face (BLIP)  →  extract artist + album text
2. Query MusicBrainz with those names     →  get candidate release groups
3. Fetch cover art from Cover Art Archive →  ORB visual match to confirm

Setup:
    pip install opencv-python requests numpy Pillow

Usage:
    python album_matcher.py test_image.png
    python album_matcher.py test_image.png --hf-token hf_xxxxxxxx

A free Hugging Face account + token is all that's needed:
    https://huggingface.co/settings/tokens  (read token, no payment required)

The token can also be set via the HF_TOKEN environment variable.
"""

import argparse
import os
import time
from urllib.parse import quote

import cv2
import numpy as np
import requests
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────

ORB            = cv2.ORB_create(nfeatures=2000)
TARGET_SIZE    = (300, 300)
SCORE_THRESHOLD = 15

MB_HEADERS = {"User-Agent": "AlbumCoverMatcher/1.0 (your@email.com)"}

# BLIP VQA model — runs locally via transformers, no API key needed
# Downloads ~1GB on first run, cached after that
BLIP_MODEL = "Salesforce/blip-vqa-base"


# ── Step 1: Vision — BLIP running locally via transformers ───────────────────

_blip_processor = None
_blip_model = None

def _load_blip():
    """Load BLIP VQA model once and cache it. Downloads ~1GB on first run."""
    global _blip_processor, _blip_model
    if _blip_processor is None:
        from transformers import BlipProcessor, BlipForQuestionAnswering
        print("  Loading BLIP model (downloads ~1GB on first run)...")
        _blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL)
        _blip_model = BlipForQuestionAnswering.from_pretrained(BLIP_MODEL)
        _blip_model.eval()
        print("  BLIP model ready.")
    return _blip_processor, _blip_model


def identify_with_huggingface(image_path: str, hf_token: str = None) -> dict:
    """
    Run BLIP VQA locally to extract artist and album title from the cover.
    hf_token is unused but kept for API compatibility.
    Returns {"artist": str, "album": str}.
    """
    import torch
    processor, model = _load_blip()
    img = Image.open(image_path).convert("RGB")

    def ask(question: str) -> str:
        inputs = processor(img, question, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True).strip()
        return answer

    artist = ask("What is the name of the music artist or band on this album cover?")
    album  = ask("What is the album title on this album cover?")

    # Clean up common BLIP artefacts
    for prefix in ("it is ", "the answer is ", "this is ", "i don't know", "unknown"):
        if artist.lower().startswith(prefix):
            artist = artist[len(prefix):].strip()
        if album.lower().startswith(prefix):
            album = album[len(prefix):].strip()

    return {"artist": artist or "Unknown", "album": album or "Unknown"}


# ── Step 2: MusicBrainz search ────────────────────────────────────────────────

def search_musicbrainz(artist: str, album: str, limit: int = 5) -> list:
    """
    Search MusicBrainz for release groups matching artist + album.
    If the exact pair returns nothing, fall back to artist-only or album-only.
    """
    def _fetch(query: str) -> list:
        url = (
            "https://musicbrainz.org/ws/2/release-group/"
            f"?query={quote(query)}&limit={limit}&fmt=json"
        )
        resp = requests.get(url, headers=MB_HEADERS, timeout=10)
        resp.raise_for_status()
        results = []
        for rg in resp.json().get("release-groups", []):
            credits = rg.get("artist-credit", [])
            names   = [c["name"] for c in credits if isinstance(c, dict) and "name" in c]
            results.append({
                "title":  rg.get("title", "Unknown title"),
                "artist": "".join(names) or "Unknown artist",
                "mbid":   rg.get("id"),
            })
        return results

    # Try exact combined query first
    if artist != "Unknown" and album != "Unknown":
        results = _fetch(f'artist:"{artist}" AND releasegroup:"{album}"')
        if results:
            return results

    # Fall back to album title only (handles cases where artist wasn't read clearly)
    if album != "Unknown":
        results = _fetch(f'releasegroup:"{album}"')
        if results:
            return results

    # Last resort: artist only
    if artist != "Unknown":
        return _fetch(f'artist:"{artist}"')

    return []


# ── Step 3: Cover Art Archive ─────────────────────────────────────────────────

def get_cover_url(mbid: str) -> str | None:
    """Return the front-cover image URL for a MusicBrainz release group."""
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


# ── Step 4: ORB visual confirmation ──────────────────────────────────────────

def preprocess(img):
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(img)


def compare_images(query_img, db_img) -> int:
    q, d = preprocess(query_img), preprocess(db_img)
    _, des1 = ORB.detectAndCompute(q, None)
    _, des2 = ORB.detectAndCompute(d, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0

    raw  = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if len((m, n)) == 2 and m.distance < 0.75 * n.distance]
    return len(good)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def identify_album(image_path: str, hf_token: str) -> tuple:
    # 1. Vision
    print("→ Asking Hugging Face (BLIP) to read the cover...")
    vision = identify_with_huggingface(image_path, hf_token)
    print(f"  Detected: artist={vision['artist']!r}, album={vision['album']!r}")

    if vision["artist"] == "Unknown" and vision["album"] == "Unknown":
        print("  Could not extract any text from the cover.")
        return None, 0, vision

    # 2. MusicBrainz
    print("\n→ Searching MusicBrainz...")
    candidates = search_musicbrainz(vision["artist"], vision["album"])
    if not candidates:
        print("  No results found on MusicBrainz.")
        return None, 0, vision

    # 3. ORB match
    print("\n→ Fetching covers and matching visually...")
    query_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if query_img is None:
        raise ValueError(f"Could not load image: {image_path!r}")

    best_album, best_score = None, 0

    for item in candidates:
        time.sleep(1.0)  # MusicBrainz rate limit

        cover_url = get_cover_url(item["mbid"])
        if not cover_url:
            print(f"  ✗ No cover art: {item['title']}")
            continue

        db_img = image_from_url(cover_url)
        if db_img is None:
            print(f"  ✗ Download failed: {item['title']}")
            continue

        score = compare_images(query_img, db_img)
        print(f"  • {item['title']} by {item['artist']}  →  score: {score}")

        if score > best_score:
            best_score = score
            best_album = item

    if best_score < SCORE_THRESHOLD:
        # Vision ID was the best we could do — return it flagged as unconfirmed
        print(f"\n  ORB score too low ({best_score}); returning vision result unconfirmed.")
        return {
            "title":     vision["album"],
            "artist":    vision["artist"],
            "mbid":      None,
            "confirmed": False,
        }, best_score, vision

    best_album["confirmed"] = True
    return best_album, best_score, vision


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify an album cover using Hugging Face BLIP + MusicBrainz + ORB."
    )
    parser.add_argument("image", help="Path to the album cover image (jpg, png, webp)")
    args = parser.parse_args()

    album, score, vision = identify_album(args.image, hf_token=None)

    print("\n── Result ──────────────────────────────────────────")
    if album:
        status = "✓ Confirmed" if album.get("confirmed") else "~ Unconfirmed (low visual score)"
        print(f"  Status : {status}")
        print(f"  Title  : {album['title']}")
        print(f"  Artist : {album['artist']}")
        if album.get("mbid"):
            print(f"  MBID   : {album['mbid']}")
        print(f"  Score  : {score}")
    else:
        print("  Could not identify the album.")

