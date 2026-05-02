"""
Album Cover Database Builder
=============================
Crawls MusicBrainz and Cover Art Archive to build a CLIP+FAISS index
of album covers. Runs indefinitely, saving progress every 100 covers
so you can stop and resume at any time.

MusicBrainz has ~2 million release groups. At ~1 req/sec this takes
weeks to fully index — but you can start matching after just a few
thousand covers.

Setup:
    pip install transformers torch faiss-cpu requests Pillow numpy tqdm

Usage:
    # Start indexing from the beginning
    python build_index.py

    # Resume after stopping (reads progress.json automatically)
    python build_index.py

    # Index a specific artist first (good for testing)
    python build_index.py --artist "Coil"
    python build_index.py --artist "The Beatles"

    # Index by genre tag
    python build_index.py --tag "post-punk"

Output files (kept in sync, safe to stop anytime):
    covers.index      FAISS vector index
    covers.json       metadata (title, artist, mbid) per vector
    progress.json     crawl state (offset, total seen, total indexed)
"""

import argparse
import io
import json
import os
import time
from urllib.parse import quote

import faiss
import numpy as np
import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ── Config ────────────────────────────────────────────────────────────────────

CLIP_MODEL     = "openai/clip-vit-base-patch32"
INDEX_FILE     = "covers.index"
META_FILE      = "covers.json"
PROGRESS_FILE  = "progress.json"
VECTOR_DIM     = 512
SAVE_EVERY     = 10          # save index to disk every N successful encodes
MB_BATCH       = 100          # MusicBrainz results per page (max 100)
MB_HEADERS     = {"User-Agent": "AlbumCoverIndexer/1.0 (your@email.com)"}

# ── CLIP ──────────────────────────────────────────────────────────────────────

_processor = None
_model     = None

def load_clip():
    global _processor, _model
    if _processor is None:
        print("Loading CLIP model (~600MB download on first run)...")
        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        _model     = CLIPModel.from_pretrained(CLIP_MODEL)
        _model.eval()
        print("CLIP ready.\n")
    return _processor, _model


def encode_image(img: Image.Image) -> np.ndarray:
    import torch
    processor, model = load_clip()
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    vec = features.pooler_output.detach().numpy().astype("float32").reshape(1, -1)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec

# ── Index I/O ─────────────────────────────────────────────────────────────────

def load_index() -> tuple:
    """Load existing index + metadata, or create fresh ones."""
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE) as f:
            meta = json.load(f)
        print(f"Resuming — {index.ntotal} covers already indexed.")
    else:
        index = faiss.IndexFlatIP(VECTOR_DIM)
        meta  = []
        print("Starting fresh index.")
    return index, meta


def save_index(index, meta):
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w") as f:
        json.dump(meta, f)


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"offset": 0, "seen": 0, "indexed": 0}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

# ── MusicBrainz ───────────────────────────────────────────────────────────────

def fetch_release_groups(query: str, limit: int, offset: int) -> list:
    """Fetch a page of release groups from MusicBrainz."""
    url = (
        "https://musicbrainz.org/ws/2/release-group/"
        f"?query={quote(query)}&limit={limit}&offset={offset}&fmt=json"
    )
    try:
        resp = requests.get(url, headers=MB_HEADERS, timeout=15)
        resp.raise_for_status()
        groups = []
        for rg in resp.json().get("release-groups", []):
            credits = rg.get("artist-credit", [])
            names   = [c["name"] for c in credits if isinstance(c, dict) and "name" in c]
            groups.append({
                "title":  rg.get("title", "Unknown"),
                "artist": "".join(names) or "Unknown",
                "mbid":   rg.get("id"),
            })
        return groups
    except Exception as e:
        print(f"  MusicBrainz error: {e}")
        return []

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
    except Exception:
        pass
    return None


def download_image(url: str) -> Image.Image | None:
    try:
        resp = requests.get(
            url,
            headers={**MB_HEADERS, "Accept": "image/*"},
            timeout=15,
            allow_redirects=True,
        )
        if resp.status_code != 200:
            return None
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.thumbnail((512, 512), Image.LANCZOS)
        return img
    except Exception:
        return None

# ── Core crawl loop ───────────────────────────────────────────────────────────

def crawl(query: str, index, meta, progress: dict):
    """
    Crawl MusicBrainz pages, encode covers with CLIP, add to index.
    Saves every SAVE_EVERY successful encodes. Runs until exhausted.
    """
    existing_mbids = {m["mbid"] for m in meta}
    since_save     = 0

    print(f"Query: {query!r}")
    print(f"Starting at offset {progress['offset']}\n")
    print(f"{'INDEX':>7}  {'CLIP':>4}  ALBUM")
    print("─" * 60)

    while True:
        groups = fetch_release_groups(query, MB_BATCH, progress["offset"])

        if not groups:
            print("\nNo more results — crawl complete.")
            break

        for item in groups:
            progress["seen"] += 1
            mbid = item["mbid"]

            if not mbid or mbid in existing_mbids:
                continue

            # Rate limit: 1 req/sec for MusicBrainz
            time.sleep(1.0)

            cover_url = get_cover_url(mbid)
            if not cover_url:
                continue

            img = download_image(cover_url)
            if img is None:
                continue

            try:
                vec = encode_image(img)
            except Exception as e:
                print(f"  Encode error: {e}")
                continue

            index.add(vec)
            meta.append({"title": item["title"], "artist": item["artist"], "mbid": mbid})
            existing_mbids.add(mbid)
            progress["indexed"] += 1
            since_save += 1

            print(f"  [{index.ntotal:>6}]  ✓  {item['title'][:40]} — {item['artist'][:20]}")

            # Save every SAVE_EVERY covers so progress is never lost
            if since_save >= SAVE_EVERY:
                save_index(index, meta)
                save_progress(progress)
                print(f"\n  ── Checkpoint saved ({index.ntotal} total) ──\n")
                since_save = 0

        progress["offset"] += MB_BATCH
        save_progress(progress)

        # Small pause between pages
        time.sleep(1.0)

    # Final save
    save_index(index, meta)
    save_progress(progress)
    print(f"\nDone. Total indexed: {index.ntotal}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a CLIP+FAISS album cover database.")
    parser.add_argument("--artist", help="Index all albums by a specific artist")
    parser.add_argument("--tag",    help="Index albums with a MusicBrainz tag (e.g. 'post-punk')")
    args = parser.parse_args()

    index, meta = load_index()

    # Build MusicBrainz query
    if args.artist:
        query    = f'artist:"{args.artist}" AND primarytype:"album"'
        progress = {"offset": 0, "seen": 0, "indexed": 0}
    elif args.tag:
        query    = f'tag:"{args.tag}" AND primarytype:"album"'
        progress = {"offset": 0, "seen": 0, "indexed": 0}
    else:
        # Full crawl — resumes from saved progress
        query    = 'primarytype:"album" AND status:"official"'
        progress = load_progress()

    try:
        crawl(query, index, meta, progress)
    except KeyboardInterrupt:
        print("\n\nStopped by user. Saving progress...")
        save_index(index, meta)
        save_progress(progress)
        print(f"Saved. Resume by running the script again. ({index.ntotal} covers indexed so far)")

