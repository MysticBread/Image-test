import cv2
import requests
import numpy as np
from urllib.parse import quote

orb = cv2.ORB_create(nfeatures=1500)

HEADERS = {
    "User-Agent": "AlbumCoverMatcher/1.0"
}


def search_musicbrainz(artist, album, limit=5):
    query = f'artist:"{artist}" AND releasegroup:"{album}"'
    url = (
        "https://musicbrainz.org/ws/2/release-group/"
        f"?query={quote(query)}&limit={limit}&fmt=json"
    )

    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()

    data = response.json()
    results = []

    for rg in data.get("release-groups", []):
        results.append({
            "title": rg.get("title", "Unknown title"),
            "artist": rg.get("artist-credit", [{}])[0].get("name", "Unknown artist"),
            "mbid": rg.get("id")
        })

    return results


def get_cover_from_archive(mbid):
    return f"https://coverartarchive.org/release-group/{mbid}/front-500"


def image_from_url(url):
    try:
        response = requests.get(url, timeout=10, allow_redirects=True)

        if response.status_code != 200:
            return None

        arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    except requests.RequestException:
        return None


def compare_images(query_img, db_img):
    kp1, des1 = orb.detectAndCompute(query_img, None)
    kp2, des2 = orb.detectAndCompute(db_img, None)

    if des1 is None or des2 is None:
        return 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    good_matches = [m for m in matches if m.distance < 55]
    return len(good_matches)


def identify_album_from_cover(query_image_path, artist, album):
    query_img = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

    if query_img is None:
        raise ValueError("Could not load query image. Check the file path.")

    possible_albums = search_musicbrainz(artist, album)

    if not possible_albums:
        return None, 0

    best_album = None
    best_score = 0

    for item in possible_albums:
        cover_url = get_cover_from_archive(item["mbid"])
        db_img = image_from_url(cover_url)

        if db_img is None:
            print(f"No cover found for {item['title']}")
            continue

        score = compare_images(query_img, db_img)
        print(f"{item['title']} by {item['artist']} score: {score}")

        if score > best_score:
            best_score = score
            best_album = item

    if best_score < 25:
        return None, best_score

    return best_album, best_score


album, score = identify_album_from_cover(
    "test_image.jpg",
    artist="The Beatles",
    album="Abbey Road"
)

print("\nBest match:")

if album:
    print(f"{album['title']} by {album['artist']}")
    print("MBID:", album["mbid"])
else:
    print("No confident match found")

print("Score:", score)