import cv2
import requests
import numpy as np

orb = cv2.ORB_create(nfeatures=1000)

# Put MusicBrainz release-group IDs here
ALBUMS = [
    {
        "title": "Abbey Road",
        "artist": "The Beatles",
        "mbid": "b84ee12a-09ef-421b-82de-0441a926375b"
    },
    {
        "title": "The Dark Side of the Moon",
        "artist": "Pink Floyd",
        "mbid": "f5093c06-23e3-404f-aeaa-40f72885ee3a"
    }
]


def image_from_url(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    arr = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    return img


def get_cover_from_archive(mbid):
    url = f"https://coverartarchive.org/release-group/{mbid}"
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        return None

    data = response.json()

    if not data.get("images"):
        return None

    image = data["images"][0]

    if "thumbnails" in image and "small" in image["thumbnails"]:
        return image["thumbnails"]["small"]

    return image["image"]


def compare_images(query_img, db_img):
    kp1, des1 = orb.detectAndCompute(query_img, None)
    kp2, des2 = orb.detectAndCompute(db_img, None)

    if des1 is None or des2 is None:
        return 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    good_matches = [m for m in matches if m.distance < 50]

    return len(good_matches)


def identify_album_from_cover(query_image_path):
    query_img = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

    if query_img is None:
        raise ValueError("Could not load query image.")

    best_album = None
    best_score = 0

    for album in ALBUMS:
        cover_url = get_cover_from_archive(album["mbid"])

        if cover_url is None:
            print(f"No cover found for {album['title']}")
            continue

        db_img = image_from_url(cover_url)

        if db_img is None:
            continue

        score = compare_images(query_img, db_img)

        print(f"{album['title']} score: {score}")

        if score > best_score:
            best_score = score
            best_album = album

    if best_score < 15:
        return None, best_score

    return best_album, best_score


album, score = identify_album_from_cover("test_image.jpg")

print("\nBest match:")
print(album)
print("Score:", score)
