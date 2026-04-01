from __future__ import annotations

import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import TensorDataset

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "input" / "quickdraw"
_QUICKDRAW_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified"

# All 345 Quick, Draw! categories from the official dataset.
QUICKDRAW_CATEGORIES: list[str] = [
    "aircraft carrier",
    "airplane",
    "alarm clock",
    "ambulance",
    "angel",
    "animal migration",
    "ant",
    "anvil",
    "apple",
    "arm",
    "asparagus",
    "axe",
    "backpack",
    "banana",
    "bandage",
    "barn",
    "baseball",
    "baseball bat",
    "basket",
    "basketball",
    "bat",
    "bathtub",
    "beach",
    "bear",
    "beard",
    "bed",
    "bee",
    "belt",
    "bench",
    "bicycle",
    "binoculars",
    "bird",
    "birthday cake",
    "blackberry",
    "blueberry",
    "book",
    "boomerang",
    "bottlecap",
    "bowtie",
    "bracelet",
    "brain",
    "bread",
    "bridge",
    "broccoli",
    "broom",
    "bucket",
    "bulldozer",
    "bus",
    "bush",
    "butterfly",
    "cactus",
    "cake",
    "calculator",
    "calendar",
    "camel",
    "camera",
    "camouflage",
    "campfire",
    "candle",
    "cannon",
    "canoe",
    "car",
    "carrot",
    "castle",
    "cat",
    "ceiling fan",
    "cello",
    "cell phone",
    "chair",
    "chandelier",
    "church",
    "circle",
    "clarinet",
    "clock",
    "cloud",
    "coffee cup",
    "compass",
    "computer",
    "cookie",
    "cooler",
    "couch",
    "cow",
    "crab",
    "crayon",
    "crocodile",
    "crown",
    "cruise ship",
    "cup",
    "diamond",
    "dishwasher",
    "diving board",
    "dog",
    "dolphin",
    "donut",
    "door",
    "dragon",
    "dresser",
    "drill",
    "drums",
    "duck",
    "dumbbell",
    "ear",
    "elbow",
    "elephant",
    "envelope",
    "eraser",
    "eye",
    "eyeglasses",
    "face",
    "fan",
    "feather",
    "fence",
    "finger",
    "fire hydrant",
    "fireplace",
    "firetruck",
    "fish",
    "flamingo",
    "flashlight",
    "flip flops",
    "floor lamp",
    "flower",
    "flying saucer",
    "foot",
    "fork",
    "frog",
    "frying pan",
    "garden",
    "garden hose",
    "giraffe",
    "goatee",
    "golf club",
    "grapes",
    "grass",
    "guitar",
    "hamburger",
    "hammer",
    "hand",
    "harp",
    "hat",
    "headphones",
    "hedgehog",
    "helicopter",
    "helmet",
    "hexagon",
    "hockey puck",
    "hockey stick",
    "horse",
    "hospital",
    "hot air balloon",
    "hot dog",
    "hot tub",
    "hourglass",
    "house",
    "house plant",
    "hurricane",
    "ice cream",
    "jacket",
    "jail",
    "kangaroo",
    "key",
    "keyboard",
    "knee",
    "knife",
    "ladder",
    "lantern",
    "laptop",
    "leaf",
    "leg",
    "light bulb",
    "lighter",
    "lighthouse",
    "lightning",
    "line",
    "lion",
    "lipstick",
    "lobster",
    "lollipop",
    "mailbox",
    "map",
    "marker",
    "matches",
    "megaphone",
    "mermaid",
    "microphone",
    "microwave",
    "monkey",
    "moon",
    "mosquito",
    "motorbike",
    "mountain",
    "mouse",
    "moustache",
    "mouth",
    "mug",
    "mushroom",
    "nail",
    "necklace",
    "nose",
    "ocean",
    "octagon",
    "octopus",
    "onion",
    "oven",
    "owl",
    "paintbrush",
    "paint can",
    "palm tree",
    "panda",
    "pants",
    "paper clip",
    "parachute",
    "parrot",
    "passport",
    "peanut",
    "pear",
    "peas",
    "pencil",
    "penguin",
    "piano",
    "pickup truck",
    "picture frame",
    "pig",
    "pillow",
    "pineapple",
    "pizza",
    "pliers",
    "police car",
    "pond",
    "pool",
    "popsicle",
    "postcard",
    "potato",
    "power outlet",
    "purse",
    "rabbit",
    "raccoon",
    "radio",
    "rain",
    "rainbow",
    "rake",
    "remote control",
    "rhinoceros",
    "rifle",
    "river",
    "roller coaster",
    "rollerskates",
    "sailboat",
    "sandwich",
    "saw",
    "saxophone",
    "school bus",
    "scissors",
    "scorpion",
    "screwdriver",
    "sea turtle",
    "see saw",
    "shark",
    "sheep",
    "shoe",
    "shorts",
    "shovel",
    "sink",
    "skateboard",
    "skull",
    "skyscraper",
    "sleeping bag",
    "smiley face",
    "snail",
    "snake",
    "snorkel",
    "snowflake",
    "snowman",
    "soccer ball",
    "sock",
    "speedboat",
    "spider",
    "spoon",
    "spreadsheet",
    "square",
    "squiggle",
    "squirrel",
    "stairs",
    "star",
    "steak",
    "stereo",
    "stethoscope",
    "stitches",
    "stop sign",
    "stove",
    "strawberry",
    "streetlight",
    "string bean",
    "submarine",
    "suitcase",
    "sun",
    "swan",
    "sweater",
    "swing set",
    "sword",
    "syringe",
    "table",
    "teapot",
    "teddy-bear",
    "telephone",
    "television",
    "tennis racquet",
    "tent",
    "The Eiffel Tower",
    "The Great Wall of China",
    "The Mona Lisa",
    "tiger",
    "toaster",
    "toe",
    "toilet",
    "tooth",
    "toothbrush",
    "toothpaste",
    "tornado",
    "tractor",
    "traffic light",
    "train",
    "tree",
    "triangle",
    "trombone",
    "truck",
    "trumpet",
    "t-shirt",
    "umbrella",
    "underwear",
    "van",
    "vase",
    "violin",
    "washing machine",
    "watermelon",
    "waterslide",
    "whale",
    "wheel",
    "windmill",
    "wine bottle",
    "wine glass",
    "wristwatch",
    "yoga",
    "zebra",
    "zigzag",
]

NUM_QUICKDRAW_CATEGORIES = len(QUICKDRAW_CATEGORIES)

CATEGORY_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(QUICKDRAW_CATEGORIES)}

# Descriptions use just the category name — the sentence transformer
# produces well-separated embeddings for distinct object names.
CATEGORY_DESCRIPTIONS: dict[str, str] = {cat: cat for cat in QUICKDRAW_CATEGORIES}

# Number of drawings to overlay per category. Low values (3-10) preserve
# recognizable shapes; high values (100+) blur into featureless blobs.
_MAX_DRAWINGS_PER_CATEGORY = 1


def _interpolate_stroke(xs: list[int], ys: list[int], step: float = 3.0) -> list[tuple[float, float]]:
    """Interpolate points along a stroke at approximately fixed spacing."""
    points: list[tuple[float, float]] = []
    for i in range(len(xs) - 1):
        x0, y0 = float(xs[i]), float(ys[i])
        x1, y1 = float(xs[i + 1]), float(ys[i + 1])
        dist = float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
        n_points = max(1, int(dist / step))
        for j in range(n_points):
            t = j / n_points
            points.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    if xs:
        points.append((float(xs[-1]), float(ys[-1])))
    return points


def _download_category(category: str, max_drawings: int = _MAX_DRAWINGS_PER_CATEGORY) -> npt.NDArray[np.float32]:
    """Download and process Quick, Draw! simplified drawings for a category.

    Returns an array of shape ``(N, 2)`` with coordinates in the ``[0, 255]`` range.
    """
    cache_file = _DATA_DIR / f"{category}.npy"
    if cache_file.exists():
        return cast(npt.NDArray[np.float32], np.load(cache_file))

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    url = f"{_QUICKDRAW_BASE_URL}/{urllib.parse.quote(category)}.ndjson"

    all_points: list[tuple[float, float]] = []
    count = 0

    with urllib.request.urlopen(url) as response:  # pyright: ignore[reportAny]
        for line in cast(Any, response):  # pyright: ignore[reportExplicitAny]
            if count >= max_drawings:
                break
            drawing: dict[str, Any] = json.loads(line)  # pyright: ignore[reportExplicitAny]
            if not drawing.get("recognized", False):
                continue
            for stroke in cast(list[list[list[int]]], drawing["drawing"]):
                all_points.extend(_interpolate_stroke(stroke[0], stroke[1]))
            count += 1

    points: npt.NDArray[np.float32] = np.array(all_points, dtype=np.float32)
    np.save(cache_file, points)
    return points


def _download_all_categories(categories: list[str], max_drawings: int = _MAX_DRAWINGS_PER_CATEGORY) -> None:
    """Download all categories in parallel, skipping those already cached."""
    to_download = [c for c in categories if not (_DATA_DIR / f"{c}.npy").exists()]
    if not to_download:
        return

    print(f"Downloading Quick, Draw! data for {len(to_download)} categories ({max_drawings} drawings each)...")
    completed = 0
    total = len(to_download)

    def _dl(cat: str) -> str:
        _download_category(cat, max_drawings)
        return cat

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_dl, c): c for c in to_download}
        for future in as_completed(futures):
            cat = future.result()
            completed += 1
            if completed % 25 == 0 or completed == total:
                print(f"  Downloaded {completed}/{total}: {cat}")

    print(f"  All {total} categories cached.")


def quickdraw_dataset(n: int = 0, categories: list[str] | None = None) -> TensorDataset:
    """Create a Quick, Draw! dataset of 2D points from Google's Quick, Draw! data.

    Downloads simplified stroke data for each category (caching locally), overlays
    a small number of drawings to form a recognizable point cloud per category,
    then optionally samples ``n`` points (0 = use all) with jitter noise and
    normalizes to approximately ``[-4, 4]``.
    """
    if categories is None:
        categories = QUICKDRAW_CATEGORIES

    # Download all categories in parallel
    _download_all_categories(categories)

    all_points_list: list[npt.NDArray[np.float32]] = []
    all_labels_list: list[npt.NDArray[np.int64]] = []

    for category in categories:
        idx = CATEGORY_TO_INDEX[category]
        points = _download_category(category)
        all_points_list.append(points)
        all_labels_list.append(np.full(len(points), idx, dtype=np.int64))

    points_arr = np.concatenate(all_points_list)
    labels_arr = np.concatenate(all_labels_list)
    total = len(points_arr)
    print(f"Quick, Draw! dataset: {total} total points across {len(categories)} categories")

    gen = torch.Generator().manual_seed(42)

    if n > 0 and n < total:
        ix = torch.randint(0, total, (n,), generator=gen)
        ix_np = ix.numpy()
    else:
        n = total
        ix_np = np.arange(total)

    x_vals = torch.tensor(points_arr[ix_np, 0], dtype=torch.float32)
    y_vals = torch.tensor(points_arr[ix_np, 1], dtype=torch.float32)
    label_tensor = torch.tensor(labels_arr[ix_np], dtype=torch.long)

    # Add small jitter noise to avoid exact point duplication
    x_vals = x_vals + torch.randn(n, generator=gen) * 0.15
    y_vals = y_vals + torch.randn(n, generator=gen) * 0.15

    # Normalize from [0, 255] to [-4, 4]
    x_vals = (x_vals / 128 - 1) * 4
    y_vals = (y_vals / 128 - 1) * 4

    # Flip y-axis (Quick, Draw! has y increasing downward)
    y_vals = -y_vals

    X = torch.stack((x_vals, y_vals), dim=1)
    return TensorDataset(X, label_tensor)
