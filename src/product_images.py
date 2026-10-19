
import csv
from functools import lru_cache
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "assets" / "product_images.csv"


# product type (last words of the title) -> photo keyword
NOUN_KEYWORDS = {
    # Electronics
    "Headphones": "headphones", "Speaker": "speaker", "Smartwatch": "smartwatch",
    "Charger": "charger", "Router": "router", "Earbuds": "earbuds",
    "Webcam": "webcam", "Power Bank": "powerbank", "Keyboard": "keyboard",
    "Monitor": "monitor",
    # Fashion
    "Jacket": "jacket", "Sneakers": "sneakers", "Backpack": "backpack",
    "Sunglasses": "sunglasses", "Scarf": "scarf", "T-Shirt": "tshirt",
    "Handbag": "handbag", "Wallet": "wallet", "Belt": "belt", "Hat": "hat",
    # Home & Kitchen
    "Blender": "blender", "Cookware Set": "cookware", "Air Fryer": "fryer",
    "Bedding Set": "bedding", "Vacuum Cleaner": "vacuum", "Coffee Maker": "coffeemaker",
    "Lamp": "lamp", "Organizer": "organizer",
    # Beauty & Personal Care
    "Face Serum": "serum", "Shampoo": "shampoo", "Moisturizer": "moisturizer",
    "Lipstick": "lipstick", "Perfume": "perfume", "Hair Dryer": "hairdryer",
    "Razor": "razor",
    # Sports & Outdoors
    "Yoga Mat": "yoga", "Dumbbell Set": "dumbbell", "Running Shoes": "runningshoes",
    "Camping Tent": "tent", "Water Bottle": "waterbottle", "Bike Helmet": "bicyclehelmet",
    # Books
    "Novel": "novel", "Cookbook": "cookbook", "Biography": "book",
    "Journal": "notebook", "Guidebook": "guidebook", "Poetry Collection": "poetry",
    "Textbook": "textbook",
    # Toys & Games
    "Board Game": "boardgame", "Puzzle": "puzzle", "Action Figure": "actionfigure",
    "Building Blocks": "lego", "Remote Car": "toycar", "Plush Toy": "teddybear",
    # Grocery
    "Organic Snack Pack": "snacks", "Coffee Beans": "coffeebeans", "Tea Set": "teapot",
    "Spice Mix": "spices", "Granola Bars": "granola", "Olive Oil": "oliveoil",
}

# fallback when the product type is not in the map
CATEGORY_KEYWORDS = {
    "Electronics": "gadget", "Fashion": "fashion", "Home & Kitchen": "kitchen",
    "Beauty & Personal Care": "cosmetics", "Sports & Outdoors": "sport",
    "Books": "book", "Toys & Games": "toy", "Grocery": "grocery",
}

# shown behind the photo (visible only if the photo fails to load)
CATEGORY_EMOJI = {
    "Electronics": "🎧", "Fashion": "👕", "Home & Kitchen": "🏠",
    "Beauty & Personal Care": "💄", "Sports & Outdoors": "⚽",
    "Books": "📚", "Toys & Games": "🧸", "Grocery": "🛒",
}


def product_keyword(title: str, category: str = "") -> str:
    """Find the product type at the end of the title and return its photo keyword."""
    title = str(title)
    # try longest product types first so "Power Bank" wins over "Bank"
    for noun in sorted(NOUN_KEYWORDS, key=len, reverse=True):
        if title.endswith(noun):
            return NOUN_KEYWORDS[noun]
    return CATEGORY_KEYWORDS.get(category, "product")


# Better search phrases for the Pexels API (used by download_images.py)
NOUN_QUERIES = {
    "Headphones": "wireless headphones", "Speaker": "bluetooth speaker",
    "Smartwatch": "smartwatch", "Charger": "phone charger", "Router": "wifi router",
    "Earbuds": "wireless earbuds", "Webcam": "webcam", "Power Bank": "power bank",
    "Keyboard": "computer keyboard", "Monitor": "computer monitor",
    "Jacket": "jacket clothing", "Sneakers": "sneakers", "Backpack": "backpack",
    "Sunglasses": "sunglasses", "Scarf": "scarf", "T-Shirt": "t-shirt",
    "Handbag": "handbag", "Wallet": "leather wallet", "Belt": "leather belt", "Hat": "hat",
    "Blender": "kitchen blender", "Cookware Set": "cookware set", "Air Fryer": "air fryer",
    "Bedding Set": "bedding set", "Vacuum Cleaner": "vacuum cleaner",
    "Coffee Maker": "coffee maker", "Lamp": "table lamp", "Organizer": "desk organizer",
    "Face Serum": "face serum bottle", "Shampoo": "shampoo bottle",
    "Moisturizer": "moisturizer cream", "Lipstick": "lipstick", "Perfume": "perfume bottle",
    "Hair Dryer": "hair dryer", "Razor": "razor",
    "Yoga Mat": "yoga mat", "Dumbbell Set": "dumbbells", "Running Shoes": "running shoes",
    "Camping Tent": "camping tent", "Water Bottle": "water bottle", "Bike Helmet": "bike helmet",
    "Novel": "novel book", "Cookbook": "cookbook", "Biography": "biography book",
    "Journal": "journal notebook", "Guidebook": "travel guidebook",
    "Poetry Collection": "poetry book", "Textbook": "textbook",
    "Board Game": "board game", "Puzzle": "jigsaw puzzle", "Action Figure": "action figure toy",
    "Building Blocks": "building blocks toy", "Remote Car": "remote control car toy",
    "Plush Toy": "plush toy", "Organic Snack Pack": "healthy snacks",
    "Coffee Beans": "coffee beans", "Tea Set": "tea set", "Spice Mix": "spices",
    "Granola Bars": "granola bars", "Olive Oil": "olive oil bottle",
}


def product_noun(title: str):
    """Return the product type at the end of the title (e.g. 'Pro Router' -> 'Router')."""
    title = str(title)
    for noun in sorted(NOUN_KEYWORDS, key=len, reverse=True):
        if title.endswith(noun):
            return noun
    return None


@lru_cache(maxsize=1)
def _load_manifest() -> dict:
    """product_id -> photo url, read from assets/product_images.csv (empty if missing)."""
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return {int(r["product_id"]): r["image_url"] for r in csv.DictReader(f)}


def image_url(title: str, category: str, product_id, width: int = 400, height: int = 300) -> str:
    pid = int(product_id)
    url = _load_manifest().get(pid)
    if url:
        return url
    keyword = product_keyword(title, category)
    return f"https://loremflickr.com/{width}/{height}/{keyword}?lock={pid}"


def fallback_emoji(category: str) -> str:
    return CATEGORY_EMOJI.get(category, "🛍️")