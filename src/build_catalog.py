from pathlib import Path
 
import numpy as np
import pandas as pd
 
ROOT = Path(__file__).resolve().parent.parent
sys_path = str(ROOT / "src")
import sys  # noqa: E402
sys.path.insert(0, sys_path)
from product_images import product_noun  # noqa: E402
 
OUT = ROOT / "assets" / "catalog.csv"
 
# product type -> real brands, ordered BUDGET -> PREMIUM
BRANDS = {
    "Headphones": ["boAt", "JBL", "Sony", "Sennheiser", "Bose"],
    "Speaker": ["boAt", "JBL", "Sony", "Marshall", "Bose"],
    "Smartwatch": ["Fire-Boltt", "Noise", "Samsung", "Garmin", "Apple"],
    "Charger": ["Ambrane", "Mi", "Samsung", "Belkin", "Anker"],
    "Router": ["D-Link", "TP-Link", "Netgear", "ASUS"],
    "Earbuds": ["boAt", "Realme", "OnePlus", "Samsung", "Apple"],
    "Webcam": ["HP", "Logitech", "Microsoft", "Razer"],
    "Power Bank": ["Ambrane", "Mi", "Realme", "Anker"],
    "Keyboard": ["HP", "Dell", "Logitech", "Corsair", "Razer"],
    "Monitor": ["Acer", "LG", "Dell", "Samsung", "ASUS"],
    "Jacket": ["H&M", "Puma", "Levi's", "Adidas", "Columbia"],
    "Sneakers": ["Skechers", "Puma", "Converse", "Adidas", "Nike"],
    "Backpack": ["Skybags", "Wildcraft", "American Tourister", "Nike"],
    "Sunglasses": ["Fastrack", "Vincent Chase", "Oakley", "Ray-Ban"],
    "Scarf": ["Fabindia", "Allen Solly", "Raymond"],
    "T-Shirt": ["H&M", "Puma", "Levi's", "Adidas", "Nike"],
    "Handbag": ["Lavie", "Caprese", "Baggit", "Hidesign"],
    "Wallet": ["Wildcraft", "Levi's", "Fossil", "Hidesign"],
    "Belt": ["Woodland", "Levi's", "Tommy Hilfiger", "Hidesign"],
    "Hat": ["Puma", "Adidas", "Nike", "New Era"],
    "Blender": ["Pigeon", "Bajaj", "Prestige", "Philips", "Havells"],
    "Cookware Set": ["Pigeon", "Wonderchef", "Prestige", "Hawkins", "Tefal"],
    "Air Fryer": ["Pigeon", "Inalsa", "Havells", "Prestige", "Philips"],
    "Bedding Set": ["Spaces", "Bombay Dyeing", "Raymond Home"],
    "Vacuum Cleaner": ["Eureka Forbes", "Philips", "Karcher", "Dyson"],
    "Coffee Maker": ["Morphy Richards", "Borosil", "Philips", "Nespresso"],
    "Lamp": ["Syska", "Wipro", "Havells", "Philips"],
    "Organizer": ["Cello", "Nilkamal", "Tupperware"],
    "Face Serum": ["Garnier", "Minimalist", "Neutrogena", "L'Oreal Paris", "The Ordinary"],
    "Shampoo": ["Dove", "Pantene", "TRESemme", "Head & Shoulders", "L'Oreal Paris"],
    "Moisturizer": ["Vaseline", "Nivea", "Lakme", "Neutrogena", "Cetaphil"],
    "Lipstick": ["Lakme", "Maybelline New York", "Revlon", "L'Oreal Paris", "MAC"],
    "Perfume": ["Fogg", "Axe", "Wild Stone", "Calvin Klein"],
    "Hair Dryer": ["Syska", "Havells", "Panasonic", "Philips", "Dyson"],
    "Razor": ["Gillette", "Schick", "Philips", "Braun"],
    "Yoga Mat": ["Boldfit", "Decathlon", "Reebok", "Adidas"],
    "Dumbbell Set": ["Boldfit", "Kore", "Decathlon", "Reebok"],
    "Running Shoes": ["Skechers", "Reebok", "Puma", "Adidas", "Nike", "ASICS"],
    "Camping Tent": ["Wildcraft", "Decathlon", "Coleman"],
    "Water Bottle": ["Cello", "Milton", "Borosil", "Tupperware"],
    "Bike Helmet": ["Decathlon", "Bell", "Giro"],
    "Novel": ["Penguin Random House", "HarperCollins", "Bloomsbury"],
    "Cookbook": ["Penguin Random House", "DK", "HarperCollins"],
    "Biography": ["Penguin Random House", "HarperCollins", "Bloomsbury"],
    "Journal": ["Classmate", "Archies", "Moleskine"],
    "Guidebook": ["DK Eyewitness", "Lonely Planet"],
    "Poetry Collection": ["Penguin Random House", "Faber & Faber", "HarperCollins"],
    "Textbook": ["Oxford University Press", "Pearson", "McGraw Hill", "Wiley"],
    "Board Game": ["Funskool", "Hasbro", "Mattel", "Ravensburger"],
    "Puzzle": ["Funskool", "Clementoni", "Hasbro", "Ravensburger"],
    "Action Figure": ["Funko", "Mattel", "Hasbro", "Bandai"],
    "Building Blocks": ["Funskool", "Mega Bloks", "LEGO"],
    "Remote Car": ["Funskool", "Maisto", "Hot Wheels"],
    "Plush Toy": ["Funskool", "Disney", "Hamleys"],
    "Organic Snack Pack": ["Kellogg's", "Nature Valley", "Organic India", "24 Mantra Organic"],
    "Coffee Beans": ["Nescafe", "Tata Coffee", "Lavazza", "Blue Tokai", "Starbucks"],
    "Tea Set": ["Cello", "Borosil", "Corelle"],
    "Spice Mix": ["Catch", "Everest", "MDH", "Tata Sampann"],
    "Granola Bars": ["Quaker", "Kellogg's", "Yoga Bar", "Nature Valley"],
    "Olive Oil": ["Del Monte", "Figaro", "Borges", "Bertolli"],
}
 
# product type -> readable product name used in the title
FULL_NAME = {
    "Headphones": "Wireless Bluetooth Headphones with Mic", "Speaker": "Portable Bluetooth Speaker",
    "Smartwatch": "Smartwatch with Fitness Tracking", "Charger": "Fast Charger Adapter",
    "Router": "Dual-Band Wi-Fi Router", "Earbuds": "True Wireless Earbuds",
    "Webcam": "Full HD Webcam with Microphone", "Power Bank": "Power Bank for Mobile Phones",
    "Keyboard": "Wireless Keyboard", "Monitor": "Full HD LED Monitor",
    "Jacket": "Men's Casual Jacket", "Sneakers": "Casual Sneakers",
    "Backpack": "Laptop Backpack", "Sunglasses": "UV Protection Sunglasses",
    "Scarf": "Soft Winter Scarf", "T-Shirt": "Cotton Round Neck T-Shirt",
    "Handbag": "Women's Handbag", "Wallet": "Leather Wallet", "Belt": "Leather Belt",
    "Hat": "Sports Cap",
    "Blender": "Mixer Blender", "Cookware Set": "Non-Stick Cookware Set",
    "Air Fryer": "Digital Air Fryer", "Bedding Set": "Cotton Bedsheet Set",
    "Vacuum Cleaner": "Vacuum Cleaner for Home", "Coffee Maker": "Coffee Maker Machine",
    "Lamp": "LED Table Lamp", "Organizer": "Storage Organizer",
    "Face Serum": "Face Serum", "Shampoo": "Daily Care Shampoo", "Moisturizer": "Moisturizing Cream",
    "Lipstick": "Matte Lipstick", "Perfume": "Eau de Parfum Perfume",
    "Hair Dryer": "Hair Dryer", "Razor": "Razor for Men",
    "Yoga Mat": "Anti-Slip Yoga Mat", "Dumbbell Set": "Dumbbell Set for Home Workout",
    "Running Shoes": "Running Shoes", "Camping Tent": "Camping Tent for Outdoors",
    "Water Bottle": "Insulated Water Bottle", "Bike Helmet": "Cycling Helmet",
    "Novel": "Bestselling Fiction Novel (Paperback)", "Cookbook": "Home Cooking Cookbook",
    "Biography": "Inspiring Biography Book", "Journal": "Ruled Notebook Journal",
    "Guidebook": "Travel Guidebook", "Poetry Collection": "Poetry Collection Book",
    "Textbook": "College Textbook",
    "Board Game": "Family Board Game", "Puzzle": "Jigsaw Puzzle", "Action Figure": "Collectible Action Figure",
    "Building Blocks": "Building Blocks Set", "Remote Car": "Remote Control Car", "Plush Toy": "Soft Plush Toy",
    "Organic Snack Pack": "Healthy Snack Pack", "Coffee Beans": "Roasted Coffee Beans",
    "Tea Set": "Ceramic Tea Set", "Spice Mix": "Ground Spice Mix",
    "Granola Bars": "Oats Granola Bars", "Olive Oil": "Extra Virgin Olive Oil",
}
 
# one key feature per product type
KEY_FEATURE = {
    "Headphones": "Bluetooth wireless connection with built-in microphone",
    "Speaker": "Portable design with wireless Bluetooth audio",
    "Smartwatch": "Tracks steps, heart rate and sleep",
    "Charger": "Fast charging with overheat protection",
    "Router": "Dual-band Wi-Fi for streaming and gaming",
    "Earbuds": "True wireless design with charging case",
    "Webcam": "Full HD video for online classes and calls",
    "Power Bank": "Charge your phone on the go",
    "Keyboard": "Comfortable, quiet keys for daily typing",
    "Monitor": "Full HD display with wide viewing angle",
    "Jacket": "Warm, durable fabric for daily wear",
    "Sneakers": "Cushioned sole for all-day comfort",
    "Backpack": "Padded compartment fits a laptop",
    "Sunglasses": "UV protection lenses",
    "Scarf": "Soft fabric that is easy to style",
    "T-Shirt": "Breathable cotton fabric",
    "Handbag": "Spacious interior with secure zip",
    "Wallet": "Multiple card slots and a coin pocket",
    "Belt": "Strong buckle with a durable finish",
    "Hat": "Adjustable fit and lightweight fabric",
    "Blender": "Powerful motor for smoothies and chutneys",
    "Cookware Set": "Non-stick coating, easy to clean",
    "Air Fryer": "Cooks with little or no oil",
    "Bedding Set": "Soft cotton that keeps its colour after washes",
    "Vacuum Cleaner": "Strong suction for dust and pet hair",
    "Coffee Maker": "Brews fresh coffee in minutes",
    "Lamp": "Energy-saving LED light",
    "Organizer": "Keeps your space neat and tidy",
    "Face Serum": "Lightweight formula for daily skincare",
    "Shampoo": "Gentle cleansing for everyday use",
    "Moisturizer": "Keeps skin soft and hydrated",
    "Lipstick": "Smooth, long-lasting colour",
    "Perfume": "Fresh long-lasting fragrance",
    "Hair Dryer": "Multiple heat settings for quick drying",
    "Razor": "Smooth, close shave",
    "Yoga Mat": "Anti-slip surface with good cushioning",
    "Dumbbell Set": "Durable build for home workouts",
    "Running Shoes": "Lightweight with good grip",
    "Camping Tent": "Easy to set up, water-resistant",
    "Water Bottle": "Leak-proof lid, easy to carry",
    "Bike Helmet": "Lightweight shell with adjustable fit",
    "Novel": "Paperback edition, easy to read",
    "Cookbook": "Step-by-step recipes for home cooks",
    "Biography": "True life story, engaging to read",
    "Journal": "Ruled pages for notes and daily writing",
    "Guidebook": "Maps and tips for planning your trip",
    "Poetry Collection": "A curated collection of poems",
    "Textbook": "Clear explanations with practice questions",
    "Board Game": "Fun for family game night",
    "Puzzle": "Builds focus and patience",
    "Action Figure": "Detailed collectible figure",
    "Building Blocks": "Encourages creativity and building skills",
    "Remote Car": "Easy remote control for kids",
    "Plush Toy": "Soft, huggable and safe for children",
    "Organic Snack Pack": "Tasty, ready-to-eat snacks",
    "Coffee Beans": "Freshly roasted, rich aroma",
    "Tea Set": "Elegant set for serving tea",
    "Spice Mix": "Adds authentic flavour to dishes",
    "Granola Bars": "Oats-based snack for busy days",
    "Olive Oil": "Cold-pressed oil for cooking and salads",
}
 
# extra generic lines per category (2 are picked at random for each product)
CATEGORY_FEATURES = {
    "Electronics": ["Easy to set up and use", "Compact, travel-friendly design", "Comes with user manual", "Modern, sleek look"],
    "Fashion": ["Available in multiple sizes", "Easy to wash and maintain", "Stylish everyday design", "Comfortable fit"],
    "Home & Kitchen": ["Easy to clean", "Designed for daily home use", "Sturdy, long-lasting build", "Space-saving design"],
    "Beauty & Personal Care": ["Dermatologically tested", "Suitable for daily use", "Pleasant, mild feel", "Travel-friendly pack"],
    "Sports & Outdoors": ["Built for regular training", "Durable and easy to carry", "Suitable for beginners", "Lightweight design"],
    "Books": ["Good print quality", "Great for gifting", "Easy to carry", "Popular with readers"],
    "Toys & Games": ["Safe, child-friendly materials", "Great gift for kids", "Encourages learning through play", "Durable build"],
    "Grocery": ["Sealed pack for freshness", "No added artificial colours", "Store in a cool, dry place", "Good for the whole family"],
}
 
COLORS = ["Black", "Blue", "White", "Grey", "Red", "Green", "Navy Blue", "Silver"]
NO_COLOR = {"Books", "Grocery", "Beauty & Personal Care"}
 
 
def build() -> pd.DataFrame:
    products = pd.read_csv(ROOT / "data" / "raw" / "products.csv")
    inter = pd.read_csv(ROOT / "data" / "raw" / "interactions.csv")
 
    products["noun"] = [product_noun(t) for t in products["title"]]
    # brand tier: cheaper products of a type get budget brands, expensive ones get premium brands
    products["price_pct"] = products.groupby("noun")["price"].rank(pct=True, method="first")
 
    rated = inter.dropna(subset=["rating"])
    rated = rated[(rated["rating"] >= 1) & (rated["rating"] <= 5)]       # ignore noisy values
    rstats = rated.groupby("product_id")["rating"].agg(n_ratings="count", avg_rating="mean")
    buys = inter[inter["interaction_type"] == "purchase"].groupby("product_id").size().rename("n_purchases")
 
    rows = []
    for r in products.itertuples():
        rng = np.random.default_rng(int(r.product_id))
        brands = BRANDS[r.noun]
        brand = brands[min(int(r.price_pct * len(brands)), len(brands) - 1)]
        title = f"{brand} {FULL_NAME[r.noun]}"
        if r.category not in NO_COLOR:
            title += f" ({rng.choice(COLORS)})"
        extra = rng.choice(CATEGORY_FEATURES[r.category], size=2, replace=False)
        features = " | ".join([KEY_FEATURE[r.noun], *extra])
        rows.append({"product_id": int(r.product_id), "brand": brand, "display_title": title, "features": features})
 
    cat = pd.DataFrame(rows).merge(rstats, on="product_id", how="left").merge(buys, on="product_id", how="left")
    cat["n_ratings"] = cat["n_ratings"].fillna(0).astype(int)
    cat["n_purchases"] = cat["n_purchases"].fillna(0).astype(int)
    cat["avg_rating"] = cat["avg_rating"].round(1)
    return cat
 
 
if __name__ == "__main__":
    catalog = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Saved {len(catalog)} products to {OUT}")
    print(f"Products with real ratings: {(catalog.n_ratings > 0).sum()} of {len(catalog)}")
    print(catalog.sample(3, random_state=1)[["brand", "display_title", "avg_rating", "n_ratings"]].to_string(index=False))
 
 