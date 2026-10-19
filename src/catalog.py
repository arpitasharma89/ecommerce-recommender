import csv
import html
from functools import lru_cache
from pathlib import Path
 
from product_images import fallback_emoji, image_url
 
CATALOG = Path(__file__).resolve().parent.parent / "assets" / "catalog.csv"
 
 
@lru_cache(maxsize=1)
def load_catalog() -> dict:
    """product_id -> details dict (empty if assets/catalog.csv is missing)."""
    if not CATALOG.exists():
        return {}
    with open(CATALOG, newline="", encoding="utf-8") as f:
        return {int(r["product_id"]): r for r in csv.DictReader(f)}
 
 
def display_title(product_id, fallback: str = "") -> str:
    d = load_catalog().get(int(product_id))
    return d["display_title"] if d else fallback
 
 
def _short_count(n: int) -> str:
    return f"{n / 1000:.1f}K" if n >= 1000 else str(n)
 
 
def card_html(item, score_html: str = "") -> str:
    """One product card as a single line of HTML (no blank lines, so Streamlit renders it)."""
    pid = int(item.get("product_id", 0))
    d = load_catalog().get(pid, {})
    e = lambda x: html.escape(str(x))
 
    brand = d.get("brand") or item.get("brand", "")
    title = d.get("display_title") or item.get("title", "Unknown")
    category = item.get("category", "")
    price = float(item.get("price", 0) or 0)
    reason = item.get("reason", "") or ""
 
    # photo stays keyed on the ORIGINAL title, which holds the product type
    img = image_url(item.get("title", ""), category, pid)
    emoji = fallback_emoji(category)
 
    # rating row (only real numbers from the data)
    n_ratings = int(d.get("n_ratings") or 0)
    if n_ratings > 0:
        avg = float(d["avg_rating"])
        filled = int(round(avg))
        stars = "★" * filled + "☆" * (5 - filled)
        rating = (f'<div class="rating-row"><span class="stars">{stars}</span> <b>{avg:.1f}</b> '
                  f'<span class="muted">({_short_count(n_ratings)} ratings)</span></div>')
    else:
        rating = '<div class="rating-row"><span class="muted">No ratings yet</span></div>'
 
    n_buy = int(d.get("n_purchases") or 0)
    bought = f'<div class="bought">{_short_count(n_buy)} bought</div>' if n_buy > 0 else ""
 
    feats = [x.strip() for x in (d.get("features") or "").split("|") if x.strip()]
    feat_html = "".join(f"<li>{e(x)}</li>" for x in feats[:3])
    feat_html = f'<ul class="features">{feat_html}</ul>' if feat_html else ""
 
    reason_html = f'<div class="reason-text">💡 {e(reason)}</div>' if reason else ""
 
    return (
        '<div class="product-card">'
        f'<div class="product-img">{emoji}<div class="photo" style="background-image:url(\'{e(img)}\')"></div></div>'
        f'<div class="brand-name">{e(brand)}</div>'
        f'<div class="p-title">{e(title)}</div>'
        f'{rating}{bought}'
        f'<div><span class="price-tag">${price:.2f}</span> &nbsp;{score_html}</div>'
        f'{feat_html}{reason_html}'
        f'<div class="muted small">{e(category)}</div>'
        '</div>'
    )
 