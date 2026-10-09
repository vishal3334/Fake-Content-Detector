"""
TASK 2: Image Analyzer (Edited Photo Detection)
Methods: 1) Error Level Analysis (ELA)  2) EXIF metadata check
"""
import io
import os
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS
import numpy as np

EDIT_SOFTWARE = ["photoshop", "gimp", "lightroom", "canva", "snapseed",
                 "picsart", "pixlr", "paint.net", "affinity", "facetune", "illustrator"]


def error_level_analysis(path, quality=90, save_to=None):
    """Re-save image as JPEG and compare. Edited regions show higher error."""
    original = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)

    diff = ImageChops.difference(original, resaved)
    arr = np.array(diff).astype(float)
    max_diff = max(float(arr.max()), 1.0)

    ela_img = ImageEnhance.Brightness(diff).enhance(255.0 / max_diff)
    if save_to:
        ela_img.save(save_to)

    gray = arr.mean(axis=2)
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "max": float(gray.max()),
        # Share of pixels with a very high error level (suspicious patches)
        "hot_ratio": float((gray > 0.6 * gray.max()).mean()) if gray.max() > 0 else 0.0,
    }


def check_metadata(path):
    """Read EXIF data and look for editing-software traces."""
    info = {"has_exif": False, "software": None, "camera": None, "date": None, "flags": []}
    try:
        exif = Image.open(path)._getexif() or {}
    except Exception:
        exif = {}

    if not exif:
        info["flags"].append("No EXIF metadata found (common for edited/downloaded images)")
        return info

    info["has_exif"] = True
    data = {TAGS.get(k, k): v for k, v in exif.items()}
    info["software"] = str(data.get("Software", "")) or None
    info["camera"] = " ".join(str(x) for x in [data.get("Make", ""), data.get("Model", "")]).strip() or None
    info["date"] = str(data.get("DateTimeOriginal", "")) or None

    if info["software"] and any(s in info["software"].lower() for s in EDIT_SOFTWARE):
        info["flags"].append(f"Editing software detected: {info['software']}")
    if not info["camera"]:
        info["flags"].append("Camera make/model missing")
    modified = str(data.get("DateTime", ""))
    if info["date"] and modified and modified != info["date"]:
        info["flags"].append("Modified date differs from original capture date")
    return info


def analyze_image(path, ela_output_path=None):
    """Combine ELA + metadata into a simple score. Returns label, confidence, details."""
    ela = error_level_analysis(path, save_to=ela_output_path)
    meta = check_metadata(path)

    score = 0   # higher = more likely edited
    reasons = []

    if ela["std"] > 6:
        score += 2
        reasons.append("High variation in error levels (ELA) - possible tampering")
    elif ela["std"] > 3:
        score += 1
        reasons.append("Moderate variation in error levels (ELA)")
    if ela["hot_ratio"] > 0.02:
        score += 1
        reasons.append("Localized high-error regions found")

    for flag in meta["flags"]:
        score += 2 if "Editing software" in flag else 1
        reasons.append(flag)

    label = "Fake" if score >= 3 else "Real"
    confidence = min(95.0, 55.0 + abs(score - 2) * 10)
    return {
        "label": label,
        "confidence": round(confidence, 2),
        "score": score,
        "reasons": reasons or ["No suspicious signs found"],
        "ela": ela,
        "metadata": meta,
    }
