"""Build a Nutrition5k benchmark subset: select dishes + download their images.

Nutrition5k (Thames et al., CVPR 2021, CC BY 4.0) gives ~5,000 real plated meals
with LAB-MEASURED total calories and overhead RGB imagery, served from a public
Google Cloud bucket. We select a spread of complete-plate dishes across the
calorie range and download their overhead photos, writing a manifest the eval
harness can consume.

    python benchmark/build_manifest.py            # default: 24 dishes

Re-runnable and deterministic (dishes are chosen by evenly spacing the
calorie-sorted candidates, no randomness), so the same manifest rebuilds.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import requests

BUCKET = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset"
UA = "calorie-pipeline-benchmark/1.0 (research; https://github.com/)"
HERE = Path(__file__).resolve().parent

# Selection filters: real complete plates, sane calorie range, enough mass that
# the overhead photo shows an actual meal rather than a single garnish.
MIN_KCAL, MAX_KCAL = 150.0, 800.0
MIN_INGREDIENTS = 3
MIN_MASS_G = 150.0
DISH_FIELDS = 6  # dish_id, total_calories, total_mass, total_fat, carb, protein
INGR_FIELDS = 7  # id, name, grams, calories, fat, carb, protein


def parse_rows(csv_path: Path) -> list[dict]:
    out = []
    for row in csv.reader(csv_path.open(newline="", encoding="utf-8")):
        if len(row) < DISH_FIELDS or not row[0].startswith("dish_"):
            continue
        try:
            kcal = float(row[1])
            mass = float(row[2])
        except ValueError:
            continue
        n_ingr = max(0, (len(row) - DISH_FIELDS)) // INGR_FIELDS
        out.append({"dish_id": row[0], "kcal": kcal, "mass": mass, "n_ingr": n_ingr})
    return out


def select(candidates: list[dict], n: int) -> list[dict]:
    keep = [
        c
        for c in candidates
        if MIN_KCAL <= c["kcal"] <= MAX_KCAL
        and c["mass"] >= MIN_MASS_G
        and c["n_ingr"] >= MIN_INGREDIENTS
    ]
    keep.sort(key=lambda c: c["kcal"])
    if len(keep) <= n:
        return keep
    # Evenly spaced across the calorie-sorted list for a clean spread.
    step = len(keep) / n
    return [keep[int(i * step)] for i in range(n)]


def download_image(dish_id: str, dest: Path) -> bool:
    url = f"{BUCKET}/imagery/realsense_overhead/{dish_id}/rgb.png"
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    except requests.RequestException:
        return False
    if r.status_code != 200 or len(r.content) < 1000:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return True


def main(argv: list[str]) -> int:
    target = int(argv[0]) if argv else 24
    csvs = [HERE / "n5k_cafe1.csv", HERE / "n5k_cafe2.csv"]
    candidates: list[dict] = []
    for c in csvs:
        if c.exists():
            candidates += parse_rows(c)
    if not candidates:
        print("No metadata CSVs found. Download them first (see README).", file=sys.stderr)
        return 1
    print(f"{len(candidates)} dishes in metadata; selecting ~{target} across the calorie range")

    # Over-select so image 404s don't shrink the final set below target.
    chosen = select(candidates, int(target * 1.4))
    dishes = []
    for c in chosen:
        if len(dishes) >= target:
            break
        img = HERE / "images" / c["dish_id"] / "rgb.png"
        if img.exists() or download_image(c["dish_id"], img):
            dishes.append(
                {
                    "dish_id": c["dish_id"],
                    "ground_truth_kcal": round(c["kcal"], 1),
                    "image_path": str(img.relative_to(HERE.parent)).replace("\\", "/"),
                    "total_mass_g": round(c["mass"], 1),
                    "n_ingredients": c["n_ingr"],
                }
            )
            print(f"  + {c['dish_id']}  {c['kcal']:.0f} kcal  ({c['n_ingr']} ingr)")
            time.sleep(0.05)
        else:
            print(f"  - {c['dish_id']}  (no overhead image, skipped)")

    manifest = HERE / "manifest.json"
    manifest.write_text(
        json.dumps({"source": "Nutrition5k (CC BY 4.0)", "dishes": dishes}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {manifest} with {len(dishes)} dishes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
