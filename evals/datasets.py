"""Ground-truth datasets for evaluation.

The reference benchmark is **Nutrition5k** (Thames et al., CVPR 2021): ~5,000
real plated meals, each with overhead RGB imagery and lab-grounded total
calories. It is the right yardstick precisely because its ground truth is
*measured*, not modeled — the same epistemic stance the pipeline takes.

    @inproceedings{thames2021nutrition5k,
      title={Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food},
      author={Thames, Quin and Karpur, Arjun and Norris, Wade and Xia, Fangting
              and Panait, Liviu and Weyand, Tobias and Sim, Jack},
      booktitle={CVPR}, year={2021}}

Dataset: https://github.com/google-research-datasets/Nutrition5k (CC BY 4.0).

This module does not vendor the dataset (it is large and lives on Google Cloud).
It parses the official ``dish_metadata_cafe*.csv`` files you download, and also
reads a small JSON manifest for ad-hoc evaluation sets.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GroundTruthDish:
    """One evaluation example: an image and its measured total calories."""

    dish_id: str
    ground_truth_kcal: float
    image_path: str | None = None
    total_mass_g: float | None = None


def load_nutrition5k(csv_path: str | Path, imagery_dir: str | Path | None = None) -> list[GroundTruthDish]:
    """Parse an official Nutrition5k ``dish_metadata_cafe*.csv``.

    Column order (the fields we depend on are the first three):

        dish_id, total_calories, total_mass, total_fat, total_carb,
        total_protein, [ingr_1_id, ingr_1_name, ingr_1_grams, ...]

    If ``imagery_dir`` is given, each dish's overhead RGB image is resolved to
    ``<imagery_dir>/<dish_id>/rgb.png`` (the Nutrition5k overhead layout). Rows
    with a non-numeric calorie field are skipped rather than crashing the run.
    """
    csv_path = Path(csv_path)
    imagery = Path(imagery_dir) if imagery_dir is not None else None
    dishes: list[GroundTruthDish] = []

    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 3 or not row[0] or row[0].startswith("#"):
                continue
            dish_id = row[0].strip()
            kcal = _to_float(row[1])
            mass = _to_float(row[2])
            if kcal is None:
                continue  # header row or malformed line
            image_path = str(imagery / dish_id / "rgb.png") if imagery else None
            dishes.append(
                GroundTruthDish(
                    dish_id=dish_id,
                    ground_truth_kcal=kcal,
                    image_path=image_path,
                    total_mass_g=mass,
                )
            )
    return dishes


def load_manifest(path: str | Path) -> list[GroundTruthDish]:
    """Load a small JSON evaluation manifest.

    Expected shape::

        {"dishes": [
            {"dish_id": "...", "ground_truth_kcal": 0,
             "image_path": "...", "total_mass_g": 0}
        ]}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("dishes", data) if isinstance(data, dict) else data
    dishes: list[GroundTruthDish] = []
    for d in raw:
        dishes.append(
            GroundTruthDish(
                dish_id=str(d["dish_id"]),
                ground_truth_kcal=float(d["ground_truth_kcal"]),
                image_path=d.get("image_path"),
                total_mass_g=_to_float(d.get("total_mass_g")),
            )
        )
    return dishes


def _to_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
