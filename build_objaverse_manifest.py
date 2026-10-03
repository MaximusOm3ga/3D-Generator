import argparse
import json
import os
import sys
from typing import Dict, List

from datasets import load_dataset
import objaverse


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    return False


def _safe_caption(row: Dict) -> str:
    for key in ["caption", "text", "description", "name", "title"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default="data/manifest.jsonl")
    p.add_argument("--min-score", type=int, default=2)
    p.add_argument("--max-objects", type=int, default=50000)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--test-ratio", type=float, default=0.05)
    p.add_argument("--image-root", type=str, default="images")
    p.add_argument("--default-view", type=str, default="view_00.png")
    p.add_argument(
        "--category",
        type=str,
        default=None,
        help="Restrict to one LVIS category (e.g. 'chair', 'car'), same as "
        "prepare_data.py --category. Omit for all categories.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    ds = load_dataset("cindyxl/ObjaversePlusPlus", split="train")

    category_uids = None
    if args.category is not None:
        lvis = objaverse.load_lvis_annotations()
        if args.category not in lvis:
            print(f"'{args.category}' is not a known LVIS category.", file=sys.stderr)
            sys.exit(1)
        category_uids = set(lvis[args.category])
        print(f"Category '{args.category}': {len(category_uids)} objects in LVIS")

    rows: List[Dict] = []
    for r in ds:
        if category_uids is not None and r.get("UID") not in category_uids:
            continue
        score = r.get("score", -1)
        if score is None or score < args.min_score:
            continue
        if _to_bool(r.get("is_scene")):
            continue
        if _to_bool(r.get("is_multi_object")):
            continue
        if _to_bool(r.get("is_transparent")):
            continue
        rows.append(r)
        if len(rows) >= args.max_objects:
            break
        if len(rows) % 1000 == 0:
            print(f"filtered {len(rows)} qualifying objects...", flush=True)

    n = len(rows)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)

    with open(args.output, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if i < n_test:
                split = "test"
            elif i < n_test + n_val:
                split = "val"
            else:
                split = "train"

            uid = r["UID"]
            image_path = os.path.join(args.image_root, uid, args.default_view)
            item = {
                "uid": uid,
                "split": split,
                "text": _safe_caption(r),
                "image": image_path,
                "source": "objaverse++",
            }
            f.write(json.dumps(item, ensure_ascii=True) + "\n")
            if (i + 1) % 1000 == 0 or i + 1 == n:
                print(f"wrote {i + 1}/{n} manifest rows", flush=True)

    print(f"Wrote {n} entries to {args.output}")
    print(f"split counts: train={n - n_val - n_test} val={n_val} test={n_test}")


if __name__ == "__main__":
    main()