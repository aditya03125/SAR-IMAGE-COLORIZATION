r"""Build data/manifest_sen12.csv from the raw SEN1-2 folders: a subset, split BY SCENE (never by patch).

  python src\prep_sen12.py --max-pairs 30000
Expected layout (unchanged from the download):  data\sen12_raw\<ROIs...season>\s1_<n>\*.png  and  ...\s2_<n>\*.png
"""
import argparse, glob, os, random
from collections import defaultdict
import numpy as np, pandas as pd
from PIL import Image
from common import ROOT, DATA


def pair_of(p):
    d, f = os.path.split(p)
    base = os.path.basename(d)
    if not base.startswith("s1"):
        return None
    q = os.path.join(os.path.dirname(d), "s2" + base[2:], f.replace("_s1_", "_s2_"))
    return q if os.path.exists(q) else None


def rel(p):
    try:
        return os.path.relpath(p, ROOT).replace("\\", "/")
    except ValueError:                      # different drive: keep absolute path
        return p.replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATA / "sen12_raw"))
    ap.add_argument("--max-pairs", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)
    files = glob.glob(os.path.join(a.root, "**", "*.png"), recursive=True)
    print("png files found:", len(files))
    scenes = defaultdict(list)
    for p in files:
        q = pair_of(p)
        if q:
            scenes[os.path.dirname(p)].append((p, q))
    if not scenes:
        raise SystemExit("No s1/s2 pairs found. Check that the SEN1-2 folders are inside data\\sen12_raw "
                         "and tell me one example file path so I can adapt pair_of().")
    keys = sorted(scenes)
    random.shuffle(keys)
    n = len(keys)
    ntr, nva = int(0.8 * n), int(0.1 * n)
    split = {k: ("train" if i < ntr else "val" if i < ntr + nva else "test") for i, k in enumerate(keys)}
    cap = max(1, a.max_pairs // n)
    chosen, left = [], []
    for k in keys:
        v = scenes[k][:]
        random.shuffle(v)
        chosen += [(k,) + x for x in v[:cap]]
        left += [(k,) + x for x in v[cap:]]
    if len(chosen) < a.max_pairs:
        random.shuffle(left)
        chosen += left[:a.max_pairs - len(chosen)]
    rows, bad = [], 0
    for k, p, q in chosen:
        try:
            s1 = np.asarray(Image.open(p).convert("L"))
            s2 = np.asarray(Image.open(q).convert("RGB"))
        except Exception:
            bad += 1
            continue
        if s1.std() < 3 or s2.std() < 3:        # blank or almost blank patch
            bad += 1
            continue
        rows.append(dict(s1=rel(p), s2=rel(q), scene=rel(k), split=split[k], source="sen12"))
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "manifest_sen12.csv", index=False)
    print(f"scenes: {n} | pairs kept: {len(df)} | dropped (unreadable/blank): {bad}")
    print(df.split.value_counts().to_string())
    print("example s1 mode/size:", Image.open(chosen[0][1]).mode, Image.open(chosen[0][1]).size, "| s2:", Image.open(chosen[0][2]).mode)


if __name__ == "__main__":
    main()
