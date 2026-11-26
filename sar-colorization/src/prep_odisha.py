r"""Turn the Earth Engine GeoTIFFs (bands VV,B4,B3,B2) into 256x256 PNG pairs + data/manifest_odisha.csv.

  python src\prep_odisha.py --holdout-block r0c0
Normalisation is done per cell and per band (mean +/- 2.5 std), like SEN1-2. Splits are by CELL,
and one whole macro-block is held out as the 'unseen region' test (split = heldout).
"""
import argparse, glob, os, random
import numpy as np, pandas as pd
from PIL import Image
from common import ROOT, DATA, stretch_u8


def tiles_from_cell(vv, rgb3, valid, tile=256, max_nodata=0.1, min_std=4.0):
    """vv: HxW float, rgb3: 3xHxW float, valid: HxW bool. Yields (y, x, sar_u8, rgb_u8 HxWx3)."""
    sar = stretch_u8(vv, valid)
    rgb = np.stack([stretch_u8(b, valid) for b in rgb3], -1)
    h, w = vv.shape
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            if (~valid[y:y + tile, x:x + tile]).mean() > max_nodata:
                continue
            s, r = sar[y:y + tile, x:x + tile], rgb[y:y + tile, x:x + tile]
            if s.std() < min_std or r.std() < min_std:
                continue
            yield y, x, s, r


def main():
    import rasterio
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATA / "odisha_raw"))
    ap.add_argument("--out", default=str(DATA / "tiles" / "odisha"))
    ap.add_argument("--holdout-block", default="r0c0")
    ap.add_argument("--include-probe", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.src, "*.tif*")))
    rows = []
    for f in files:
        tag = os.path.splitext(os.path.basename(f))[0]
        if tag.startswith("probe") and not a.include_probe:
            continue
        with rasterio.open(f) as ds:
            if ds.count < 4:
                print("skip (needs 4 bands):", f)
                continue
            arr = ds.read().astype(np.float32)
        vv, rgb3 = arr[0], arr[1:4]
        valid = np.isfinite(arr).all(0) & (vv != 0) & (rgb3.sum(0) > 0)
        os.makedirs(os.path.join(a.out, tag, "s1"), exist_ok=True)
        os.makedirs(os.path.join(a.out, tag, "s2"), exist_ok=True)
        k = 0
        for y, x, s, r in tiles_from_cell(vv, rgb3, valid):
            p1 = os.path.join(a.out, tag, "s1", f"{tag}_y{y}_x{x}.png")
            p2 = os.path.join(a.out, tag, "s2", f"{tag}_y{y}_x{x}.png")
            Image.fromarray(s, "L").save(p1)
            Image.fromarray(r, "RGB").save(p2)
            rows.append(dict(s1=os.path.relpath(p1, ROOT).replace("\\", "/"),
                             s2=os.path.relpath(p2, ROOT).replace("\\", "/"), scene=tag, source="odisha"))
            k += 1
        print(f"{tag}: {k} tiles")
    if not rows:
        raise SystemExit("No tiles written. Put the GeoTIFFs from Drive into data\\odisha_raw first.")
    df = pd.DataFrame(rows)
    tags = sorted(df.scene.unique())
    held = [t for t in tags if t.split("_")[0] == a.holdout_block]
    rest = [t for t in tags if t not in held]
    random.Random(a.seed).shuffle(rest)
    sp = {t: "train" for t in rest}
    for i, t in enumerate(rest):
        sp[t] = "val" if i < int(0.1 * len(rest)) else "test" if i < int(0.2 * len(rest)) else "train"
    sp.update({t: "heldout" for t in held})
    df["split"] = df.scene.map(sp)
    df.to_csv(DATA / "manifest_odisha.csv", index=False)
    print(f"cells: {len(tags)} | tiles: {len(df)}")
    print(df.split.value_counts().to_string())
    if not held:
        print("WARNING: no cells in held-out block", a.holdout_block, "- pick another block name (r0c0 ... r2c2)")


if __name__ == "__main__":
    main()
