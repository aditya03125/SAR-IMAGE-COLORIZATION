r"""Export matched Sentinel-1 (VV, dB) + Sentinel-2 L1C (B4,B3,B2) cells over Odisha to Google Drive.

  python src\gee_export.py plan                      build the sampling plan (data/odisha_plan.json)
  python src\gee_export.py probe                     export 3 random cells to test everything
  python src\gee_export.py run --start 0 --end 100   export a batch of cells
  python src\gee_export.py status                    count export task states
  python src\gee_export.py s1 --lon 85.9 --lat 20.4 --date0 2025-08-01 --date1 2025-08-08 --tag monsoon_cuttack
                                                     radar-only scene (e.g. monsoon) for the demo, 1 band (VV)
Output bands (in this order): VV, B4, B3, B2.   Drive folder: odisha_sar
"""
import argparse, json, random
from collections import Counter
import ee
from common import DATA

PROJECT = "sar-colorization-515810"
W, S, E, N, STEP = 81.3, 17.7, 87.6, 22.7, 0.1          # Odisha bounding box, ~10 km grid
START, END = "2025-12-01", "2026-01-31"                  # clear-sky winter window
CITIES = {"bhubaneswar": (85.82, 20.30), "cuttack": (85.88, 20.46), "puri": (85.83, 19.81),
          "berhampur": (84.79, 19.31), "rourkela": (84.85, 22.26), "sambalpur": (83.97, 21.47),
          "balasore": (86.93, 21.49), "paradip": (86.61, 20.32)}
PLAN = DATA / "odisha_plan.json"


def block(cx, cy):
    r = min(2, int((cy - S) / ((N - S) / 3)))
    c = min(2, int((cx - W) / ((E - W) / 3)))
    return f"r{r}c{c}"


def build_plan(per_block=35, seed=42):
    st = (ee.FeatureCollection("FAO/GAUL/2025/level1")
          .filter(ee.Filter.eq("ISO3_CODE", "IND"))
          .filter(ee.Filter.inList("GAUL1_NAME", ["Odisha", "Orissa"])))
    n = st.size().getInfo()
    print("Odisha features:", n)
    if n == 0:
        raise SystemExit("Odisha not found. Print GAUL1_NAME for ISO3_CODE=='IND' and fix the name list.")
    odisha = st.geometry()
    nx, ny = int(round((E - W) / STEP)), int(round((N - S) / STEP))
    pts = [ee.Feature(ee.Geometry.Point([W + (i + .5) * STEP, S + (j + .5) * STEP]),
                      {"cx": W + (i + .5) * STEP, "cy": S + (j + .5) * STEP})
           for i in range(nx) for j in range(ny)]
    cells = [f["properties"] for f in ee.FeatureCollection(pts).filterBounds(odisha).getInfo()["features"]]
    print("grid cells whose centre is inside Odisha:", len(cells))
    random.seed(seed)
    by = {}
    for p in cells:
        by.setdefault(block(p["cx"], p["cy"]), []).append(p)
    plan, used = [], set()
    for b, v in sorted(by.items()):
        print(" block", b, "cells:", len(v))
        for k, p in enumerate(random.sample(v, min(per_block, len(v)))):
            plan.append(dict(tag=f"{b}_{k}", cx=p["cx"], cy=p["cy"], block=b)); used.add((p["cx"], p["cy"]))
    for name, (lon, lat) in CITIES.items():
        p = min(cells, key=lambda q: (q["cx"] - lon) ** 2 + (q["cy"] - lat) ** 2)
        if (p["cx"], p["cy"]) not in used:
            plan.append(dict(tag=f"city_{name}", cx=p["cx"], cy=p["cy"], block="city")); used.add((p["cx"], p["cy"]))
    DATA.mkdir(exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=1))
    print("planned cells:", len(plan), "->", PLAN)


def utm(lon):
    return "EPSG:326%02d" % (int((lon + 180) // 6) + 1)


def export_cell(cx, cy, tag, cloud=5):
    s1c = (ee.ImageCollection("COPERNICUS/S1_GRD").filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))
    s2c = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")             # Level-1C, like SEN1-2
    roi = ee.Geometry.Point([cx, cy]).buffer(5120).bounds()           # ~10.24 km square
    s2 = (s2c.filterBounds(roi).filterDate(START, END)
          .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud)).sort("CLOUDY_PIXEL_PERCENTAGE"))
    if s2.size().getInfo() == 0:
        return "no_clear_s2"
    s2img = ee.Image(s2.first())
    t = s2img.date()
    s1 = s1c.filterBounds(roi).filterDate(t.advance(-6, "day"), t.advance(6, "day"))
    if s1.size().getInfo() == 0:
        return "no_s1_nearby"
    img = s1.select("VV").mosaic().addBands(s2img.select(["B4", "B3", "B2"])).toFloat().clip(roi)
    ee.batch.Export.image.toDrive(image=img, description=tag, fileNamePrefix=tag, folder="odisha_sar",
                                  region=roi, scale=10, crs=utm(cx), maxPixels=1e9).start()
    return "started"


def export_s1(lon, lat, tag, d0, d1):
    roi = ee.Geometry.Point([lon, lat]).buffer(5120).bounds()
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filterBounds(roi).filterDate(d0, d1))
    if s1.size().getInfo() == 0:
        return "no_s1"
    ee.batch.Export.image.toDrive(image=s1.select("VV").mosaic().toFloat().clip(roi), description=tag,
                                  fileNamePrefix=tag, folder="odisha_sar", region=roi, scale=10,
                                  crs=utm(lon), maxPixels=1e9).start()
    return "started"


def run_cells(cells, cloud):
    st = Counter()
    for c in cells:
        try:
            r = export_cell(c["cx"], c["cy"], c["tag"], cloud)
        except Exception as e:                      # keep going, report at the end
            r = "error"; print("  error on", c["tag"], str(e)[:150])
        st[r] += 1
        print(c["tag"], r)
    print(dict(st))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "probe", "run", "status", "s1"])
    ap.add_argument("--lon", type=float)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--date0")
    ap.add_argument("--date1")
    ap.add_argument("--tag", default="s1_scene")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=100)
    ap.add_argument("--cloud", type=float, default=5.0)
    a = ap.parse_args()
    ee.Initialize(project=PROJECT)
    if a.cmd == "plan":
        build_plan()
        return
    if a.cmd == "s1":
        if None in (a.lon, a.lat, a.date0, a.date1):
            raise SystemExit("s1 needs --lon --lat --date0 --date1 (and optionally --tag)")
        print(export_s1(a.lon, a.lat, a.tag, a.date0, a.date1))
        return
    if a.cmd == "status":
        print(Counter(getattr(t, "state", "?") for t in ee.batch.Task.list()[:500]))
        return
    if not PLAN.exists():
        raise SystemExit("Run 'python src\\gee_export.py plan' first.")
    plan = json.loads(PLAN.read_text())
    cells = random.Random(1).sample(plan, 3) if a.cmd == "probe" else plan[a.start:a.end]
    if a.cmd == "probe":
        for c in cells:
            c["tag"] = "probe_" + c["tag"]
    run_cells(cells, a.cloud)


if __name__ == "__main__":
    main()
