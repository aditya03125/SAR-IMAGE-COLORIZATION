"""Colorize a whole scene by tiling with overlap, plus the confidence layer and GeoTIFF export."""
import numpy as np
from PIL import Image
from common import stretch_u8, tile_positions, hann2d


def read_sar(path):
    """GeoTIFF -> band 1 (VV in dB, e.g. from Earth Engine), normalised like the training data.
    PNG/JPG -> used as is (assumed to be an 8-bit radar image already). Returns (uint8 HxW, rasterio profile or None)."""
    if str(path).lower().endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(path) as ds:
            a = ds.read(1).astype(np.float32)
            prof = ds.profile
        return stretch_u8(a, valid=np.isfinite(a) & (a != 0)), prof
    return np.asarray(Image.open(path).convert("L")), None


def colorize(g, sar_u8, tile=256, overlap=32, mc=1, dev="cpu"):
    """Returns (rgb uint8 HxWx3, uncertainty float HxW or None). mc>1 = repeated runs with dropout on."""
    import torch
    from models import enable_mc_dropout
    h, w = sar_u8.shape
    ph, pw, pos = tile_positions(h, w, tile, overlap)
    pad = np.pad(sar_u8, ((0, ph - h), (0, pw - w)), mode="reflect")
    wt = hann2d(tile)
    acc = np.zeros((3, ph, pw), np.float32)
    accu = np.zeros((ph, pw), np.float32)
    accw = np.zeros((ph, pw), np.float32)
    use_mc = mc > 1
    g.eval()
    if use_mc:
        enable_mc_dropout(g)
    with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
        for y, x in pos:
            t = torch.from_numpy(pad[y:y + tile, x:x + tile].astype(np.float32) / 127.5 - 1.0)[None, None].to(dev)
            if use_mc:
                o = g(t.repeat(mc, 1, 1, 1)).float()
                mean = o.mean(0)
                accu[y:y + tile, x:x + tile] += o.std(0).mean(0).cpu().numpy() * wt
            else:
                mean = g(t).float()[0]
            acc[:, y:y + tile, x:x + tile] += mean.cpu().numpy() * wt
            accw[y:y + tile, x:x + tile] += wt
    g.eval()
    rgb = np.clip((acc / accw + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)[:h, :w]
    return rgb, ((accu / accw)[:h, :w] if use_mc else None)


def fade(rgb, unc, strength=0.85):
    """Blend toward gray where the model is less sure. Relative to this scene (top 5% = fully faded)."""
    if unc is None:
        return rgb
    a = np.clip(unc / (np.percentile(unc, 95) + 1e-6), 0, 1)[..., None] * strength
    gray = rgb.astype(np.float32).mean(2, keepdims=True)
    return (rgb * (1 - a) + gray * a).astype(np.uint8)


def heat(unc):
    import matplotlib
    v = np.clip(unc / (np.percentile(unc, 99) + 1e-6), 0, 1)
    return (matplotlib.colormaps["inferno"](v)[..., :3] * 255).astype(np.uint8)


def write_geotiff(path, rgb, prof):
    """Save the colorized image with the same coordinates as the input scene."""
    import rasterio
    p = prof.copy()
    for k in ("blockxsize", "blockysize", "tiled", "photometric"):
        p.pop(k, None)
    p.update(driver="GTiff", count=3, dtype="uint8", nodata=None, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(rgb.transpose(2, 0, 1))
