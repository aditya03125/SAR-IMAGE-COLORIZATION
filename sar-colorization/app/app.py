r"""Gradio demo.   python app\app.py   (needs at least one trained checkpoint in checkpoints\)"""
import sys, glob, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, torch, gradio as gr
from common import CKPT, RESULTS
from models import load_generator
from infer import read_sar, colorize, fade, heat, write_geotiff

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HAS_SLIDER = hasattr(gr, "ImageSlider")
_models = {}


def list_ckpts():
    return sorted(glob.glob(str(CKPT / "**" / "*.pt"), recursive=True))


def pick_default(c):
    for p in c[::-1]:
        if Path(p).name == "best.pt":
            return p
    return c[-1] if c else None


def run(file, ckpt, mc, use_fade):
    if not file:
        raise gr.Error("Upload a SAR image first.")
    if not ckpt:
        raise gr.Error("No checkpoint found in checkpoints/. Train a model first.")
    sar, prof = read_sar(file)
    if sar.shape[0] * sar.shape[1] > 4096 * 4096:
        raise gr.Error("Image is too large for the demo. Clip it to about 4000 x 4000 pixels in QGIS first.")
    if ckpt not in _models:
        _models[ckpt] = load_generator(ckpt, DEV)
    rgb, unc = colorize(_models[ckpt], sar, mc=int(mc), dev=DEV)
    shown = fade(rgb, unc) if (use_fade and unc is not None) else rgb
    tif = None
    if prof is not None:
        tif = str(Path(tempfile.mkdtemp()) / "colorized.tif")
        write_geotiff(tif, shown, prof)
    view = (np.stack([sar] * 3, -1), shown) if HAS_SLIDER else shown
    note = ("Colors are the model's best guess, an aid for interpretation and not a measurement. "
            "Faded areas are where repeated runs disagreed most (relative to this scene).")
    return view, (heat(unc) if unc is not None else None), tif, note


with gr.Blocks(title="TrustColor-SAR") as demo:
    gr.Markdown("# TrustColor-SAR\nTurn a radar (SAR) image into a readable color image, with a confidence layer.\n\n"
                "Input: a Sentinel-1 VV GeoTIFF in dB (e.g. from Earth Engine), or an 8-bit PNG radar image.")
    with gr.Row():
        with gr.Column(scale=1):
            f = gr.File(label="SAR image (.tif or .png)", type="filepath",
                        file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"])
            cks = list_ckpts()
            ck = gr.Dropdown(choices=cks, value=pick_default(cks), label="Model checkpoint")
            mc = gr.Slider(1, 16, value=8, step=1, label="Repeated runs for the confidence layer (1 = off)")
            fd = gr.Checkbox(value=True, label="Fade uncertain areas toward gray")
            btn = gr.Button("Colorize", variant="primary")
        with gr.Column(scale=2):
            view = (gr.ImageSlider(label="Radar | Colorized (drag the divider)", type="numpy")
                    if HAS_SLIDER else gr.Image(label="Colorized"))
            hm = gr.Image(label="Uncertainty (brighter = less sure)")
            dl = gr.File(label="Download GeoTIFF (only when the input had coordinates)")
            msg = gr.Markdown()
    btn.click(run, [f, ck, mc, fd], [view, hm, dl, msg])
    demos = sorted(glob.glob(str(RESULTS / "demo_scenes" / "*.tif*")) + glob.glob(str(RESULTS / "demo_scenes" / "*.png")))
    if demos:
        gr.Examples([[d] for d in demos], inputs=[f], label="Sample scenes")

if __name__ == "__main__":
    demo.launch(inbrowser=True)
