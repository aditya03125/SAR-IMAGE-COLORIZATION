r"""Evaluate a checkpoint: PSNR, SSIM, and whether the uncertainty map really tracks the error.

  python src\eval.py --ckpt checkpoints\base\best.pt --split test --sources sen12
  python src\eval.py --ckpt checkpoints\india\best.pt --split heldout --sources odisha
Run it for the baseline too, so you can show the comparison on a slide.
"""
import argparse
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from torchmetrics.functional import structural_similarity_index_measure as tm_ssim
from common import RESULTS
from dataset import make_loader
from models import load_generator, enable_mc_dropout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "heldout"])
    ap.add_argument("--sources", nargs="+", default=["sen12", "odisha"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--mc", type=int, default=8, help="repeated runs for uncertainty (1 = skip)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = load_generator(a.ckpt, dev)
    dl, ds = make_loader(a.split, 4, 256, False, a.sources, None, None, 0)
    if dl is None:
        raise SystemExit(f"No data in split '{a.split}' for sources {a.sources}")
    src = ds.df.source.values                      # loader is not shuffled, so order matches the dataframe
    rows, seen = [], 0
    for sar, rgb in dl:
        sar, rgb = sar.to(dev), rgb.to(dev)
        with torch.no_grad():
            g.eval()
            det = g(sar)
            if a.mc > 1:
                enable_mc_dropout(g)
                unc = torch.stack([g(sar) for _ in range(a.mc)]).std(0).mean(1)      # B x H x W
                g.eval()
        f, r = (det + 1) / 2, (rgb + 1) / 2
        mse = ((f - r) ** 2).flatten(1).mean(1)
        psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
        ss = tm_ssim(f, r, data_range=1.0, reduction="none")
        err = (det - rgb).abs().mean(1)                                              # B x H x W
        for i in range(sar.size(0)):
            row = dict(source=src[seen + i], psnr=psnr[i].item(), ssim=ss[i].item(), err=err[i].mean().item())
            if a.mc > 1:
                row["unc"] = unc[i].mean().item()
                idx = torch.randperm(unc[i].numel(), device=dev)[:4000]
                row["pix_spearman"] = spearmanr(unc[i].flatten()[idx].cpu().numpy(),
                                                err[i].flatten()[idx].cpu().numpy())[0]
            rows.append(row)
        seen += sar.size(0)
        if seen >= a.n:
            break
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    name = a.ckpt.replace("\\", "/").split("/")[-2] + "_" + a.ckpt.replace("\\", "/").split("/")[-1][:-3]
    df.to_csv(RESULTS / f"eval_{name}_{a.split}.csv", index=False)
    print(f"\n{name} on split '{a.split}' ({len(df)} tiles)")
    print(df.groupby("source")[["psnr", "ssim"]].mean().round(3).to_string())
    if a.mc > 1:
        tile_r = spearmanr(df.unc, df.err)[0]
        print(f"uncertainty vs error, tile level (Spearman): {tile_r:.2f}")
        print(f"uncertainty vs error, pixel level (mean Spearman): {np.nanmean(df.pix_spearman):.2f}")
        print("Only claim that the confidence map is meaningful if these are clearly above 0.")


if __name__ == "__main__":
    main()
