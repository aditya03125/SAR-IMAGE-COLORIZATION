r"""Train the baseline U-Net (--baseline) or Pix2Pix (default). Defaults are chosen for a 4 GB GPU.

  python src\train.py --name baseline --baseline --epochs 5 --samples 5000
  python src\train.py --name base --epochs 20 --samples 20000
  python src\train.py --name india --init checkpoints\base\last.pt --sources odisha sen12 --odisha-frac 0.7 --epochs 10 --lr 1e-4 --samples 8000
If you run out of GPU memory: --batch 1, or --crop 128, or --ngf 24.  Resume after a crash: add --resume.
"""
import argparse, csv, time
import torch, torch.nn.functional as F
from torchvision.utils import save_image
from torchmetrics.functional import peak_signal_noise_ratio as tm_psnr, structural_similarity_index_measure as tm_ssim
from common import CKPT, RESULTS
from dataset import make_loader
from models import UNet, PatchD, init_weights
from losses import edge_loss, water_loss


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pix2pix")
    ap.add_argument("--baseline", action="store_true", help="U-Net + L1 only (no GAN)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--samples", type=int, default=20000, help="training samples drawn per epoch")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--ngf", type=int, default=32)
    ap.add_argument("--ndf", type=int, default=32)
    ap.add_argument("--downs", type=int, default=7)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--l1", type=float, default=100.0)
    ap.add_argument("--edge", type=float, default=10.0, help="edge (structure) loss weight, 0 = off")
    ap.add_argument("--water", type=float, default=0.0, help="experimental water-colour loss weight (try 5), 0 = off")
    ap.add_argument("--sources", nargs="+", default=["sen12"])
    ap.add_argument("--odisha-frac", type=float, default=None, help="share of Odisha samples when mixing")
    ap.add_argument("--init", default=None, help="checkpoint to start from (fine-tuning)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=1000, help="iterations between crash-safety saves")
    ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--no-amp", action="store_true")
    return ap.parse_args()


@torch.no_grad()
def validate(G, loader, dev, ac, n_max):
    G.eval()
    ps, ss, seen = [], [], 0
    for sar, rgb in loader:
        sar, rgb = sar.to(dev), rgb.to(dev)
        with ac():
            fake = G(sar)
        f, r = (fake.float() + 1) / 2, (rgb + 1) / 2
        ps.append(tm_psnr(f, r, data_range=1.0).item())
        ss.append(tm_ssim(f, r, data_range=1.0).item())
        seen += sar.size(0)
        if seen >= n_max:
            break
    return sum(ps) / len(ps), sum(ss) / len(ss)


@torch.no_grad()
def save_samples(G, fixed, dev, ac, path):
    G.eval()
    sar, rgb = fixed[0].to(dev), fixed[1].to(dev)
    with ac():
        fake = G(sar).float()
    grid = torch.cat([sar.repeat(1, 3, 1, 1), fake, rgb], 0)          # rows: radar / generated / real
    save_image(grid * 0.5 + 0.5, path, nrow=sar.size(0))


def main():
    a = get_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("WARNING: no CUDA GPU found. Training on CPU is extremely slow.")
    torch.backends.cudnn.benchmark = True
    out = CKPT / a.name
    out.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    amp = dev == "cuda" and not a.no_amp
    ac = lambda: torch.autocast(device_type=dev, dtype=torch.float16, enabled=amp)
    cfg = dict(ngf=a.ngf, ndf=a.ndf, downs=a.downs, baseline=a.baseline)

    G = UNet(1, 3, a.ngf, a.downs).to(dev)
    G.apply(init_weights)
    D = None if a.baseline else PatchD(4, a.ndf).to(dev)
    if D is not None:
        D.apply(init_weights)
    oG = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(0.5, 0.999))
    oD = None if D is None else torch.optim.Adam(D.parameters(), lr=a.lr, betas=(0.5, 0.999))
    half = a.epochs // 2
    lam = lambda e: 1.0 if e < half else max(0.0, 1.0 - (e - half) / max(1, a.epochs - half))
    sG = torch.optim.lr_scheduler.LambdaLR(oG, lam)
    sD = None if oD is None else torch.optim.lr_scheduler.LambdaLR(oD, lam)
    scG = torch.amp.GradScaler("cuda", enabled=amp)
    scD = torch.amp.GradScaler("cuda", enabled=amp)

    start, last = 0, out / "last.pt"
    if a.init:
        ck = torch.load(a.init, map_location=dev, weights_only=False)
        G.load_state_dict(ck["G"])
        if D is not None and ck.get("D") is not None:
            D.load_state_dict(ck["D"])
        print("initialised from", a.init)
    if a.resume and last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        G.load_state_dict(ck["G"]); oG.load_state_dict(ck["oG"]); sG.load_state_dict(ck["sG"])
        if D is not None and ck.get("D") is not None:
            D.load_state_dict(ck["D"]); oD.load_state_dict(ck["oD"]); sD.load_state_dict(ck["sD"])
        start = ck["epoch"]
        print("resumed at epoch", start)

    tl, _ = make_loader("train", a.batch, a.crop, True, a.sources, a.samples, a.odisha_frac, a.workers)
    if tl is None:
        raise SystemExit("No training data. Run the prep scripts first.")
    vl, _ = make_loader("val", 4, 256, False, a.sources, None, None, 0)
    fixed = next(iter(vl)) if vl is not None else None

    def save_last(ep):
        torch.save(dict(G=G.state_dict(), D=None if D is None else D.state_dict(), oG=oG.state_dict(),
                        oD=None if oD is None else oD.state_dict(), sG=sG.state_dict(),
                        sD=None if sD is None else sD.state_dict(), epoch=ep, cfg=cfg), last)

    best, logf = -1.0, out / "log.csv"
    try:
        for ep in range(start, a.epochs):
            G.train()
            if D is not None:
                D.train()
            t0, sums, n = time.time(), dict(g=0.0, d=0.0, l1=0.0, edge=0.0), 0
            for it, (sar, rgb) in enumerate(tl, 1):
                sar, rgb = sar.to(dev, non_blocking=True), rgb.to(dev, non_blocking=True)
                with ac():
                    fake = G(sar)
                d_loss = torch.zeros((), device=dev)
                if D is not None:                                   # discriminator step
                    with ac():
                        pr, pf = D(sar, rgb), D(sar, fake.detach())
                    d_loss = 0.5 * (F.mse_loss(pr.float(), torch.ones_like(pr, dtype=torch.float32)) +
                                    F.mse_loss(pf.float(), torch.zeros_like(pf, dtype=torch.float32)))
                    oD.zero_grad(set_to_none=True)
                    scD.scale(d_loss).backward(); scD.step(oD); scD.update()
                fake32 = fake.float()                               # generator step
                l1 = F.l1_loss(fake32, rgb)
                g_loss = a.l1 * l1
                if D is not None:
                    with ac():
                        pf = D(sar, fake)
                    g_loss = g_loss + F.mse_loss(pf.float(), torch.ones_like(pf, dtype=torch.float32))
                e = edge_loss(fake32, rgb)
                if a.edge > 0:
                    g_loss = g_loss + a.edge * e
                if a.water > 0:
                    g_loss = g_loss + a.water * water_loss(sar, fake32)
                oG.zero_grad(set_to_none=True)
                scG.scale(g_loss).backward(); scG.step(oG); scG.update()
                n += 1
                sums["g"] += g_loss.item(); sums["d"] += d_loss.item(); sums["l1"] += l1.item(); sums["edge"] += e.item()
                if it % 200 == 0:
                    el = time.time() - t0
                    print(f"ep {ep + 1}/{a.epochs} it {it}/{len(tl)} g {sums['g'] / n:.3f} d {sums['d'] / n:.3f} "
                          f"l1 {sums['l1'] / n:.3f} | {el / it:.2f}s/it, ~{el / it * (len(tl) - it) / 60:.1f} min left in epoch")
                if it % a.save_every == 0:
                    save_last(ep)
            sG.step()
            if sD is not None:
                sD.step()
            psnr_v, ssim_v = validate(G, vl, dev, ac, a.val_n) if vl is not None else (float("nan"),) * 2
            if fixed is not None:
                save_samples(G, fixed, dev, ac, RESULTS / f"samples_{a.name}_ep{ep + 1:03d}.png")
            save_last(ep + 1)
            torch.save(dict(G=G.state_dict(), cfg=cfg, epoch=ep + 1), out / f"epoch_{ep + 1:03d}.pt")
            old = out / f"epoch_{ep - 2:03d}.pt"                    # keep the 3 newest epoch files
            if old.exists():
                old.unlink()
            if ssim_v == ssim_v and ssim_v > best:                  # NOTE: SSIM can favour blurrier epochs
                best = ssim_v
                torch.save(dict(G=G.state_dict(), cfg=cfg, epoch=ep + 1, ssim=ssim_v), out / "best.pt")
            print(f"== epoch {ep + 1} done in {(time.time() - t0) / 60:.1f} min | val PSNR {psnr_v:.2f} SSIM {ssim_v:.3f}")
            new = not logf.exists()
            with open(logf, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["epoch", "g", "d", "l1", "edge", "val_psnr", "val_ssim", "minutes"])
                w.writerow([ep + 1, sums["g"] / n, sums["d"] / n, sums["l1"] / n, sums["edge"] / n,
                            psnr_v, ssim_v, (time.time() - t0) / 60])
    except KeyboardInterrupt:
        print("Interrupted - checkpoint last.pt kept; continue later with --resume")
    print("finished. checkpoints in", out)


if __name__ == "__main__":
    main()
