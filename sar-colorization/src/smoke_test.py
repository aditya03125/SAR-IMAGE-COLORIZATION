r"""Quick self-check with random data (no dataset needed). Run first:  python src\smoke_test.py"""
import numpy as np, torch, torch.nn.functional as F
from models import UNet, PatchD, init_weights, enable_mc_dropout
from losses import edge_loss, water_loss
from infer import colorize, fade, heat


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "|", torch.cuda.get_device_name(0) if dev == "cuda" else "")
    for s in (128, 256):
        y = UNet(1, 3, 16, 7).to(dev)(torch.randn(2, 1, s, s, device=dev))
        assert y.shape == (2, 3, s, s), y.shape
    print("U-Net shapes OK (128 and 256)")
    G, D = UNet(1, 3, 32, 7).to(dev), PatchD(4, 32).to(dev)
    G.apply(init_weights); D.apply(init_weights)
    sar, rgb = torch.randn(2, 1, 256, 256, device=dev), torch.rand(2, 3, 256, 256, device=dev) * 2 - 1
    fake = G(sar)
    loss = F.l1_loss(fake, rgb) + edge_loss(fake, rgb) + water_loss(sar, fake) + F.mse_loss(D(sar, fake), torch.ones_like(D(sar, fake)))
    loss.backward()
    print("one forward/backward step OK, loss =", round(loss.item(), 3))
    G.eval(); enable_mc_dropout(G)
    with torch.no_grad():
        a, b = G(sar), G(sar)
    assert (a - b).abs().max() > 0, "dropout is not active in MC mode"
    print("MC dropout gives different outputs: OK")
    img = (np.random.rand(300, 500) * 255).astype(np.uint8)
    out, unc = colorize(G, img, mc=3, dev=dev)
    assert out.shape == (300, 500, 3) and unc.shape == (300, 500)
    fade(out, unc); heat(unc)
    print("tiled colorize + confidence layer OK")
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
        G.train(); l = F.l1_loss(G(torch.randn(2, 1, 256, 256, device=dev)), torch.zeros(2, 3, 256, 256, device=dev)); l.backward()
        print(f"peak GPU memory for batch 2 at 256x256 (generator only): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("ALL OK")


if __name__ == "__main__":
    main()
