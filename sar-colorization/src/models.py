"""U-Net generator and PatchGAN discriminator (Pix2Pix style), sized for a 4 GB GPU by default."""
import torch, torch.nn as nn


def _down(i, o, norm=True):
    layers = [nn.Conv2d(i, o, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(o))
    layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)


def _up(i, o, drop):
    layers = [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.InstanceNorm2d(o)]
    if drop:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    """downs=7 works for 128 and 256 pixel inputs (input size must be divisible by 2**downs)."""

    def __init__(self, in_ch=1, out_ch=3, ngf=32, downs=7):
        super().__init__()
        c = [min(ngf * 2 ** k, ngf * 8) for k in range(downs)]
        self.down = nn.ModuleList([_down(in_ch, c[0], norm=False)] +
                                  [_down(c[k - 1], c[k], norm=(k < downs - 1)) for k in range(1, downs)])
        self.up = nn.ModuleList()
        for n, k in enumerate(range(downs - 1, 0, -1)):
            self.up.append(_up(c[k] if k == downs - 1 else 2 * c[k], c[k - 1], drop=n < 3))
        self.final = nn.Sequential(nn.ConvTranspose2d(2 * c[0], out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        es = []
        for d in self.down:
            x = d(x)
            es.append(x)
        for n, k in enumerate(range(len(es) - 1, 0, -1)):
            x = torch.cat([self.up[n](x), es[k - 1]], 1)
        return self.final(x)


class PatchD(nn.Module):
    """70x70 PatchGAN. Input: SAR (1 ch) and optical (3 ch), concatenated."""

    def __init__(self, in_ch=4, ndf=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.InstanceNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.InstanceNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False), nn.InstanceNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1))

    def forward(self, sar, rgb):
        return self.net(torch.cat([sar, rgb], 1))


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def enable_mc_dropout(model):
    """Eval mode, but keep Dropout layers random: repeated runs then differ (used for the confidence map)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def load_generator(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {"ngf": 32, "downs": 7})
    g = UNet(1, 3, cfg["ngf"], cfg["downs"]).to(device)
    g.load_state_dict(ck["G"])
    return g.eval()
