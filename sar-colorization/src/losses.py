"""Extra loss terms. Inputs are in [-1, 1]."""
import torch, torch.nn.functional as F


def sobel_mag(x):
    g = x.float().mean(1, keepdim=True)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device).view(1, 1, 3, 3)
    gx, gy = F.conv2d(g, kx, padding=1), F.conv2d(g, kx.transpose(2, 3), padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def edge_loss(fake, real):
    """Keeps outlines (roads, rivers, coastlines) where the real optical image has them.
    Edges are taken from the REAL optical image, not from the speckled SAR."""
    return F.l1_loss(sobel_mag(fake), sobel_mag(real))


def water_loss(sar, fake, thr=-0.7):
    """EXPERIMENTAL and not validated. Where the radar is very dark (often water, but also shadow,
    smooth roads, bare sand) the output should be bluer than red. Keep the weight small and
    check the result by eye before you claim anything about it."""
    m = (sar < thr).float()
    return (m * F.relu(fake[:, 0:1] - fake[:, 2:3])).sum() / (m.sum() + 1.0)
