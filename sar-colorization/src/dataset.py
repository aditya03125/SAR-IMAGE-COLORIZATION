"""Paired SAR/optical loader. SAR -> 1 x H x W, optical -> 3 x H x W, both scaled to [-1, 1]."""
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from common import ROOT, DATA

MANIFESTS = {"sen12": "manifest_sen12.csv", "odisha": "manifest_odisha.csv"}


class PairDataset(Dataset):
    def __init__(self, split, sources=("sen12", "odisha"), crop=256, augment=False):
        frames = []
        for src, fn in MANIFESTS.items():
            p = DATA / fn
            if src in sources and p.exists():
                frames.append(pd.read_csv(p))
        if not frames:
            raise FileNotFoundError("No manifest found in data/. Run prep_sen12.py and/or prep_odisha.py first.")
        df = pd.concat(frames, ignore_index=True)
        splits = [split] if isinstance(split, str) else list(split)
        self.df = df[df.split.isin(splits)].reset_index(drop=True)
        self.crop, self.augment = crop, augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        s1 = np.asarray(Image.open(ROOT / r.s1).convert("L"), dtype=np.float32)
        s2 = np.asarray(Image.open(ROOT / r.s2).convert("RGB"), dtype=np.float32)
        h, w = min(s1.shape[0], s2.shape[0]), min(s1.shape[1], s2.shape[1])
        c = min(self.crop, h, w)
        if self.augment:
            y, x = np.random.randint(0, h - c + 1), np.random.randint(0, w - c + 1)
        else:
            y, x = (h - c) // 2, (w - c) // 2
        s1, s2 = s1[y:y + c, x:x + c], s2[y:y + c, x:x + c]
        if self.augment:                               # identical geometric augmentation for both images
            if np.random.rand() < 0.5:
                s1, s2 = s1[:, ::-1], s2[:, ::-1]
            if np.random.rand() < 0.5:
                s1, s2 = s1[::-1], s2[::-1]
            k = np.random.randint(4)
            s1, s2 = np.rot90(s1, k), np.rot90(s2, k, axes=(0, 1))
        sar = torch.from_numpy(np.ascontiguousarray(s1) / 127.5 - 1.0).float().unsqueeze(0)
        rgb = torch.from_numpy(np.ascontiguousarray(s2) / 127.5 - 1.0).float().permute(2, 0, 1)
        return sar, rgb


def make_loader(split, batch=2, crop=256, augment=False, sources=("sen12", "odisha"),
                samples=None, odisha_frac=None, workers=2):
    """Returns (DataLoader or None, dataset). With augment=True it is a training loader."""
    ds = PairDataset(split, sources, crop, augment)
    if len(ds) == 0:
        return None, ds
    sampler = None
    if augment and samples:
        src = ds.df.source.values
        w = np.ones(len(ds))
        if odisha_frac is not None and (src == "odisha").any() and (src == "sen12").any():
            n_o = int((src == "odisha").sum())
            w = np.where(src == "odisha", odisha_frac / n_o, (1 - odisha_frac) / (len(ds) - n_o))
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), int(samples), replacement=True)
    dl = DataLoader(ds, batch_size=batch, sampler=sampler, shuffle=(sampler is None and augment),
                    num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=augment,
                    persistent_workers=workers > 0)
    return dl, ds
