# g-seg2D_predict/dataset.py
from __future__ import annotations

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from oct_converter.readers import FDA


class FDADataset(Dataset):
    """
    Dataset for a directory of Heidelberg *.fda files.

    Returns
    -------
    data          : FloatTensor (1, B, H, W)
    dummy_surface : FloatTensor (10, W)
    dummy_mask    : LongTensor  (H, W)
    has_seg       : LongTensor  (1,)
    s_e_idx       : np.ndarray  (2,)
    name          : str  (filename w/o extension)
    """

    def __init__(self, fda_dir: str):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(fda_dir, "*.fda")))
        if not self.files:
            raise RuntimeError(f"No *.fda files found in: {fda_dir}")

    # ─────────────────────────────────────────

    def __len__(self):
        return len(self.files)

    # ─────────────────────────────────────────

    def __getitem__(self, idx: int):
        fpath = self.files[idx]

        # 1) FDA → list|ndarray
        try:
            vol = FDA(fpath, printing=False).read_oct_volume().volume
        except Exception as e:
            raise RuntimeError(f"Failed to read {fpath}: {e}") from e

        # 2) list → ndarray (B, H, W)
        if isinstance(vol, list):
            vol = np.stack([np.asarray(v) for v in vol], axis=0)
        elif not isinstance(vol, np.ndarray):
            vol = np.asarray(vol)

        vol = vol.astype(np.float32)  # (B, H, W)

        # 3) torch tensor & dummy placeholders
        data = torch.from_numpy(vol).unsqueeze(0)  # (1, B, H, W)

        B, H, W = vol.shape
        dummy_surface = torch.zeros((10, W), dtype=torch.float32)
        dummy_mask = torch.zeros((H, W), dtype=torch.long)
        has_seg = torch.tensor([1], dtype=torch.long)
        s_e_idx = np.zeros(2, dtype=np.int32)

        return data, dummy_surface, dummy_mask, has_seg, s_e_idx, os.path.basename(fpath)[:-4]
