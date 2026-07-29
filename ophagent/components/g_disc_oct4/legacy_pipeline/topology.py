"""
Retinal-OCT surface topology fixer
---------------------------------
* すべての層が列方向に単調増加 (上→下) になるよう補正する。
* 多層が交差していれば信頼度の高い順に位置を確定し、残りを高さ昇順で埋める。
* 必要なら ``min_gap`` だけ下方向へ押し出してギャップを保証。
"""

from __future__ import annotations
from typing import Optional

import torch


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def correct_surface_topology(
    surfaces: torch.Tensor,
    surface_maps: Optional[torch.Tensor] = None,
    *,
    min_gap: Optional[int] = None,
) -> torch.Tensor:
    """
    Parameters
    ----------
    surfaces
        (B, N_layer, W)  あるいは (N_layer, W) shape の境界予測。
    surface_maps
        (B, N_layer, H, W) の各層確信度ヒートマップ (なくても可)。
    min_gap
        隣接層の最小間隔 [pixel] を保証したいときに指定。
    Returns
    -------
    torch.Tensor
        交差のない層位置 (入力と同 shape)。
    """
    if surfaces.dim() == 2:              # → (1,N,W)
        surfaces = surfaces.unsqueeze(0)

    B, N, W = surfaces.shape
    fixed = surfaces.clone()

    # --- 列ごとに単調性を回復 ---------------------------------------------- #
    for b in range(B):
        for c in range(W):
            col_vec = fixed[b, :, c]                 # (N,)
            if (col_vec[1:] < col_vec[:-1]).any().item():   # ← item()!!
                prob_slice = (
                    None
                    if surface_maps is None
                    else surface_maps[b, :, :, c]
                )
                fixed[b, :, c] = _resolve_column(
                    col_vec, prob_slice, min_gap
                )

    return fixed


# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #
def _resolve_column(
    col_vec: torch.Tensor,                    # (N,)
    col_prob: Optional[torch.Tensor],         # (N,H) or None
    min_gap: Optional[int],
) -> torch.Tensor:
    """
    1 列分を「上→下」単調に直し、必要ならギャップも保証。
    """
    # ---------- 衝突解消 ---------------------------------------------------- #
    if not (col_vec[1:] < col_vec[:-1]).any().item():
        new_vec = col_vec.clone()
    else:
        # (a) 確信度マップがあればそれを優先
        if col_prob is not None and col_prob.size(0) == col_vec.numel():
            h_idx = torch.clamp(
                col_vec.round().long(), 0, col_prob.size(1) - 1
            )
            scores = col_prob[
                torch.arange(col_vec.numel(), device=col_vec.device), h_idx
            ]
            order = torch.argsort(scores, descending=True)      # 信頼度降順
            sorted_h, _ = torch.sort(col_vec)                   # 高さ昇順
            new_vec = torch.empty_like(col_vec)
            new_vec[order] = sorted_h
        # (b) ヒートマップがないなら単純に昇順
        else:
            new_vec, _ = torch.sort(col_vec)

    # ---------- 最小ギャップの保証 ----------------------------------------- #
    if min_gap is not None and min_gap > 0:
        for i in range(1, new_vec.numel()):
            needed = new_vec[i - 1] + min_gap
            if (new_vec[i] < needed).item():    # ← item()!!
                new_vec[i] = needed

    return new_vec
