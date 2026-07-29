"""
CRAE and CRVE via Knudtson's revised formula (2003).

Definition
----------
Let d₁ ≥ d₂ ≥ … ≥ dₙ be the widths of the largest N vessels of one type
within the measurement zone. Knudtson's iterative pair-combining formula
combines the widest with the narrowest:

    W' = k · sqrt(W_big² + W_small²)

with k = 0.88 for arterioles, k = 0.95 for venules. The operation is applied
repeatedly: pair (d₁, dₙ), (d₂, dₙ₋₁), …, take those outputs as the new
vector, sort descending, repeat until a single value remains. That value is
the central retinal artery / vein equivalent (CRAE / CRVE).

Reference
---------
Knudtson MD, Lee KE, Hubbard LD, Wong TY, Klein R, Klein BEK.
"Revised formulas for summarizing retinal vessel diameters."
Curr Eye Res. 2003;27(3):143-9.

We default N = 6 per the original paper. If fewer than N vessels are available
we use what is present and flag `qc.reduced_sample`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


K_ARTERIOLE = 0.88
K_VENULE = 0.95
DEFAULT_TOP_N = 6


@dataclass
class KnudtsonResult:
    value: float                       # CRAE or CRVE, same units as input diameters
    k: float
    n_vessels_used: int
    top_diameters: List[float]         # the diameters fed in, descending
    method: str = "knudtson_2003"


def knudtson_combine(diameters: Sequence[float], k: float) -> float:
    """Apply Knudtson's iterative pair-wise combination.

    Parameters
    ----------
    diameters : sequence of float
        Vessel diameters in any consistent unit.
    k : float
        Knudtson coefficient (0.88 for arterioles, 0.95 for venules).

    Returns
    -------
    float
        The combined equivalent. 0.0 when the input is empty.

    Notes
    -----
    Handling of odd counts: we pair the outer values and pass the lone middle
    value through unchanged into the next round (per the original paper's
    description and subsequent reference implementations).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    current = sorted((float(d) for d in diameters), reverse=True)
    if not current:
        return 0.0
    current = [d for d in current if d > 0]
    if not current:
        return 0.0

    while len(current) > 1:
        n = len(current)
        nxt: List[float] = []
        # pair index i from the end with index i from the start; for odd n the
        # central element passes through
        half = n // 2
        for i in range(half):
            big = current[i]
            small = current[n - 1 - i]
            nxt.append(k * float(np.sqrt(big * big + small * small)))
        if n % 2 == 1:
            nxt.append(current[half])  # middle value passes through
        current = sorted(nxt, reverse=True)
    return float(current[0])


def compute_equivalent(diameters: Sequence[float],
                       k: float,
                       top_n: int = DEFAULT_TOP_N) -> KnudtsonResult:
    """Select the top-N largest diameters and reduce them via Knudtson.

    Returns a `KnudtsonResult` carrying the value plus the k, n used and the
    raw diameters for auditability.
    """
    ds = sorted((float(d) for d in diameters if d > 0), reverse=True)
    top = ds[:top_n]
    value = knudtson_combine(top, k)
    return KnudtsonResult(
        value=value,
        k=k,
        n_vessels_used=len(top),
        top_diameters=top,
    )
