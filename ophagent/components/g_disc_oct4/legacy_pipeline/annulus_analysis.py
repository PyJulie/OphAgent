# annulus_analysis.py  -------------------------------------------------------
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib import cm, colors, patheffects as pe
from scipy.ndimage import map_coordinates as mcoord


def analyse_annulus(
    core_id         : str,
    layer_name      : str,
    thick_map       : np.ndarray,            # 512×512 μm
    disc_center_xy  : tuple[float, float],   # (x0, y0) in px
    *,
    outer_diam_mm   : float = 3.46,
    inner_diam_mm   : float = 3.46,
    scan_range_mm   : float = 6.0,
    vmin            : float = 0,
    vmax            : float = 150,
    alpha           : float = 0.8,
):
    """
    12-sector / TSNI の上下が反転しないよう，すべて画像座標系（y↓）で
    サンプリング・描画を行う。
    戻り値: ({"fig_key": Figure, ...}, result_df)
    """
    # ---------- 座標系 & マスク -------------------------------------------
    x0, y0 = disc_center_xy
    px_per_mm = 512 / scan_range_mm
    r_out = outer_diam_mm / 2 * px_per_mm
    r_in  = inner_diam_mm  / 2 * px_per_mm
    r_mid = (r_in + r_out) / 2
    line_mode = np.isclose(r_out, r_in, atol=1e-3)     # 幅ゼロ＝線状

    H, W = thick_map.shape
    Y, X = np.ogrid[:H, :W]
    dx, dy = X - x0, Y - y0
    r_px   = np.sqrt(dx**2 + dy**2)
    angle  = (np.degrees(np.arctan2(-dy, dx)) + 360) % 360   # 0°=Temporal
    annulus = (r_px >= r_in) & (r_px <= r_out)

    cmap, norm = cm.get_cmap("jet"), colors.Normalize(vmin=vmin, vmax=vmax)

    # ---------- helpers ----------------------------------------------------
    def display_coords(theta_deg, radius):
        """θ(deg) & radius(px) → (x, y) in image coords (y↓)"""
        theta_rad = np.deg2rad(theta_deg)
        return x0 + radius * np.cos(theta_rad), y0 - radius * np.sin(theta_rad)

    def sample_ring(deg_lo, deg_hi):
        """幅 1px の外周リングから [deg_lo, deg_hi) を線形補間でサンプリング"""
        if deg_lo < deg_hi:
            angs = np.arange(deg_lo, deg_hi)
        else:                                   # wrap-around
            angs = np.concatenate([np.arange(deg_lo, 360), np.arange(0, deg_hi)])
        xs, ys = display_coords(angs, r_out)
        return mcoord(thick_map, [ys, xs], order=1, mode="nearest")

    def wedge_angles(lo, hi):
        """数学座標 lo→hi を画像座標用 (y↓) に反転"""
        return (360 - hi) % 360, (360 - lo) % 360

    # ---------- ① global --------------------------------------------------
    global_mean = (
        np.nanmean(sample_ring(0, 360))
        if line_mode
        else np.nanmean(thick_map[annulus])
    )

    # ---------- ② 12-sector ----------------------------------------------
    sector12 = []
    for i in range(12):
        lo, hi = i * 30, (i + 1) * 30
        if line_mode:
            sector12.append(np.nanmean(sample_ring(lo, hi)))
        else:
            sector12.append(
                np.nanmean(
                    thick_map[annulus & (angle >= lo) & (angle < hi)]
                )
            )

    # ---------- ③ Upper / Lower ------------------------------------------
    if line_mode:
        upper_mean = np.nanmean(sample_ring(180, 360))
        lower_mean = np.nanmean(sample_ring(0, 180))
    else:
        upper_mean = np.nanmean(thick_map[annulus & (Y < y0)])
        lower_mean = np.nanmean(thick_map[annulus & (Y >= y0)])

    # ---------- ④ TSNI ----------------------------------------------------
    seg_def = {"T": [315, 45], "S": [45, 135], "N": [135, 225], "I": [225, 315]}
    tsni = {}
    for lab, (lo, hi) in seg_def.items():
        if line_mode:
            tsni[lab] = np.nanmean(sample_ring(lo, hi))
        else:
            mask = (
                (angle >= lo) | (angle < hi)
                if lo > hi
                else (angle >= lo) & (angle < hi)
            )
            tsni[lab] = np.nanmean(thick_map[annulus & mask])

    # ---------- DataFrame --------------------------------------------------
    df = pd.DataFrame(
        {
            "core_id": core_id,
            "layer": layer_name,
            "global": global_mean,
            **{f"{i+1}h": v for i, v in enumerate(sector12)},
            "upper": upper_mean,
            "lower": lower_mean,
            **tsni,
        },
        index=[0],  # 1-row DataFrame
    )

    # ---------- 可視化 -----------------------------------------------------
    figs = {}

    def base_fig():
        f, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(thick_map, cmap="jet", origin="upper", vmin=vmin, vmax=vmax)
        ax.add_patch(Circle((x0, y0), r_out, fill=False, edgecolor="white", lw=1.5))
        if not line_mode:
            ax.add_patch(
                Circle((x0, y0), r_in, fill=False, edgecolor="white", lw=1.5)
            )
        ax.scatter(x0, y0, marker="x", c="black", s=80, lw=2)
        ax.axis("off")
        return f, ax, im

    # -- global -------------------------------------------------------------
    f, ax, _ = base_fig()
    if line_mode:
        ring_theta = np.arange(0, 361)
        xs, ys = display_coords(ring_theta, r_out)
        ax.plot(xs, ys, c=cmap(norm(global_mean)), lw=2)
    else:
        a0, a1 = wedge_angles(0, 360)
        ax.add_patch(
            Wedge(
                (x0, y0),
                r_out,
                a0,
                a1,
                width=r_out - r_in,
                facecolor=cmap(norm(global_mean)),
                edgecolor="none",
                alpha=alpha,
            )
        )
    tx, ty = display_coords(0, r_mid)
    ax.text(
        tx,
        ty,
        f"{global_mean:.0f}",
        color="white",
        ha="center",
        va="center",
        path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
    )
    ax.set_title(f"{layer_name} global")
    figs["global"] = f

    # -- 12-sector ----------------------------------------------------------
    f, ax, _ = base_fig()
    for i, val in enumerate(sector12):
        lo, hi = i * 30, (i + 1) * 30
        if line_mode:
            ring_theta = np.arange(lo, hi)
            xs, ys = display_coords(ring_theta, r_out)
            ax.plot(xs, ys, c=cmap(norm(val)), lw=2)
        else:
            a0, a1 = wedge_angles(lo, hi)
            ax.add_patch(
                Wedge(
                    (x0, y0),
                    r_out,
                    a0,
                    a1,
                    width=r_out - r_in,
                    facecolor=cmap(norm(val)),
                    edgecolor="none",
                    alpha=alpha,
                )
            )
        ang_disp = lo + 15
        tx, ty = display_coords(ang_disp, r_mid)
        ax.text(
            tx,
            ty,
            f"{val:.0f}",
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
        )
    # sector 区切り線
    for i in range(12):
        ang = i * 30
        x_in, y_in = display_coords(ang, r_in)
        x_out, y_out = display_coords(ang, r_out)
        ax.plot([x_in, x_out], [y_in, y_out], c="white", lw=1)
    ax.set_title(f"{layer_name} 12-sector")
    figs["sector12"] = f

    # -- Upper / Lower ------------------------------------------------------
    f, ax, _ = base_fig()

    def add_sector(col, ang1, ang2):
        if line_mode:
            ring_theta = np.arange(ang1, ang2)
            xs, ys = display_coords(ring_theta, r_out)
            ax.plot(xs, ys, c=col, lw=2)
        else:
            a0, a1 = wedge_angles(ang1, ang2)
            ax.add_patch(
                Wedge(
                    (x0, y0),
                    r_out,
                    a0,
                    a1,
                    width=r_out - r_in,
                    facecolor=col,
                    edgecolor="none",
                    alpha=alpha,
                )
            )

    add_sector(cmap(norm(upper_mean)), 180, 360)
    add_sector(cmap(norm(lower_mean)), 0, 180)

    # 中心線
    x_left, _ = display_coords(180, r_out)
    x_right, _ = display_coords(0, r_out)
    ax.plot([x_left, x_right], [y0, y0], c="white", lw=1)

    # ラベル
    _, uy = display_coords(270, r_mid)  # 270°: 画面上
    _, ly = display_coords(90, r_mid)   #  90°: 画面下
    ax.text(
        x0,
        uy,
        f"{upper_mean:.0f}",
        color="white",
        ha="center",
        va="center",
        path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
    )
    ax.text(
        x0,
        ly,
        f"{lower_mean:.0f}",
        color="white",
        ha="center",
        va="center",
        path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
    )
    ax.set_title(f"{layer_name} Upper / Lower")
    figs["upper_lower"] = f

    # -- TSNI ---------------------------------------------------------------
    f, ax, _ = base_fig()
    for lab, (lo, hi) in seg_def.items():
        col = cmap(norm(tsni[lab]))
        if line_mode:
            if lo < hi:
                ring_theta = np.arange(lo, hi)
            else:  # wrap
                ring_theta = np.concatenate(
                    [np.arange(lo, 360), np.arange(0, hi)]
                )
            xs, ys = display_coords(ring_theta, r_out)
            ax.plot(xs, ys, c=col, lw=2)
        else:
            a0, a1 = wedge_angles(lo, hi)
            if lo < hi:
                ax.add_patch(
                    Wedge(
                        (x0, y0),
                        r_out,
                        a0,
                        a1,
                        width=r_out - r_in,
                        facecolor=col,
                        edgecolor="none",
                        alpha=alpha,
                    )
                )
            else:  # 分割して wrap-around
                ax.add_patch(
                    Wedge(
                        (x0, y0),
                        r_out,
                        a0,
                        360,
                        width=r_out - r_in,
                        facecolor=col,
                        edgecolor="none",
                        alpha=alpha,
                    )
                )
                ax.add_patch(
                    Wedge(
                        (x0, y0),
                        r_out,
                        0,
                        a1,
                        width=r_out - r_in,
                        facecolor=col,
                        edgecolor="none",
                        alpha=alpha,
                    )
                )
    # 区切り線
    for deg in [45, 135, 225, 315]:
        x_d, y_d = display_coords(deg, r_out)
        ax.plot([x0, x_d], [y0, y_d], c="white", lw=1)

    pos_deg = {"T": 0, "S": 90, "N": 180, "I": 270}
    for lab, deg in pos_deg.items():
        tx, ty = display_coords(deg, r_mid)
        ax.text(
            tx,
            ty,
            f"{tsni[lab]:.0f}",
            color="white",
            fontsize=9,
            ha="center",
            va="center",
            path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
        )
    ax.set_title(f"{layer_name} TSNI")
    figs["tsni"] = f

    # ---------- 返却 -------------------------------------------------------
    return figs, df
