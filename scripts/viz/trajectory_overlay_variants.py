"""Dump several static trajectory overlay variants for E1 blind vs labeled.

Variants:
  A_action_grid    : small-multiples (3 blind + 3 labeled), each path colored
                     by discretized action (main / left / right / coast)
  B_reversal_dots  : 50-overlay + reversal events as bright dots on each path
  C_linewidth      : 50-overlay with line width modulated by |Δaction|
  D_thrust_strips  : 50-overlay on top, per-episode thrust-direction strips below
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

VIEWPORT_W, VIEWPORT_H = 600, 400
SCALE = 30.0
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
HELIPAD_Y = H / 4
LEG_OFFSET = 18 / SCALE


def state_to_pixels(states):
    sx, sy = states[:, 0], states[:, 1]
    pos_x = (sx + 1.0) * (W / 2)
    pos_y = sy * (H / 2) + (HELIPAD_Y + LEG_OFFSET)
    return np.stack([pos_x * SCALE, VIEWPORT_H - pos_y * SCALE], axis=1)


def load_episode(path):
    """Load episode npz. Drops the final state, which is the env auto-reset
    observation (a ~1.4 state-unit teleport back to the start), not a real
    continuation of the trajectory.
    """
    d = np.load(path, allow_pickle=True)
    meta = json.loads(d["metadata_json"].item())
    states = d["states"][:-1]  # drop reset observation
    return {
        "states": states,
        "actions": d["actions"],
        "outcome": meta["outcome"],
        "length": meta["episode_length"],
        "rgb_frames": d["rgb_frames"] if "rgb_frames" in d.files else None,
    }


def pick(traj_dir: Path, n: int, outcome="landed"):
    out = []
    for p in sorted(traj_dir.glob("episode_*.npz")):
        ep = load_episode(p)
        if outcome is None or ep["outcome"] == outcome:
            out.append(ep)
            if len(out) >= n:
                break
    return out


# Action discretization (continuous LunarLander):
#   main thrust active when action[0] > 0
#   lateral: left if action[1] < -0.5, right if > 0.5
# Categories: 0=coast, 1=main only, 2=left only, 3=right only,
#             4=main+left, 5=main+right
ACTION_COLORS = {
    0: "#777777",  # coast - grey
    1: "#2ca02c",  # main - green
    2: "#1f77b4",  # left - blue
    3: "#d62728",  # right - red
    4: "#17a2c4",  # main+left - teal
    5: "#ff7f0e",  # main+right - orange
}
ACTION_LABELS = {0: "coast", 1: "main", 2: "left", 3: "right",
                 4: "main+L", 5: "main+R"}


def discretize_actions(actions):
    main = actions[:, 0] > 0
    left = actions[:, 1] < -0.5
    right = actions[:, 1] > 0.5
    cat = np.zeros(len(actions), dtype=int)
    cat[main & ~left & ~right] = 1
    cat[~main & left] = 2
    cat[~main & right] = 3
    cat[main & left] = 4
    cat[main & right] = 5
    return cat


def thrust_reversals(actions, kind="any"):
    """Boolean mask of length T marking reversal events.

    kind="main"    main thrust on/off toggles
    kind="lateral" lateral category changes (left/none/right)
    kind="any"     either of the above (union)
    """
    main_on = (actions[:, 0] > 0).astype(int)
    lat = np.zeros(len(actions), dtype=int)
    lat[actions[:, 1] < -0.5] = -1
    lat[actions[:, 1] > 0.5] = 1
    main_change = np.diff(main_on) != 0
    lat_change = np.diff(lat) != 0
    rev = np.zeros(len(actions), dtype=bool)
    if kind == "main":
        rev[1:] = main_change
    elif kind == "lateral":
        rev[1:] = lat_change
    else:
        rev[1:] = main_change | lat_change
    return rev


# ---- Variant A: action-colored small-multiples ----
def variant_A(blind_eps, labeled_eps, backdrop, out: Path, n_per=3):
    fig, axes = plt.subplots(2, n_per, figsize=(4 * n_per, 6.5), dpi=150)
    for row, (eps, name) in enumerate([(blind_eps, "Blind"),
                                        (labeled_eps, "Labeled")]):
        for col in range(n_per):
            ax = axes[row, col]
            ax.imshow(backdrop, extent=[0, VIEWPORT_W, VIEWPORT_H, 0])
            if col >= len(eps):
                ax.axis("off")
                continue
            ep = eps[col]
            pts = state_to_pixels(ep["states"])
            cats = discretize_actions(ep["actions"])  # length T (=len(actions))
            # segments: between consecutive states; color by action at step i
            segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
            # Use min(len(cats), len(segs))
            m = min(len(cats), len(segs))
            colors = [ACTION_COLORS[c] for c in cats[:m]]
            lc = LineCollection(segs[:m], colors=colors, linewidths=1.6,
                                alpha=0.95)
            ax.add_collection(lc)
            ax.set_xlim(0, VIEWPORT_W); ax.set_ylim(VIEWPORT_H, 0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{name} ep{col}  T={ep['length']}", fontsize=9)
    handles = [plt.Line2D([], [], color=c, lw=3, label=ACTION_LABELS[k])
               for k, c in ACTION_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("V_A — action-colored single trajectories")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)


# ---- Variant B: 50-overlay + reversal dots ----
def variant_B(blind_eps, labeled_eps, backdrop, out: Path, t_max: int,
              stats=None):
    """stats: dict {'Blind': {'landed':int,'total':int}, 'Labeled': {...}}.
    If provided, prints landing % in titles. Uses the entire population for
    landing rate; the supplied eps lists are used only for the plotted dots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), dpi=150,
                             constrained_layout=True)
    MAIN_DOT = "#ffd400"      # yellow — main thrust on/off
    LAT_DOT = "#e040fb"       # magenta — lateral L/R/none change
    for ax, eps, title in [
        (axes[0], blind_eps, "Blind"),
        (axes[1], labeled_eps, "Labeled"),
    ]:
        ax.imshow(backdrop, extent=[0, VIEWPORT_W, VIEWPORT_H, 0])
        m_x, m_y, l_x, l_y = [], [], [], []
        for ep in eps:
            pts = state_to_pixels(ep["states"])
            T = len(pts)
            if T < 2:
                continue
            segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
            ax.add_collection(LineCollection(segs, colors="#bbbbbb",
                                             linewidths=0.5, alpha=0.35))
            m_rev = thrust_reversals(ep["actions"], "main")
            l_rev = thrust_reversals(ep["actions"], "lateral")
            for i in np.where(m_rev)[0]:
                if i + 1 < len(pts):
                    m_x.append(pts[i + 1, 0]); m_y.append(pts[i + 1, 1])
            for i in np.where(l_rev)[0]:
                if i + 1 < len(pts):
                    l_x.append(pts[i + 1, 0]); l_y.append(pts[i + 1, 1])
        ax.scatter(l_x, l_y, s=2.5, c=LAT_DOT, alpha=0.55,
                   linewidths=0, zorder=9)
        ax.scatter(m_x, m_y, s=2.5, c=MAIN_DOT, alpha=0.65,
                   linewidths=0, zorder=10)
        ax.set_xlim(0, VIEWPORT_W); ax.set_ylim(VIEWPORT_H, 0)
        ax.set_xticks([]); ax.set_yticks([])
        n_eps = len(eps)
        m_rate = np.mean([thrust_reversals(ep['actions'], 'main').sum()
                          / max(len(ep['actions']), 1) for ep in eps])
        l_rate = np.mean([thrust_reversals(ep['actions'], 'lateral').sum()
                          / max(len(ep['actions']), 1) for ep in eps])
        if stats and title in stats:
            s = stats[title]
            land_str = f"  landed={s['landed']}/{s['total']} ({100*s['landed']/s['total']:.0f}%)"
        else:
            land_str = ""
        ax.set_title(
            f"{title} agent{land_str}\n"
            f"main reversals: {m_rate:.2f}/step   "
            f"lateral reversals: {l_rate:.2f}/step",
            fontsize=11, pad=10,
        )
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=MAIN_DOT, markeredgecolor=MAIN_DOT,
                   markersize=9, label="main thrust on/off"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=LAT_DOT, markeredgecolor=LAT_DOT,
                   markersize=9, label="lateral L/R/none change"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=2,
               frameon=False, fontsize=10)
    fig.suptitle("When does the agent reverse its thrust?",
                 fontsize=13, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)


# ---- Variant C: line width modulated by |Δaction| ----
def variant_C(blind_eps, labeled_eps, backdrop, out: Path, t_max: int):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for ax, eps, cmap_name, title in [
        (axes[0], blind_eps, "Oranges", f"Blind (n={len(blind_eps)})"),
        (axes[1], labeled_eps, "Blues", f"Labeled (n={len(labeled_eps)})"),
    ]:
        cmap = plt.get_cmap(cmap_name)
        ax.imshow(backdrop, extent=[0, VIEWPORT_W, VIEWPORT_H, 0])
        for ep in eps:
            pts = state_to_pixels(ep["states"])
            T = len(pts)
            if T < 3:
                continue
            segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
            actions = ep["actions"]
            da = np.zeros(len(actions))
            da[1:] = np.linalg.norm(np.diff(actions, axis=0), axis=1)
            # normalize to 0..1, then map to width 0.4..2.5
            da = np.clip(da / 1.5, 0, 1)
            lw = 0.4 + 2.1 * da
            abs_t = np.arange(T - 1) / max(t_max - 1, 1)
            colors = cmap(0.2 + 0.8 * abs_t)
            m = min(len(segs), len(lw))
            ax.add_collection(LineCollection(segs[:m], colors=colors[:m],
                                             linewidths=lw[:m], alpha=0.4))
        ax.set_xlim(0, VIEWPORT_W); ax.set_ylim(VIEWPORT_H, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)
    fig.suptitle("V_C — line width ∝ |Δaction|  (thicker = bigger control change)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)


# ---- Variant D: overlay + thrust strips ----
def variant_D(blind_eps, labeled_eps, backdrop, out: Path, t_max: int,
              n_strips: int = 20, stats=None):
    fig = plt.figure(figsize=(14, 10), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 2])
    for col, (eps, cmap_name, title) in enumerate([
        (blind_eps, "Oranges", "Blind"),
        (labeled_eps, "Blues", "Labeled"),
    ]):
        n_eps = len(eps)
        if stats and title in stats:
            s = stats[title]
            land_str = f"   landed {s['landed']}/{s['total']} ({100*s['landed']/s['total']:.0f}%)"
        else:
            land_str = ""
        ax = fig.add_subplot(gs[0, col])
        cmap = plt.get_cmap(cmap_name)
        ax.imshow(backdrop, extent=[0, VIEWPORT_W, VIEWPORT_H, 0])
        for ep in eps:
            pts = state_to_pixels(ep["states"])
            T = len(pts)
            if T < 2:
                continue
            segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
            abs_t = np.arange(T - 1) / max(t_max - 1, 1)
            colors = cmap(0.15 + 0.85 * abs_t)
            ax.add_collection(LineCollection(segs, colors=colors,
                                             linewidths=0.7, alpha=0.35))
        ax.set_xlim(0, VIEWPORT_W); ax.set_ylim(VIEWPORT_H, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"$\\bf{{{title}\\ agent}}$ (n={n_eps}){land_str}",
                     fontsize=12, pad=10)

        ax2 = fig.add_subplot(gs[1, col])
        # Sort all eps by length ascending so the strip plot forms a staircase
        # — the triangle directly encodes the episode-duration distribution.
        sorted_eps = sorted(eps, key=lambda e: e["length"])
        n_rows = len(sorted_eps)
        img = np.full((n_rows, t_max), -1, dtype=int)
        for r, ep in enumerate(sorted_eps):
            cats = discretize_actions(ep["actions"])
            img[r, :len(cats)] = cats
        # Render as colored cells via imshow with categorical cmap-like trick:
        # use a discrete colormap
        rgb = np.ones((*img.shape, 3), dtype=float)
        for cat, hex_c in ACTION_COLORS.items():
            mask = img == cat
            rgb[mask] = np.array(plt.matplotlib.colors.to_rgb(hex_c))
        ax2.imshow(rgb, aspect="auto", interpolation="nearest")
        # Outcome marker at the end of each strip
        good_x, good_y, bad_x, bad_y = [], [], [], []
        for r, ep in enumerate(sorted_eps):
            x = ep["length"] + 8  # nudge marker just past strip end
            if ep["outcome"] == "landed":
                good_x.append(x); good_y.append(r)
            else:
                bad_x.append(x); bad_y.append(r)
        ax2.scatter(good_x, good_y, marker=">", s=22, c="#2ecc71",
                    linewidths=0, zorder=5)
        ax2.scatter(bad_x, bad_y, marker=">", s=22, c="#e74c3c",
                    linewidths=0, zorder=5)
        ax2.set_xlim(0, t_max + 25)
        ax2.set_xlabel("step")
        ax2.set_ylabel("episode (sorted by length)")
        ax2.set_title(f"per-step thrust action — {n_rows} episodes "
                      f"(sorted shortest → longest;  ▶ green = landed,  "
                      f"▶ red = failed)",
                      fontsize=10, pad=8)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="",
                          markerfacecolor=c, markeredgecolor=c,
                          markersize=10, label=ACTION_LABELS[k])
               for k, c in ACTION_COLORS.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=6,
               frameon=False, fontsize=10)
    fig.suptitle("Landing trajectories and per-step thrust action over time",
                 fontsize=13, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)


def landing_stats(traj_dir: Path, n_pop=200):
    """Survey the first n_pop episodes regardless of outcome."""
    eps = pick(traj_dir, n_pop, outcome=None)
    landed = sum(1 for ep in eps if ep["outcome"] == "landed")
    return {"landed": landed, "total": len(eps)}


def render_scenario(tag, blind_dir, labeled_dir, rgb_path, out_dir,
                    n=50, outcome_filter="landed"):
    backdrop = load_episode(rgb_path)["rgb_frames"][-1]
    blind = pick(blind_dir, n, outcome=outcome_filter)
    labeled = pick(labeled_dir, n, outcome=outcome_filter)
    if not labeled:
        print(f"[{tag}] no labeled episodes match outcome={outcome_filter}; "
              f"falling back to all outcomes")
        labeled = pick(labeled_dir, n, outcome=None)
    if not blind:
        blind = pick(blind_dir, n, outcome=None)
    stats = {
        "Blind": landing_stats(blind_dir),
        "Labeled": landing_stats(labeled_dir),
    }
    print(f"[{tag}] loaded {len(blind)} blind, {len(labeled)} labeled  "
          f"(filter={outcome_filter})  landing pop-stats={stats}")
    t_max = max(max(ep["length"] for ep in blind),
                max(ep["length"] for ep in labeled))
    print(f"[{tag}] t_max={t_max}")
    variant_B(blind, labeled, backdrop, out_dir / f"{tag}_vB.png", t_max,
              stats=stats)
    variant_D(blind, labeled, backdrop, out_dir / f"{tag}_vD.png", t_max,
              stats=stats)


def main():
    base = Path("/media/hdd1/physics-priors-latent-space/lunar-lander-networks/rl_agents/full-variation")
    rgb_path = base / "labeled-ppo-easy-128-lowent-Feb122026/trajectories-rgb/episode_0000.npz"
    out_dir = Path.home() / "vsr-tmp/e1-overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scenario 1: easy in-distribution (already rendered as vA/vB/vC/vD; redo as easy-ID_*)
    render_scenario(
        "easy-ID",
        base / "blind-ppo-easy-128-lowent-Feb122026/trajectories",
        base / "labeled-ppo-easy-128-lowent-Feb122026/trajectories",
        rgb_path, out_dir,
        outcome_filter=None,
    )
    # Scenario 2: easy-trained agents OOD on medium physics
    # Labeled lands only 6/100 here — must keep all outcomes to have something to plot
    render_scenario(
        "easy-OOD-medium",
        base / "blind-ppo-easy-128-lowent-Feb122026/trajectories-ood-medium",
        base / "labeled-ppo-easy-128-lowent-Feb122026/trajectories-ood-medium",
        rgb_path, out_dir,
        outcome_filter=None,  # mostly crashes; show everything
    )
    # Scenario 3: medium-trained agents in-distribution
    render_scenario(
        "medium-ID",
        base / "blind-ppo-medium-128-lowent-Feb122026/trajectories",
        base / "labeled-ppo-medium-128-lowent-Feb122026/trajectories",
        rgb_path, out_dir,
        outcome_filter=None,
    )


if __name__ == "__main__":
    main()
