"""Static figures exported with the results (PNG 300 dpi + PDF, IEEE double column).

* recall_ci: per slice, episode recall at 1 % FPR with the 95 % bootstrap CI, one
  row per detector (identity by the row label; GRAPHIOT emphasised in blue, the
  baselines in neutral ink, so the figure survives grayscale printing);
* pr_curves: per slice, precision-recall curves of the continuous detectors with
  direct labels and distinct line styles; binary detectors (B1, B4) as their single
  operating point; dotted line = prevalence.
Palette: first four slots of the dataviz reference instance, validated (light).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLES = ["-", "--", "-.", ":"]
ORDER = ["full", "b0", "b1", "b2", "b3", "b4", "b5"]
SLICE_TITLES = {"all": "All harmful commands", "allow": "Passing the allow-list", "ctx": "Context-shifted"}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white", "pdf.fonttype": 42})
    return plt


def recall_ci(res: dict, out: Path, labels: dict[str, str]) -> list[Path]:
    plt = _style()
    names = [n for n in ORDER if n in res["point"]]
    if not names:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 0.32 * len(names) + 0.9), sharey=True)
    for ax, sl in zip(axes, ("all", "allow", "ctx")):
        for i, n in enumerate(names):
            p = res["point"][n][sl]["recall"]
            lo, hi = res["ci"][n][sl]["recall"]
            y = len(names) - 1 - i
            color = SLOTS[0] if n == "full" else MUTED
            if p is None or p != p:
                ax.text(2, y, "n/a", va="center", ha="left", color=MUTED, fontsize=6.5)
                continue
            if lo == lo:
                ax.plot([100 * lo, 100 * hi], [y, y], color=color, lw=1.4, solid_capstyle="round")
            ax.plot([100 * p], [y], "o", ms=4.5, color=color, mec="white", mew=0.8, zorder=3, clip_on=False)
            # value just above its own dot, kept inside the panel
            x = min(max(100 * p, 4), 96)
            ax.text(x, y + 0.18, f"{100 * p:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, len(names) - 0.35)
        ax.set_title(SLICE_TITLES[sl], fontsize=7.5, color=INK)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Episode recall at 1 % FPR (%)")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels([labels.get(n, n) for n in reversed(names)])
    fig.tight_layout()
    return _save(fig, out / "recall_ci", plt)


def pr_curves(t, dets: dict, masks: dict, out: Path, labels: dict[str, str]) -> list[Path]:
    from sklearn.metrics import precision_recall_curve

    from .metrics import finite_scores
    plt = _style()
    cont = [n for n in ORDER if n in dets and n not in ("b1", "b4")][:4]
    binary = [n for n in ("b1", "b4") if n in dets]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7), sharey=True)
    for ax, sl in zip(axes, ("all", "allow", "ctx")):
        mask, y = masks[sl]
        yy = y[mask].astype(int)
        ax.set_title(SLICE_TITLES[sl], fontsize=7.5, color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall")
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        if not (0 < yy.sum() < len(yy)):
            ax.text(0.5, 0.5, "no positives in this slice", ha="center", va="center", color=MUTED, fontsize=7)
            continue
        ax.axhline(yy.mean(), color=MUTED, lw=0.8, ls=":")
        for i, n in enumerate(cont):
            s = np.nanmean(np.stack([d.score for d in dets[n]]), axis=0)[mask]
            keep = ~np.isnan(s)
            if keep.sum() < 2 or not (0 < yy[keep].sum() < keep.sum()):
                continue
            prec, rec, _ = precision_recall_curve(yy[keep], finite_scores(s[keep]))
            ax.plot(rec, prec, STYLES[i], color=SLOTS[i], lw=1.5 if n == "full" else 1.2, drawstyle="steps-post",
                    label=labels.get(n, n))
            # direct label at a different recall per curve, below the curve when it hugs the top
            j = int(np.argmin(np.abs(rec - (0.2 + 0.2 * i))))
            ax.annotate(labels.get(n, n).split(" ")[0], (rec[j], prec[j]), xytext=(3, -9 if prec[j] > 0.9 else 3),
                        textcoords="offset points", fontsize=6.5, color=INK)
        for n in binary:
            a = dets[n][0].alarm[mask]
            tp = int((a & (yy == 1)).sum())
            if a.sum():
                ax.plot([tp / yy.sum()], [tp / a.sum()], "s", ms=4.5, color=MUTED, mec="white", mew=0.8)
                ax.annotate(labels.get(n, n).split(" ")[0], (tp / yy.sum(), tp / a.sum()), xytext=(4, -8),
                            textcoords="offset points", fontsize=6.5, color=INK)
    axes[0].set_ylabel("Precision")
    handles, labs = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labs, loc="lower center", ncol=len(handles), frameon=False, fontsize=6.5,
                   bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.07, 1, 1))
    else:
        fig.tight_layout()
    return _save(fig, out / "pr_curves", plt)


def _save(fig, stem: Path, plt) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths
