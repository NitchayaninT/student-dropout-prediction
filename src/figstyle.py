"""Publication figure style for this project.

Implements the helper API of the `scientific-figure-making` skill from figures4papers
(https://github.com/ChenLiu-1996/figures4papers, CC BY-NC 4.0): palette, rcParams preset,
export helper and a few plot helpers. The skill documents these signatures but ships no
code, so this module is our implementation of them.

    from src.figstyle import FigureStyle, PALETTE, apply_publication_style, finalize_figure
    apply_publication_style(FigureStyle(font_size=16, axes_linewidth=2))
    fig, ax = plt.subplots(figsize=(8, 5))
    ...
    finalize_figure(fig, C.FIGURES_DIR / "my_figure", formats=["png", "pdf"])

Colour roles from the skill: blues for the main result, greens for improvements, reds for
baselines or contrasts, neutral grey for background. Decide project-specific roles (for
example which colour means Dropout) once, here, and reuse them in every script.
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                  # headless and batch runs
import matplotlib.pyplot as plt       # noqa: E402
import numpy as np                    # noqa: E402

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}
DEFAULT_COLORS = [PALETTE[k] for k in
                  ("blue_main", "green_3", "red_strong", "teal", "violet", "neutral")]
EXPORT_FORMATS = {"pdf", "svg", "eps", "png", "jpg", "jpeg", "tif", "tiff"}


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16          # 24 for large bar panels, 15-16 for compact plots
    axes_linewidth: float = 2.5  # 3 for big bars, 2 for compact plots
    use_tex: bool = False        # only when LaTeX is installed and math labels need it
    font_family: tuple = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


def apply_publication_style(style=None):
    """Set rcParams once, before creating figures."""
    s = style or FigureStyle()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(s.font_family),
        "font.size": s.font_size,
        "axes.linewidth": s.axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "text.usetex": s.use_tex,
        "svg.fonttype": "none",       # keep text editable in SVG
        "pdf.fonttype": 42,           # embed TrueType so PDF text stays editable
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def create_subplots(nrows=1, ncols=1, figsize=None, **kwargs):
    """Return (fig, axes) with axes always a flat 1-D array."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, **kwargs)
    return fig, axes.ravel()


def finalize_figure(fig, out_path, formats=None, dpi=300, close=True, pad=0.05, **kwargs):
    """Save `fig` in one or more formats and return the saved paths.

    With formats=None the extension of `out_path` is used, or pdf, svg and eps when it has
    none. Parent folders are created. `pad` is the padding (inches) around the tight box.
    """
    out_path = Path(out_path)
    if formats is None:
        formats = [out_path.suffix.lstrip(".")] if out_path.suffix else ["pdf", "svg", "eps"]
    formats = [f.lower().lstrip(".") for f in formats]
    unknown = set(formats) - EXPORT_FORMATS
    if unknown:
        raise ValueError(f"unsupported format(s) {sorted(unknown)}; use {sorted(EXPORT_FORMATS)}")
    base = out_path.with_suffix("") if out_path.suffix else out_path
    base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        p = base.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=pad, **kwargs)
        saved.append(p)
    if close:
        plt.close(fig)
    return saved


def _as_1d(a, name):
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    return a


def make_grouped_bar(ax, categories, series, labels, ylabel="Value", colors=None,
                     annotate=False):
    """Grouped bars: one group per category, one bar per series. Returns the last BarContainer."""
    series = np.asarray(series, dtype=float)
    if series.ndim != 2 or series.shape[1] != len(categories):
        raise ValueError("series must be (n_series, len(categories))")
    if len(labels) != series.shape[0]:
        raise ValueError("one label per series")
    colors = colors or DEFAULT_COLORS
    x = np.arange(len(categories))
    width = 0.8 / series.shape[0]
    bars = None
    for i, (vals, lab) in enumerate(zip(series, labels)):
        bars = ax.bar(x + (i - (series.shape[0] - 1) / 2) * width, vals, width,
                      label=lab, color=colors[i % len(colors)], edgecolor="black",
                      linewidth=1.5)
        if annotate:
            annotate_bars(ax, bars)
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    return bars


def annotate_bars(ax, bars, fmt="{:.2f}", fontsize=10, padding=3):
    """Write each bar's value above it."""
    ax.bar_label(bars, labels=[fmt.format(b.get_height()) for b in bars],
                 fontsize=fontsize, padding=padding)


def make_trend(ax, x, y_series, labels, colors=None, ylabel=None, xlabel=None,
               show_shadow=True, bands=None):
    """Lines that share an x-axis. `bands` is an optional list of (low, high) arrays per
    line, drawn as an uncertainty band; `show_shadow` draws a soft halo under each line."""
    x = _as_1d(x, "x")
    colors = colors or DEFAULT_COLORS
    for i, (y, lab) in enumerate(zip(y_series, labels)):
        y = _as_1d(y, f"y_series[{i}]")
        if len(y) != len(x):
            raise ValueError(f"y_series[{i}] has length {len(y)}, x has {len(x)}")
        c = colors[i % len(colors)]
        if bands is not None and bands[i] is not None:
            lo, hi = bands[i]
            ax.fill_between(x, lo, hi, color=c, alpha=0.18, lw=0)
        if show_shadow:
            ax.plot(x, y, color=c, lw=6, alpha=0.15, solid_capstyle="round")
        ax.plot(x, y, color=c, lw=2.5, label=lab)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def make_heatmap(ax, matrix, x_labels=None, y_labels=None, cmap="magma", cbar_label=None,
                 annotate=False, fmt="{:.2f}", **imshow_kwargs):
    """2-D heatmap with optional tick labels, colourbar label and cell values."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {m.shape}")
    im = ax.imshow(m, cmap=cmap, aspect="auto", **imshow_kwargs)
    if x_labels is not None:
        ax.set_xticks(range(m.shape[1]), x_labels, rotation=45, ha="right")
    if y_labels is not None:
        ax.set_yticks(range(m.shape[0]), y_labels)
    cb = ax.figure.colorbar(im, ax=ax)
    if cbar_label:
        cb.set_label(cbar_label)
    if annotate:
        lo, hi = np.nanmin(m), np.nanmax(m)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]):
                    dark = (m[i, j] - lo) / (hi - lo + 1e-12) < 0.5
                    ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center",
                            color="white" if dark else "black", fontsize=9)
    return im


def make_scatter(ax, x, y, label=None, color=None, size=50, alpha=0.7, **kwargs):
    """Single-series scatter."""
    x, y = _as_1d(x, "x"), _as_1d(y, "y")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    return ax.scatter(x, y, s=size, alpha=alpha, label=label,
                      color=color or PALETTE["blue_main"], **kwargs)


# ---- project colour roles (decided once, used by every new figure) ------------------
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Outcome: Dropout red, Graduate blue, Enrolled neutral grey (final outcome unknown).
OUTCOME_COLORS = {"Dropout": PALETTE["red_strong"], "Graduate": PALETTE["blue_main"],
                  "Enrolled": "#A8A7A7"}
# Predicted dropout risk 0 -> 1: light grey (visible on white) -> light red -> red -> dark red.
# Luminance falls monotonically, so the scale also reads in greyscale.
RISK_CMAP = LinearSegmentedColormap.from_list(
    "risk", ["#DCDCDC", PALETTE["red_1"], PALETTE["red_2"], PALETTE["red_strong"], "#4A1515"])
# Clusters get hues that are not used for outcomes, so a cluster is never read as one.
CLUSTER_COLORS = [PALETTE["teal"], PALETTE["violet"], "#C9A227", PALETTE["green_3"],
                  "#EA84DD", "#767676", PALETTE["blue_secondary"], PALETTE["red_2"]]
# T0 / T1 / T2 shades for predicted risk markers (light to dark red).
STAGE_RISK_COLORS = {"T0": PALETTE["red_1"], "T1": PALETTE["red_2"], "T2": PALETTE["red_strong"]}
SLIDE_WIDTH = 12.0      # inches: fits a 16:9 slide; at report width (6.3 in) text scales to ~0.5x
SLIDE_STYLE = FigureStyle(font_size=15, axes_linewidth=1.6,
                          font_family=("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"))


def panel_label(ax, letter, x=0.0, y=1.02):
    """Bold panel letter, e.g. (a), just above the axes' top-left corner."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontweight="bold",
            ha="left", va="bottom")


def audit_figure(fig, min_font=9.0):
    """Return layout problems: text below `min_font` pt, text that overlaps other text,
    text running off the canvas, and quantitative axes without a label."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Tick labels just outside the view limits exist as artists but are never drawn.
    offscreen = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            span = (hi - lo) or 1.0
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                if not lo - span * 1e-6 <= tick.get_loc() <= hi + span * 1e-6:
                    offscreen.update({id(tick.label1), id(tick.label2)})
    problems = []
    boxes = []
    for t in fig.findobj(matplotlib.text.Text):
        if id(t) in offscreen or not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_fontsize() < min_font:
            problems.append(f"text below {min_font}pt: {t.get_text()[:30]!r} ({t.get_fontsize():.1f}pt)")
        try:
            box = t.get_window_extent(renderer)
        except (RuntimeError, ValueError):
            continue
        if box.width > 0 and box.height > 0:
            boxes.append((t, box))
    fb = fig.bbox
    off = [t.get_text() for t, b in boxes
           if b.x0 < fb.x0 - 1.5 or b.x1 > fb.x1 + 1.5 or b.y0 < fb.y0 - 1.5 or b.y1 > fb.y1 + 1.5]
    if off:
        problems.append(f"text off the canvas: {sorted(set(off))[:3]}")
    hits = []
    for i, (ta, ba) in enumerate(boxes):
        for tb, bb in boxes[i + 1:]:
            ov = matplotlib.transforms.Bbox.intersection(ba, bb)
            if ov is not None and ov.width > 1.5 and ov.height > 1.5:
                hits.append(f"{ta.get_text()[:20]!r} over {tb.get_text()[:20]!r}")
    if hits:
        problems.append("overlapping text: " + "; ".join(hits[:3])
                        + (f" (+{len(hits) - 3} more)" if len(hits) > 3 else ""))
    for ax in fig.axes:
        if not ax.axison or ax.get_label() == "<colorbar>":
            continue
        for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            ticks = [t.get_text() for t in axis.get_ticklabels() if t.get_text().strip()]
            numeric = ticks and all(t.replace("−", "-").replace(".", "", 1).replace("%", "")
                                    .lstrip("-").isdigit() for t in ticks)
            if numeric and not axis.get_label().get_text():
                problems.append(f"a numeric {name} axis has no label")
    return problems


def save_figure(fig, stem, dpi=200):
    """Write stem.png (slides, README), stem.pdf and stem.svg (report) without trimming,
    so the figure keeps the size it was built at, then print the layout audit."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf", "svg"):
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=dpi if ext == "png" else None)
        paths.append(p)
    problems = audit_figure(fig)
    print(f"[figure] {stem.name}: " + ("layout audit clean" if not problems
                                       else "AUDIT: " + " | ".join(problems)), flush=True)
    plt.close(fig)
    return paths
