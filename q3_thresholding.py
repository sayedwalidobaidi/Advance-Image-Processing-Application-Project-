"""
Assignment 2 - Part A - Question 3: Image thresholding
  1. Global threshold, value chosen by hand
  2. Global threshold, value chosen from the histogram
     (lecture slide 8: cumulative histogram, S = grey level below which P% of pixels lie;
      Otsu's method is also shown for comparison)
  3. Local threshold, max-min method (lecture slide 7): S(i,j) = (max + min) / 2 in an N x M window

Run:  py q3_thresholding.py
Figures are shown on screen and saved in the "results" folder next to this file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter
from skimage.filters import threshold_otsu

from q1_edge_detection import IMAGE_PATH, OUT_DIR, load_gray

# ---------------------------------------------------------------
# Settings (change these and re-run to see the effect)
# ---------------------------------------------------------------
MANUAL_T = 0.5                       # manual global threshold (0..1)
MANUAL_T_TRIALS = [0.3, 0.5, 0.7]    # to show what happens when T goes down / up
PERCENT_BELOW = 80                   # lecture method: P% of pixels are below S
PERCENT_TRIALS = [30, 50, 80]
WINDOW_SIZE = 15                     # local max-min window N x M (odd number)
WINDOW_TRIALS = [3, 15, 31]
MIN_CONTRAST = 0.15                  # local method: if max - min < this, the window is flat


# ---------------------------------------------------------------
# Step 1: global threshold (lecture slide 7)
#   I(i,j) = 1 if f(i,j) > S, else 0     -> one S for the whole image
# ---------------------------------------------------------------
def global_threshold(img, s):
    return img > s


# ---------------------------------------------------------------
# Step 2: choose S from the histogram
# ---------------------------------------------------------------
def histogram(img, bins=256):
    counts, edges = np.histogram(img, bins=bins, range=(0, 1))
    levels = (edges[:-1] + edges[1:]) / 2          # centre grey level of each bin
    return levels, counts


def threshold_from_cumulative(img, percent):
    """Lecture slide 8: build the cumulative histogram (in %) and read off the
    grey level S where the curve reaches `percent`."""
    levels, counts = histogram(img)
    cumulative = 100 * np.cumsum(counts) / counts.sum()
    return levels[np.searchsorted(cumulative, percent)]


# Otsu (for comparison): tries every possible S and keeps the one that best
# separates the histogram into two groups (largest between-class variance).
# Works well when the histogram has two peaks (dark objects + bright background).


# ---------------------------------------------------------------
# Step 3: local threshold, max-min method (lecture slide 7)
#   S(i,j) = (max(i,j) + min(i,j)) / 2, max/min taken in an N x M window around (i,j)
# ---------------------------------------------------------------
def local_maxmin_threshold(img, size, min_contrast=0.0, fallback_t=None):
    local_max = maximum_filter(img, size=size, mode="nearest")
    local_min = minimum_filter(img, size=size, mode="nearest")
    s = (local_max + local_min) / 2                # a different threshold for every pixel
    result = img > s
    if min_contrast > 0:
        # In a flat window max ~ min, so S ~ the pixel itself and the result is
        # decided by tiny noise. There, fall back to the global threshold instead.
        flat = (local_max - local_min) < min_contrast
        result[flat] = img[flat] > fallback_t
    return result, s


# ---------------------------------------------------------------
# Step 4: display and save
# ---------------------------------------------------------------
def show_histogram(img, t_manual, t_cum, t_otsu):
    levels, counts = histogram(img)
    cumulative = 100 * np.cumsum(counts) / counts.sum()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(levels, counts, width=1 / 256, color="gray")
    axes[0].set_title("Histogram")
    axes[1].plot(levels, cumulative, color="black")
    axes[1].axhline(PERCENT_BELOW, color="tab:red", ls=":")
    axes[1].set_title(f"Cumulative histogram (S where {PERCENT_BELOW}% of pixels are below)")
    axes[1].set_ylabel("% of pixels")
    for ax in axes:
        ax.axvline(t_manual, color="tab:blue", ls="--", label=f"manual T = {t_manual:.2f}")
        ax.axvline(t_cum, color="tab:red", ls="--", label=f"cumulative {PERCENT_BELOW}%: S = {t_cum:.2f}")
        ax.axvline(t_otsu, color="tab:green", ls="--", label=f"Otsu: S = {t_otsu:.2f}")
        ax.set_xlabel("grey level")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "q3_histogram.png", dpi=150, bbox_inches="tight")


def show_row(axes, img, panels):
    for ax, (title, im) in zip(axes, panels):
        ax.imshow(im, cmap="gray")
        if im.dtype == bool:
            title += f"\nwhite: {100 * im.mean():.0f}%"
        ax.set_title(title, fontsize=10)
        ax.axis("off")


def show_global(img, t_otsu):
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.5), layout="constrained")
    show_row(axes[0], img, [("Original", img)] +
             [(f"Manual T = {t}", global_threshold(img, t)) for t in MANUAL_T_TRIALS])
    hist_panels = [(f"Cumulative {p}%: S = {threshold_from_cumulative(img, p):.2f}",
                    global_threshold(img, threshold_from_cumulative(img, p))) for p in PERCENT_TRIALS]
    show_row(axes[1], img, hist_panels + [(f"Otsu: S = {t_otsu:.2f}", global_threshold(img, t_otsu))])
    fig.suptitle("Global thresholding: manual (top) and histogram-based (bottom)", fontsize=13)
    fig.savefig(OUT_DIR / "q3_global.png", dpi=150, bbox_inches="tight")


def show_local(img, t_otsu):
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.5), layout="constrained")
    first = local_maxmin_threshold(img, WINDOW_SIZE)[1]
    show_row(axes[0], img, [(f"Local threshold S(i,j)\n{WINDOW_SIZE}x{WINDOW_SIZE} window", first)] +
             [(f"Max-min, {n}x{n} (lecture formula only)", local_maxmin_threshold(img, n)[0])
              for n in WINDOW_TRIALS])
    show_row(axes[1], img, [("Original", img)] +
             [(f"Max-min, {n}x{n}\n+ contrast check ({MIN_CONTRAST})",
               local_maxmin_threshold(img, n, MIN_CONTRAST, t_otsu)[0]) for n in WINDOW_TRIALS])
    fig.suptitle("Local max-min thresholding: effect of window size", fontsize=13)
    fig.savefig(OUT_DIR / "q3_local.png", dpi=150, bbox_inches="tight")


def show_comparison(img, t_cum, t_otsu):
    local, _ = local_maxmin_threshold(img, WINDOW_SIZE, MIN_CONTRAST, t_otsu)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8))
    show_row(axes, img, [("Original", img),
                         (f"Global, manual T = {MANUAL_T}", global_threshold(img, MANUAL_T)),
                         (f"Global, cumulative {PERCENT_BELOW}%: S = {t_cum:.2f}", global_threshold(img, t_cum)),
                         (f"Global, Otsu: S = {t_otsu:.2f}", global_threshold(img, t_otsu)),
                         (f"Local max-min {WINDOW_SIZE}x{WINDOW_SIZE}", local)])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "q3_comparison.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    image = load_gray(IMAGE_PATH)

    t_cum = threshold_from_cumulative(image, PERCENT_BELOW)
    t_otsu = threshold_otsu(image)
    print(f"Manual threshold:              T = {MANUAL_T:.3f}")
    print(f"Cumulative histogram ({PERCENT_BELOW}%):   S = {t_cum:.3f}")
    print(f"Otsu:                          S = {t_otsu:.3f}")
    for n in WINDOW_TRIALS:
        res, _ = local_maxmin_threshold(image, n)
        res_c, _ = local_maxmin_threshold(image, n, MIN_CONTRAST, t_otsu)
        print(f"Local max-min {n:2d}x{n:<2d}  white: {100 * res.mean():4.1f}%   "
              f"with contrast check: {100 * res_c.mean():4.1f}%")

    show_histogram(image, MANUAL_T, t_cum, t_otsu)
    show_global(image, t_otsu)
    show_local(image, t_otsu)
    show_comparison(image, t_cum, t_otsu)
    print(f"Figures saved in: {OUT_DIR}")
    plt.show()
