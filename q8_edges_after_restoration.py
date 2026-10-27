"""
Assignment 2 - Part C - Question 8: Effect of restoration on edge detection
Sobel on (1) the original, (2) the salt-and-pepper image, (3) the median-filtered image.

Run:  py q8_edges_after_restoration.py
Figures are shown on screen and saved in the "results" folder next to this file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from q1_edge_detection import IMAGE_PATH, OUT_DIR, load_gray, apply_mask, gradient_magnitude, SOBEL_X, SOBEL_Y
from q4_degradation import make_salt_pepper, ZOOM
from q5_spatial_filters import median_filter_n, MAIN_SIZE

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
EDGE_THRESHOLD = 0.25    # fraction of the ORIGINAL image's max Sobel strength
TOLERANCE = 1            # an edge counts as "matching" if it is within this many pixels


# ---------------------------------------------------------------
# STEP 1: Sobel magnitude (same masks as Q1)
# ---------------------------------------------------------------
def sobel(img):
    return gradient_magnitude(apply_mask(img, SOBEL_X), apply_mask(img, SOBEL_Y))


# ---------------------------------------------------------------
# STEP 2: compare each edge map with the edges of the original
#   false edges  = edges found here but NOT in the original   (created by noise)
#   missed edges = edges of the original NOT found here       (lost)
# A small tolerance allows an edge to be shifted by 1 pixel.
# ---------------------------------------------------------------
def compare(ref_edges, edges, tol=TOLERANCE):
    ref_near = binary_dilation(ref_edges, iterations=tol)
    near = binary_dilation(edges, iterations=tol)
    false_edges = edges & ~ref_near
    missed = ref_edges & ~near
    return false_edges, missed


if __name__ == "__main__":
    image = load_gray(IMAGE_PATH)
    noisy = make_salt_pepper(image)
    restored = median_filter_n(noisy, MAIN_SIZE)
    inputs = [("Original", image), ("Salt & pepper", noisy), (f"Median {MAIN_SIZE}x{MAIN_SIZE}", restored)]

    # Same absolute threshold for all three, taken from the original image, so the
    # comparison is fair (normalising each image by its own max would hide the noise).
    strengths = [sobel(im) for _, im in inputs]
    limit = EDGE_THRESHOLD * strengths[0].max()
    edges = [s > limit for s in strengths]

    print(f"Sobel threshold = {EDGE_THRESHOLD} x original max = {limit:.3f}")
    print(f"{'Image':16s} {'edge px %':>10s} {'false %':>9s} {'missed %':>9s}")
    stats = []
    for (name, _), e in zip(inputs, edges):
        false_e, missed = compare(edges[0], e)
        f_pct = 100 * false_e.sum() / max(e.sum(), 1)          # share of this map's edges that are false
        m_pct = 100 * missed.sum() / max(edges[0].sum(), 1)    # share of the original edges that are lost
        stats.append((false_e, missed, f_pct, m_pct))
        print(f"{name:16s} {100 * e.mean():10.1f} {f_pct:9.1f} {m_pct:9.1f}")

    # Figure 1: the three edge images side by side (what the question asks for)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    for col, ((name, im), e, (_, _, f_pct, m_pct)) in enumerate(zip(inputs, edges, stats)):
        axes[0, col].imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(name)
        axes[1, col].imshow(e, cmap="gray")
        axes[1, col].set_title(f"Sobel edges of: {name}\nedge pixels {100 * e.mean():.1f}%, "
                               f"false {f_pct:.0f}%, missed {m_pct:.0f}%", fontsize=10)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"Q8: Sobel edge detection before and after restoration (T = {EDGE_THRESHOLD} x original max)",
                 fontsize=13)
    fig.savefig(OUT_DIR / "q8_sobel.png", dpi=150, bbox_inches="tight")

    # Figure 2: zoom, with false edges in red and missed edges in blue
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
    for ax, (name, _), e, (false_e, missed, _, _) in zip(axes, inputs, edges, stats):
        rgb = np.zeros(e.shape + (3,))
        rgb[e] = 1                                   # white = edge that matches the original
        rgb[false_e] = [1, 0.2, 0.2]                 # red   = false edge (not in the original)
        rgb[missed] = [0.3, 0.5, 1]                  # blue  = original edge that was lost
        ax.imshow(rgb[ZOOM])
        ax.set_title(f"Zoom: {name}\nwhite = correct, red = false, blue = missed", fontsize=10)
        ax.axis("off")
    fig.savefig(OUT_DIR / "q8_false_missed.png", dpi=150, bbox_inches="tight")

    print(f"Figures saved in: {OUT_DIR}")
    plt.show()
