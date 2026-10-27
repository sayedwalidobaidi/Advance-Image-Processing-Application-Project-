"""
Assignment 2 - Part A - Question 1: Gradient-based edge detection
Operators: Roberts, Prewitt, Sobel, Kirsch (masks from lecture chapter 4, slides 2-6).

Run:  py q1_edge_detection.py
Figures are shown on screen and saved in the "results" folder next to this file.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from skimage import data, io, color, img_as_float

# ---------------------------------------------------------------
# Settings (change these and re-run to see the effect)
# ---------------------------------------------------------------
IMAGE_PATH = None        # e.g. r"D:\Master of EEE KBU\AIPA\my_image.jpg"; None = built-in "cameraman"
EDGE_THRESHOLD = 0.25    # 0..1, applied to each normalised edge map to get a black/white edge image
EDGE_PERCENT = 80        # lecture slide 8: S is chosen so that 80% of the gradient values are below it

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------
# STEP 1: load the image as grayscale, values in [0, 1]
# ---------------------------------------------------------------
def load_gray(path):
    if path is None:
        img = data.camera()                      # 512x512 grayscale test image
    else:
        img = io.imread(path)
        if img.ndim == 3:                        # colour photo -> grayscale
            img = color.rgb2gray(img[..., :3])
    return img_as_float(img)                     # uint8 0..255 -> float 0..1


# ---------------------------------------------------------------
# STEP 2: the masks (same values as the lecture slides)
# Gx responds to changes left->right  => shows VERTICAL edges
# Gy responds to changes top->bottom  => shows HORIZONTAL edges
# ---------------------------------------------------------------
ROBERTS_1 = np.array([[0, 1],
                      [-1, 0]])                  # diagonal (NW-SE direction)
ROBERTS_2 = np.array([[1, 0],
                      [0, -1]])                  # diagonal (SW-NE direction)

PREWITT_X = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])
PREWITT_Y = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]])

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],                  # centre row weighted x2 = built-in smoothing
                    [-1, 0, 1]])
SOBEL_Y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

KIRSCH_G = np.array([[-3, -3, 5],
                     [-3, 0, 5],
                     [-3, -3, 5]])               # lecture "horizontal (G)" mask
KIRSCH_D = np.array([[-3, -3, -3],
                     [-3, 0, -3],
                     [5, 5, 5]])                 # lecture "vertical (D)" mask


def kirsch_compass_masks():
    """Kirsch is a compass operator: rotate the ring of 8 outer values
    one step at a time to get 8 masks (N, NE, E, SE, S, SW, W, NW)."""
    ring_positions = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    ring_values = [KIRSCH_G[r, c] for r, c in ring_positions]
    masks = []
    for shift in range(8):
        m = np.zeros((3, 3))
        rolled = np.roll(ring_values, shift)
        for (r, c), v in zip(ring_positions, rolled):
            m[r, c] = v
        masks.append(m)
    return masks


# ---------------------------------------------------------------
# STEP 3: apply a mask = slide it over the image and, at every pixel,
# multiply the mask with the pixels under it and add them up.
# correlate() does exactly that (no mask flipping, same as filter2 in MATLAB).
# ---------------------------------------------------------------
def apply_mask(img, mask):
    return correlate(img, mask.astype(float), mode="nearest")


# STEP 4: combine the two directions
def gradient_magnitude(gx, gy):
    """Edge strength |grad f| = sqrt(gx^2 + gy^2)  (lecture eq. V.6)."""
    return np.sqrt(gx ** 2 + gy ** 2)


def normalise(x):
    """Scale to 0..1 so operators with different mask weights can be compared."""
    # WHY: Kirsch weights (5, -3) give values up to ~11, Roberts only ~1. Dividing by the
    #      maximum puts all four on the same 0..1 scale before using one threshold.
    x = np.abs(x)
    return x / x.max() if x.max() > 0 else x


def cumulative_threshold(values, percent, bins=256):
    """Lecture slide 8: build the cumulative histogram (in %) of `values` and return the
    level S below which `percent` % of the values lie."""
    counts, edges = np.histogram(values, bins=bins, range=(0, values.max()))
    cumulative = 100 * np.cumsum(counts) / counts.sum()
    return edges[1:][np.searchsorted(cumulative, percent)]


def edge_detect(img):
    results = {}

    gx, gy = apply_mask(img, ROBERTS_1), apply_mask(img, ROBERTS_2)
    results["Roberts"] = (gx, gy, gradient_magnitude(gx, gy))

    gx, gy = apply_mask(img, PREWITT_X), apply_mask(img, PREWITT_Y)
    results["Prewitt"] = (gx, gy, gradient_magnitude(gx, gy))

    gx, gy = apply_mask(img, SOBEL_X), apply_mask(img, SOBEL_Y)
    results["Sobel"] = (gx, gy, gradient_magnitude(gx, gy))

    # Kirsch: show the two lecture masks as directional responses,
    # and use the MAX over all 8 compass masks as the edge strength.
    gx, gy = apply_mask(img, KIRSCH_G), apply_mask(img, KIRSCH_D)
    compass = np.stack([apply_mask(img, m) for m in kirsch_compass_masks()])
    results["Kirsch"] = (gx, gy, compass.max(axis=0))

    return results


# ---------------------------------------------------------------
# STEP 5: display and save
# ---------------------------------------------------------------
def show_original(img):
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    plt.title("Original grayscale image")
    plt.axis("off")
    plt.savefig(OUT_DIR / "q1_original.png", dpi=150, bbox_inches="tight")


def show_directional(results):
    labels = {"Roberts": ("Diagonal 1 [[0,1],[-1,0]]", "Diagonal 2 [[1,0],[0,-1]]"),
              "Prewitt": ("Gx (vertical edges)", "Gy (horizontal edges)"),
              "Sobel": ("Gx (vertical edges)", "Gy (horizontal edges)"),
              "Kirsch": ("G mask (vertical edges)", "D mask (horizontal edges)")}
    fig, axes = plt.subplots(4, 3, figsize=(11, 14))
    for row, (name, (gx, gy, mag)) in enumerate(results.items()):
        for col, (im, title) in enumerate([(gx, labels[name][0]),
                                           (gy, labels[name][1]),
                                           (mag, "Magnitude")]):
            ax = axes[row, col]
            ax.imshow(normalise(im), cmap="gray")
            ax.set_title(f"{name}: {title}", fontsize=9)
            ax.axis("off")
    fig.suptitle("Directional responses and edge magnitude", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "q1_directional.png", dpi=150, bbox_inches="tight")


def show_side_by_side(results, threshold, percent):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), layout="constrained")
    for col, (name, (_, _, mag)) in enumerate(results.items()):
        m = normalise(mag)
        edges = m > threshold                                   # I(i,j) = 1 if > S, else 0
        s = cumulative_threshold(m, percent)
        edges_hist = m > s
        axes[0, col].imshow(m, cmap="gray")
        axes[0, col].set_title(f"{name} - edge strength")
        axes[1, col].imshow(edges, cmap="gray")
        axes[1, col].set_title(f"{name} - fixed T = {threshold}\n"
                               f"edge pixels: {100 * edges.mean():.1f}%")
        axes[2, col].imshow(edges_hist, cmap="gray")
        axes[2, col].set_title(f"{name} - histogram {percent}%: S = {s:.3f}\n"
                               f"edge pixels: {100 * edges_hist.mean():.1f}%")
        for ax in axes[:, col]:
            ax.axis("off")
    fig.savefig(OUT_DIR / "q1_comparison.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    image = load_gray(IMAGE_PATH)
    print(f"Image size: {image.shape}, min = {image.min():.2f}, max = {image.max():.2f}")

    edge_results = edge_detect(image)
    for op, (_, _, magnitude) in edge_results.items():
        share = 100 * (normalise(magnitude) > EDGE_THRESHOLD).mean()
        s = cumulative_threshold(normalise(magnitude), EDGE_PERCENT)
        print(f"{op:8s} max strength = {magnitude.max():6.2f}   edge pixels at T={EDGE_THRESHOLD}: {share:5.1f}%"
              f"   histogram {EDGE_PERCENT}% threshold S = {s:.3f}")

    show_original(image)
    show_directional(edge_results)
    show_side_by_side(edge_results, EDGE_THRESHOLD, EDGE_PERCENT)
    print(f"Figures saved in: {OUT_DIR}")
    plt.show()
