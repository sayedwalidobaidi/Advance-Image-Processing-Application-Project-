"""
Assignment 2 - Part B - Question 5: Spatial filtering of the salt-and-pepper image
  1. Average (mean) filter      - like MATLAB filter2(fspecial('average', n), g)
  2. Median filter              - like MATLAB medfilt2(g, [n n])
  3. Spatial (adaptive) Wiener  - like MATLAB wiener2(g, [n n])
Each filter is tried with several neighbourhood sizes.

Run:  py q5_spatial_filters.py
Figures are shown on screen and saved in the "results" folder next to this file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, median_filter

from q1_edge_detection import IMAGE_PATH, OUT_DIR, load_gray
from q4_degradation import make_salt_pepper, ZOOM

# ---------------------------------------------------------------
# Settings (change these and re-run to see the effect)
# ---------------------------------------------------------------
SIZES = [3, 5, 7]        # neighbourhood sizes n x n to try
MAIN_SIZE = 3            # size used in the main comparison figure


# ---------------------------------------------------------------
# Step 1: average filter
# Replace each pixel by the MEAN of the n x n pixels around it.
# Linear: every neighbour counts equally, including the 0/1 noise pixels.
# ---------------------------------------------------------------
def average_filter(img, n):
    return uniform_filter(img, size=n, mode="nearest")


# ---------------------------------------------------------------
# Step 2: median filter
# Sort the n x n pixels around each pixel and take the MIDDLE value.
# Non-linear: an extreme value (0 or 1) ends up at the end of the sorted
# list, so it is simply ignored instead of being averaged in.
# ---------------------------------------------------------------
def median_filter_n(img, n):
    return median_filter(img, size=n, mode="nearest")


# ---------------------------------------------------------------
# Step 3: spatial (adaptive) Wiener filter, same formula as MATLAB wiener2
#   local mean      mu(i,j)      = mean of the n x n window
#   local variance  s2(i,j)      = mean of g^2 in the window - mu^2
#   noise variance  v2           = average of all local variances (estimated)
#   output = mu + max(s2 - v2, 0) / max(s2, v2) * (g - mu)
# Flat area (s2 ~ v2):   the fraction is ~0  -> output ~ mu   (strong smoothing)
# Edge/detail (s2 >> v2): the fraction is ~1 -> output ~ g    (pixel kept)
# ---------------------------------------------------------------
def spatial_wiener(img, n, noise_var=None):
    mu = uniform_filter(img, size=n, mode="nearest")
    s2 = uniform_filter(img ** 2, size=n, mode="nearest") - mu ** 2
    if noise_var is None:
        noise_var = s2.mean()
    gain = np.maximum(s2 - noise_var, 0) / np.maximum(s2, noise_var)
    return mu + gain * (img - mu), noise_var


# ---------------------------------------------------------------
# Quality measures (used again in Q7)
# ---------------------------------------------------------------
def mse(ref, test):
    return np.mean((ref - np.clip(test, 0, 1)) ** 2)


def psnr(ref, test):
    """PSNR = 10 log10(MAX^2 / MSE), MAX = 1 because the image range is 0..1."""
    return 10 * np.log10(1.0 / mse(ref, test))


def restore_all(noisy, sizes):
    results = {}
    for n in sizes:
        results[("Average", n)] = average_filter(noisy, n)
        results[("Median", n)] = median_filter_n(noisy, n)
        results[("Spatial Wiener", n)] = spatial_wiener(noisy, n)[0]
    return results


# ---------------------------------------------------------------
# Step 4: display and save
# ---------------------------------------------------------------
def show_main(img, noisy, results, n):
    panels = [("Original", img), (f"Salt & pepper", noisy)] + \
             [(f"{name} {n}x{n}", results[(name, n)]) for name in ("Average", "Median", "Spatial Wiener")]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5), layout="constrained")
    for col, (title, im) in enumerate(panels):
        label = title if title == "Original" else f"{title}\nPSNR {psnr(img, im):.1f} dB"
        axes[0, col].imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(label, fontsize=10)
        axes[1, col].imshow(np.clip(im[ZOOM], 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title("Zoom: " + title, fontsize=10)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"Restoring the salt-and-pepper image, {n}x{n} neighbourhood", fontsize=13)
    fig.savefig(OUT_DIR / "q5_filters.png", dpi=150, bbox_inches="tight")


def show_sizes(img, results, sizes):
    names = ("Average", "Median", "Spatial Wiener")
    fig, axes = plt.subplots(len(names), len(sizes), figsize=(5 * len(sizes), 15), layout="constrained")
    for row, name in enumerate(names):
        for col, n in enumerate(sizes):
            im = results[(name, n)]
            axes[row, col].imshow(np.clip(im[ZOOM], 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"{name} {n}x{n}\nMSE {mse(img, im):.4f}   PSNR {psnr(img, im):.1f} dB",
                                     fontsize=10)
            axes[row, col].axis("off")
    fig.suptitle("Effect of the neighbourhood size (zoomed)", fontsize=13)
    fig.savefig(OUT_DIR / "q5_sizes.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    image = load_gray(IMAGE_PATH)
    noisy = make_salt_pepper(image)
    results = restore_all(noisy, SIZES)

    print(f"{'Method':22s} {'MSE':>8s} {'PSNR (dB)':>10s}")
    print(f"{'Noisy (no filter)':22s} {mse(image, noisy):8.4f} {psnr(image, noisy):10.2f}")
    for (name, n), im in results.items():
        print(f"{name + f' {n}x{n}':22s} {mse(image, im):8.4f} {psnr(image, im):10.2f}")
    print(f"Spatial Wiener estimated noise variance (3x3): {spatial_wiener(noisy, 3)[1]:.4f}")

    show_main(image, noisy, results, MAIN_SIZE)
    show_sizes(image, results, SIZES)
    print(f"Figures saved in: {OUT_DIR}")
    plt.show()
