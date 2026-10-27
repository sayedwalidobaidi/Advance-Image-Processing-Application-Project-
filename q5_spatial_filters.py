"""
Assignment 2 - Part B - Question 5: Spatial filtering of the salt-and-pepper image
  1. Average (mean) filter
  2. Median filter
  3. Spatial (adaptive) Wiener filter
Each filter is tried with several neighbourhood sizes.

Run:  py q5_spatial_filters.py
Figures are shown on screen and saved in the "results" folder next to this file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, median_filter

from q1_edge_detection import IMAGE_PATH, OUT_DIR, load_gray, apply_mask
from q4_degradation import make_salt_pepper, ZOOM

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
SIZES = [3, 5, 7]        # window sizes n x n
MAIN_SIZE = 3            # size for the main figure


# ---------------------------------------------------------------
# STEP 1: average filter, mask = ones(n, n) / n^2
# ---------------------------------------------------------------
def average_filter(img, n):
    mask = np.ones((n, n)) / (n * n)
    return apply_mask(img, mask)


# ---------------------------------------------------------------
# STEP 2: median filter, middle value of the sorted window
# ---------------------------------------------------------------
def median_filter_n(img, n):
    return median_filter(img, size=n, mode="nearest")


# ---------------------------------------------------------------
# STEP 3: spatial Wiener filter
#   output = mu + max(s2 - v2, 0) / max(s2, v2) * (g - mu)
#   mu, s2 = local mean and variance, v2 = noise variance (mean of s2)
# ---------------------------------------------------------------
def spatial_wiener(img, n, noise_var=None):
    mu = uniform_filter(img, size=n, mode="nearest")
    s2 = uniform_filter(img ** 2, size=n, mode="nearest") - mu ** 2
    if noise_var is None:
        noise_var = s2.mean()
    gain = np.maximum(s2 - noise_var, 0) / np.maximum(s2, noise_var)
    return mu + gain * (img - mu), noise_var


# ---------------------------------------------------------------
# STEP 4: MSE (0..255 scale) and PSNR = 10 log10(255^2 / MSE)
# ---------------------------------------------------------------
def mse(ref, test):
    return np.mean((255 * (ref - np.clip(test, 0, 1))) ** 2)


def psnr(ref, test):
    return 10 * np.log10(255 ** 2 / mse(ref, test))


def restore_all(noisy, sizes):
    results = {}
    for n in sizes:
        results[("Average", n)] = average_filter(noisy, n)
        results[("Median", n)] = median_filter_n(noisy, n)
        results[("Spatial Wiener", n)] = spatial_wiener(noisy, n)[0]
    return results


# ---------------------------------------------------------------
# STEP 5: display and save
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
            axes[row, col].set_title(f"{name} {n}x{n}\nMSE {mse(img, im):.1f}   PSNR {psnr(img, im):.1f} dB",
                                     fontsize=10)
            axes[row, col].axis("off")
    fig.suptitle("Effect of the neighbourhood size (zoomed)", fontsize=13)
    fig.savefig(OUT_DIR / "q5_sizes.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    image = load_gray(IMAGE_PATH)
    noisy = make_salt_pepper(image)
    results = restore_all(noisy, SIZES)

    print(f"{'Method':22s} {'MSE':>8s} {'PSNR (dB)':>10s}")
    print(f"{'Noisy (no filter)':22s} {mse(image, noisy):8.1f} {psnr(image, noisy):10.2f}")
    for (name, n), im in results.items():
        print(f"{name + f' {n}x{n}':22s} {mse(image, im):8.1f} {psnr(image, im):10.2f}")
    print(f"Spatial Wiener estimated noise variance (3x3): {spatial_wiener(noisy, 3)[1]:.4f}")

    show_main(image, noisy, results, MAIN_SIZE)
    show_sizes(image, results, SIZES)
    print(f"Figures saved in: {OUT_DIR}")
    plt.show()
