"""
Assignment 2 - Part B - Question 5: Spatial filtering of the salt-and-pepper image
  1. Average (mean) filter      - chapter 3 slide 15, MATLAB filter2(fspecial('average', n), g)
  2. Median filter              - chapter 3 slide 19, MATLAB medfilt2(g, [n n])
  3. Spatial (adaptive) Wiener  - MATLAB wiener2(g, [n n]) from the lecture code
Each filter is tried with several neighbourhood sizes.

Run:  py q5_spatial_filters.py
Figures are shown on screen and saved in the "results" folder next to this file.

VIDEO SCRIPT (about 50 seconds) - read the "# SAY:" comments from top to bottom.
  1. Mean filter = linear average (1/9 mask).
  2. Median filter = sort and take the middle (non-linear).
  3. Spatial Wiener = adapts to the local variance.
  4. MSE / PSNR, then compare: which filter wins, edges, window size, "less noise = better?"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, median_filter

from q1_edge_detection import IMAGE_PATH, OUT_DIR, load_gray, apply_mask
from q4_degradation import make_salt_pepper, ZOOM

# ---------------------------------------------------------------
# Settings (change these and re-run to see the effect)
# ---------------------------------------------------------------
SIZES = [3, 5, 7]        # neighbourhood sizes n x n to try
MAIN_SIZE = 3            # size used in the main comparison figure


# ---------------------------------------------------------------
# STEP 1: average (mean) filter - chapter 3 slide 15
# Mask = 1/9 x ones(3,3) (or 1/25 x ones(5,5)): each pixel becomes the MEAN
# of the n x n pixels around it. Linear: every neighbour counts equally,
# including the 0/1 noise pixels.
# ---------------------------------------------------------------
# SAY: "The mean filter uses the mask one ninth times a 3 by 3 block of ones, like slide 15.
#       Each pixel is replaced by the average of its 9 neighbours. It is LINEAR: every
#       neighbour counts equally - including the black and white noise dots."
def average_filter(img, n):
    mask = np.ones((n, n)) / (n * n)             # weights sum to 1 -> brightness is kept
    return apply_mask(img, mask)


# ---------------------------------------------------------------
# STEP 2: median filter - chapter 3 slide 19
# Sort the n x n pixels around each pixel and take the MIDDLE value.
# Non-linear: an extreme value (0 or 1) ends up at the end of the sorted
# list, so it is simply ignored instead of being averaged in.
# ---------------------------------------------------------------
# SAY: "The median filter sorts the 9 neighbours and takes the middle one. A white dot of 1.0
#       goes to the END of the sorted list, so it is never chosen - it is ignored, not averaged.
#       This is a NON-LINEAR filter."
def median_filter_n(img, n):
    return median_filter(img, size=n, mode="nearest")


# ---------------------------------------------------------------
# STEP 3: spatial (adaptive) Wiener filter, same formula as MATLAB wiener2
#   local mean      mu(i,j)      = mean of the n x n window
#   local variance  s2(i,j)      = mean of g^2 in the window - mu^2
#   noise variance  v2           = average of all local variances (estimated)
#   output = mu + max(s2 - v2, 0) / max(s2, v2) * (g - mu)
# Flat area (s2 ~ v2):   the fraction is ~0  -> output ~ mu   (strong smoothing)
# Edge/detail (s2 >> v2): the fraction is ~1 -> output ~ g    (pixel kept)
# ---------------------------------------------------------------
# SAY: "The spatial Wiener filter, wiener2 in MATLAB, adapts to each window. It measures the
#       local mean and the local variance. Where the variance is small - a flat area - it
#       smooths strongly. Where the variance is large - an edge - it keeps the pixel."
def spatial_wiener(img, n, noise_var=None):
    mu = uniform_filter(img, size=n, mode="nearest")
    s2 = uniform_filter(img ** 2, size=n, mode="nearest") - mu ** 2
    if noise_var is None:
        noise_var = s2.mean()
    gain = np.maximum(s2 - noise_var, 0) / np.maximum(s2, noise_var)
    return mu + gain * (img - mu), noise_var


# ---------------------------------------------------------------
# STEP 4: quality measures - chapter 3 slide 24 (used again in Q7)
#   MSE  = 1/(M N) * sum (I - I_hat)^2          on the 8-bit scale 0..255
#   PSNR = 10 log10((2^n - 1)^2 / MSE), n = 8  -> (2^8 - 1)^2 = 255^2
# ---------------------------------------------------------------
# SAY: "To measure quality I use the lecture formulas. MSE is the average squared difference
#       from the original - a large MSE means low quality. PSNR is 10 log10 of 255 squared over
#       the MSE, in decibels - a HIGH PSNR means GOOD quality."
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
    # SAY (show q5_filters.png): "Here are the three filters at 3 by 3, with the zoom below."
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
    # SAY (show q5_sizes.png): "And here the same filters with 3, 5 and 7 by 7 windows."
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


# RESULT (cameraman, density 0.05) - SAY when comparing:
#   PSNR: noisy 17.8 dB | 3x3: average 24.9, MEDIAN 30.1, spatial Wiener 19.8 dB
#         5x5: 25.1 / 27.8 / 22.5 | 7x7: 24.3 / 26.2 / 23.9
#   SAY: "The MEDIAN 3x3 removes salt and pepper best - 30 dB, and every dot is gone - because
#         the dots are extreme values and the median never picks them."
#   SAY: "The AVERAGE filter does not remove the dots, it spreads them into grey blocks, and it
#         blurs every edge because it mixes both sides of the edge."
#   SAY: "The spatial WIENER filter is the worst here: a dot makes a huge local variance, so
#         the filter thinks it is an EDGE and keeps it. It is designed for Gaussian noise."
#   SAY: "Bigger windows: the median gets WORSE, 30 down to 26 dB - there is no extra noise to
#         remove, so it only erases detail like the hair. So reducing noise does NOT always
#         improve quality - every filter trades noise removal against detail."
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
