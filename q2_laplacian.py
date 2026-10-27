"""
Assignment 2 - Part A - Question 2: Laplacian edge detection
Masks from the lecture (chapter 4, slides 9-10), compared with Sobel and Prewitt.

Run:  py q2_laplacian.py
Figures are shown on screen and saved in the "results" folder next to this file.

VIDEO SCRIPT (about 45 seconds) - read the "# SAY:" comments from top to bottom.
  1. The Laplacian is the SECOND derivative; how the mask is built from V.13-V.16.
  2. The masks, and the sign error on slide 10.
  3. Zero crossings = where the edge is.
  4. Compare with Sobel/Prewitt and show the noise test (1.1x vs 4.2x).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Reuse the image loader, mask filter and Sobel/Prewitt masks from Question 1
from q1_edge_detection import (IMAGE_PATH, OUT_DIR, load_gray, apply_mask, normalise,
                               gradient_magnitude, SOBEL_X, SOBEL_Y, PREWITT_X, PREWITT_Y)

# SAY: "I reuse the image and the Sobel and Prewitt masks from question 1, so both
#       questions use exactly the same image."

# ---------------------------------------------------------------
# Settings (change these and re-run to see the effect)
# ---------------------------------------------------------------
EDGE_THRESHOLD = 0.25    # 0..1, same idea as Q1: keep only the strong responses
ZC_THRESHOLD = 0.20      # 0..1, minimum jump across a zero crossing to count as an edge
NOISE_SIGMA = 0.05       # std. dev. of the Gaussian noise used in the noise-sensitivity test
SMOOTH_SIGMA = 1.0       # Gaussian smoothing before the Laplacian (study notes: "smooth first")
RNG_SEED = 0             # fixed seed so the noise is the same every run

# ---------------------------------------------------------------
# STEP 1: the Laplacian masks
# Second derivative in x:  f(x+1,y) + f(x-1,y) - 2 f(x,y)        (lecture V.13)
# Second derivative in y:  f(x,y+1) + f(x,y-1) - 2 f(x,y)        (lecture V.15)
# Add them:                sum of 4 neighbours - 4 f(x,y)        (lecture V.16)
# Every Laplacian mask sums to 0, so a flat region gives 0.
# ---------------------------------------------------------------
# SAY: "The Laplacian is the SECOND derivative. In x it is f(x+1) plus f(x-1) minus
#       2 f(x); the same in y. Adding them gives the four neighbours minus four times the
#       centre - that is exactly the first mask, equation V.16."
# SAY: "Every Laplacian mask must add up to zero, so a flat area gives zero. The 8-neighbour
#       mask on slide 10 is printed with all minus ones and minus 8, which adds up to minus 16,
#       so it would just give a dark copy of the image. I corrected the signs."
LAPLACIAN_MASKS = {
    "4-neighbour": np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]]),        # slide 10, top mask (eq. V.16)
    "4-neighbour (negative)": np.array([[0, -1, 0],
                                        [-1, 4, -1],
                                        [0, -1, 0]]),   # same mask, sign reversed
    "8-neighbour": np.array([[1, 1, 1],
                             [1, -8, 1],
                             [1, 1, 1]]),        # slide 10 middle mask, signs corrected so it sums to 0
    "Diagonal-weighted": np.array([[1, -2, 1],
                                   [-2, 4, -2],
                                   [1, -2, 1]]),      # slide 10, right mask
}
# SAY: "The negative mask gives the same edges with black and white swapped. The 8-neighbour
#       mask also uses the diagonals, so it reacts to every direction. The alternating mask
#       reacts mostly to fine texture."


def laplacian(img, mask):
    return apply_mask(img, mask)


# ---------------------------------------------------------------
# STEP 2: turn the Laplacian into an edge image
# The Laplacian is positive on one side of an edge and negative on the
# other, so the edge itself is where the sign changes = ZERO CROSSING.
# ---------------------------------------------------------------
# SAY: "With the first derivative we look for the MAXIMUM. With the second derivative the
#       value is positive on one side of the edge and negative on the other, so the edge is
#       where it crosses ZERO. I mark a pixel when the sign changes and the jump is big enough."
def zero_crossings(lap, min_jump):
    """Mark a pixel as an edge if the Laplacian changes sign between it and its
    right or lower neighbour, and the jump is large enough (to ignore tiny noise)."""
    jump = min_jump * np.abs(lap).max()
    edges = np.zeros(lap.shape, dtype=bool)
    right = (np.sign(lap[:, :-1]) != np.sign(lap[:, 1:])) & (np.abs(lap[:, :-1] - lap[:, 1:]) > jump)
    down = (np.sign(lap[:-1, :]) != np.sign(lap[1:, :])) & (np.abs(lap[:-1, :] - lap[1:, :]) > jump)
    edges[:, :-1] |= right
    edges[:-1, :] |= down
    return edges


def first_order(img):
    """Sobel and Prewitt magnitudes from Q1, used for the comparison."""
    return {"Sobel": gradient_magnitude(apply_mask(img, SOBEL_X), apply_mask(img, SOBEL_Y)),
            "Prewitt": gradient_magnitude(apply_mask(img, PREWITT_X), apply_mask(img, PREWITT_Y))}


def edge_share(strength, threshold):
    return 100 * (normalise(strength) > threshold).mean()


# ---------------------------------------------------------------
# STEP 3: display and save
# ---------------------------------------------------------------
def show_laplacians(img, laps):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for col, (name, lap) in enumerate(laps.items()):
        # Row 1: raw Laplacian (grey = 0, white = positive, black = negative)
        lim = np.abs(lap).max()
        axes[0, col].imshow(lap, cmap="gray", vmin=-lim, vmax=lim)
        axes[0, col].set_title(f"{name}\nraw output (grey = 0)", fontsize=10)
        # Row 2: |Laplacian|, thresholded like Q1
        axes[1, col].imshow(normalise(lap) > EDGE_THRESHOLD, cmap="gray")
        axes[1, col].set_title(f"|Laplacian| > {EDGE_THRESHOLD}\n"
                               f"edge pixels: {edge_share(lap, EDGE_THRESHOLD):.1f}%", fontsize=10)
        # Row 3: zero crossings
        zc = zero_crossings(lap, ZC_THRESHOLD)
        axes[2, col].imshow(zc, cmap="gray")
        axes[2, col].set_title(f"Zero crossings (jump > {ZC_THRESHOLD})\n"
                               f"edge pixels: {100 * zc.mean():.1f}%", fontsize=10)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Laplacian masks from the lecture", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "q2_laplacian_masks.png", dpi=150, bbox_inches="tight")


def show_vs_first_order(img, laps):
    # SAY (show q2_vs_sobel_prewitt.png): "Sobel and Prewitt give thick, solid outlines.
    #       The Laplacian gives thinner but broken edges, and already reacts to the grass."
    first = first_order(img)
    panels = [("Original", img, None),
              ("Sobel (1st order)", first["Sobel"], EDGE_THRESHOLD),
              ("Prewitt (1st order)", first["Prewitt"], EDGE_THRESHOLD),
              ("Laplacian 4-nb (2nd order)", laps["4-neighbour"], EDGE_THRESHOLD),
              ("Laplacian 8-nb (2nd order)", laps["8-neighbour"], EDGE_THRESHOLD)]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8))
    for ax, (title, im, t) in zip(axes, panels):
        if t is None:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(normalise(im) > t, cmap="gray")
            title += f"\nedge pixels: {edge_share(im, t):.1f}%"
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "q2_vs_sobel_prewitt.png", dpi=150, bbox_inches="tight")


def noise_test(img):
    """Add a little Gaussian noise and count how many edge pixels each operator
    finds before and after. More new edge pixels = more sensitive to noise."""
    # SAY: "To test noise sensitivity I add Gaussian noise and count the edge pixels before and
    #       after, with the SAME threshold. More new edge pixels means more sensitive to noise.
    #       I also test the study-notes advice: smooth with a Gaussian first, then Laplacian."
    rng = np.random.default_rng(RNG_SEED)
    noisy = np.clip(img + rng.normal(0, NOISE_SIGMA, img.shape), 0, 1)

    def strengths(x):
        f = first_order(x)
        return {"Sobel": f["Sobel"], "Prewitt": f["Prewitt"],
                "Laplacian 4-nb": laplacian(x, LAPLACIAN_MASKS["4-neighbour"]),
                "Laplacian 8-nb": laplacian(x, LAPLACIAN_MASKS["8-neighbour"]),
                f"Gauss({SMOOTH_SIGMA}) + Lap 4-nb": laplacian(gaussian_filter(x, SMOOTH_SIGMA),
                                                              LAPLACIAN_MASKS["4-neighbour"])}

    clean, dirty = strengths(img), strengths(noisy)

    # Use the SAME absolute threshold for clean and noisy (taken from the clean image),
    # otherwise the noise raises the maximum and hides its own effect.
    limits = {name: EDGE_THRESHOLD * np.abs(s).max() for name, s in clean.items()}

    print(f"\nNoise test (Gaussian noise, sigma = {NOISE_SIGMA}), threshold = {EDGE_THRESHOLD} x clean max")
    print(f"{'Operator':22s} {'clean %':>8s} {'noisy %':>8s} {'increase':>9s}")
    for name in clean:
        a = 100 * (np.abs(clean[name]) > limits[name]).mean()
        b = 100 * (np.abs(dirty[name]) > limits[name]).mean()
        print(f"{name:22s} {a:8.1f} {b:8.1f} {b / a:8.1f}x")

    fig, axes = plt.subplots(1, 6, figsize=(24, 4.8))
    axes[0].imshow(noisy, cmap="gray")
    axes[0].set_title(f"Noisy image (sigma = {NOISE_SIGMA})", fontsize=10)
    for ax, (name, s) in zip(axes[1:], dirty.items()):
        edges = np.abs(s) > limits[name]
        ax.imshow(edges, cmap="gray")
        ax.set_title(f"{name} on noisy image\nedge pixels: {100 * edges.mean():.1f}%", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "q2_noise_sensitivity.png", dpi=150, bbox_inches="tight")


# RESULT (cameraman) - SAY when comparing:
#   Clean image, T = 0.25: Laplacian 4-nb 2.4 %, 8-nb 2.9 %, alternating 4.0 % (mostly texture);
#   Sobel 4.2 %, Prewitt 4.5 % -> the Laplacian gives thinner, more broken edges.
#   Noise test: Sobel and Prewitt grow only ~1.1x, Laplacian 4-nb grows ~4.2x and 8-nb ~2.7x.
#   SAY: "First order measures the SLOPE of brightness, so an edge is a peak. Second order
#         measures the CHANGE of the slope, so an edge is a zero crossing. Differentiating
#         twice amplifies noise twice - that is why the Laplacian is much more sensitive."
#   Gaussian (sigma = 1) THEN Laplacian: 3.2 % -> 3.5 %, only ~1.1x, as stable as Sobel.
#   SAY: "When I smooth with a Gaussian first, as the study notes say, the Laplacian grows
#         only 1.1 times instead of 4.2 times - the smoothing removes the noise before the
#         second derivative can amplify it. This is the idea of the Laplacian of Gaussian."
# LIVE DEMO: set ZC_THRESHOLD = 0.05 -> the zero crossings fill with noise; set it back to 0.20.
if __name__ == "__main__":
    image = load_gray(IMAGE_PATH)

    laps = {name: laplacian(image, m) for name, m in LAPLACIAN_MASKS.items()}
    for name, lap in laps.items():
        print(f"{name:24s} min = {lap.min():6.2f}  max = {lap.max():6.2f}  "
              f"edge pixels at T={EDGE_THRESHOLD}: {edge_share(lap, EDGE_THRESHOLD):4.1f}%")

    show_laplacians(image, laps)
    show_vs_first_order(image, laps)
    noise_test(image)
    print(f"Figures saved in: {OUT_DIR}")
    plt.show()
