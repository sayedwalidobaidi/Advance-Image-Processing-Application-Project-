import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from q1_edge_detection import (IMAGE_PATH, OUT_DIR, load_gray, apply_mask, normalise,
                               gradient_magnitude, SOBEL_X, SOBEL_Y, PREWITT_X, PREWITT_Y)

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
EDGE_THRESHOLD = 0.25    # edge threshold (as in Q1)
ZC_THRESHOLD = 0.20      # minimum jump at a zero crossing
NOISE_SIGMA = 0.05       # noise for the noise test
SMOOTH_SIGMA = 1.0       # Gaussian smoothing before the Laplacian
RNG_SEED = 0

# ---------------------------------------------------------------
# STEP 1: the Laplacian masks (second derivative, each mask sums to 0)
# ---------------------------------------------------------------
LAPLACIAN_MASKS = {
    "4-neighbour": np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]]),
    "4-neighbour (negative)": np.array([[0, -1, 0],
                                        [-1, 4, -1],
                                        [0, -1, 0]]),
    "8-neighbour": np.array([[1, 1, 1],
                             [1, -8, 1],
                             [1, 1, 1]]),
    "Diagonal-weighted": np.array([[1, -2, 1],
                                   [-2, 4, -2],
                                   [1, -2, 1]]),
}


def laplacian(img, mask):
    return apply_mask(img, mask)


# ---------------------------------------------------------------
# STEP 2: edges = zero crossings (sign change) of the Laplacian
# ---------------------------------------------------------------
def zero_crossings(lap, min_jump):
    """Sign change to the right or lower neighbour, with a large enough jump."""
    jump = min_jump * np.abs(lap).max()
    edges = np.zeros(lap.shape, dtype=bool)
    right = (np.sign(lap[:, :-1]) != np.sign(lap[:, 1:])) & (np.abs(lap[:, :-1] - lap[:, 1:]) > jump)
    down = (np.sign(lap[:-1, :]) != np.sign(lap[1:, :])) & (np.abs(lap[:-1, :] - lap[1:, :]) > jump)
    edges[:, :-1] |= right
    edges[:-1, :] |= down
    return edges


def first_order(img):
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
        lim = np.abs(lap).max()
        axes[0, col].imshow(lap, cmap="gray", vmin=-lim, vmax=lim)
        axes[0, col].set_title(f"{name}\nraw output (grey = 0)", fontsize=10)
        axes[1, col].imshow(normalise(lap) > EDGE_THRESHOLD, cmap="gray")
        axes[1, col].set_title(f"|Laplacian| > {EDGE_THRESHOLD}\n"
                               f"edge pixels: {edge_share(lap, EDGE_THRESHOLD):.1f}%", fontsize=10)
        zc = zero_crossings(lap, ZC_THRESHOLD)
        axes[2, col].imshow(zc, cmap="gray")
        axes[2, col].set_title(f"Zero crossings (jump > {ZC_THRESHOLD})\n"
                               f"edge pixels: {100 * zc.mean():.1f}%", fontsize=10)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Laplacian masks", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "q2_laplacian_masks.png", dpi=150, bbox_inches="tight")


def show_vs_first_order(img, laps):
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
    """Count edge pixels before and after adding Gaussian noise."""
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

    # same threshold (from the clean image) for clean and noisy
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
