"""
Assignment 2 - Part B - Question 4: Image degradation
  1. Salt-and-pepper noise
  2. Motion blur + additive Gaussian noise

Model: g = h * f + n   (frequency domain: G = H F + N)

Run:  py q4_degradation.py
Q5 and Q6 import the degraded images from this file.
"""

import numpy as np
import matplotlib.pyplot as plt

from q1_edge_detection import IMAGE_PATH, OUT_DIR, load_gray

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
SP_DENSITY = 0.05        # share of pixels hit by salt and pepper
MOTION_LENGTH = 28       # blur length in pixels
MOTION_ANGLE = 15        # blur direction in degrees
GAUSS_SIGMA = 0.01       # Gaussian noise std
RNG_SEED = 0

ZOOM = (slice(60, 220), slice(170, 330))   # crop for the zoomed views


# ---------------------------------------------------------------
# STEP 1: salt and pepper: black with probability d/2, white with d/2
# ---------------------------------------------------------------
def make_salt_pepper(img, density=SP_DENSITY, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    r = rng.random(img.shape)
    noisy = img.copy()
    noisy[r < density / 2] = 0.0               # pepper
    noisy[(r >= density / 2) & (r < density)] = 1.0   # salt
    return noisy


# ---------------------------------------------------------------
# STEP 2: motion-blur kernel h: a line of length L, weights sum to 1
# ---------------------------------------------------------------
def motion_psf(length=MOTION_LENGTH, angle=MOTION_ANGLE):
    size = length if length % 2 == 1 else length + 1
    psf = np.zeros((size, size))
    c = size // 2
    theta = np.deg2rad(angle)
    for t in np.linspace(-(length - 1) / 2, (length - 1) / 2, 4 * length):
        x = c + t * np.cos(theta)
        y = c - t * np.sin(theta)
        psf[int(round(y)), int(round(x))] = 1.0
    return psf / psf.sum()


def psf_to_otf(psf, shape):
    """H(u,v): FFT of h, padded to the image size and centred at (0,0)."""
    padded = np.zeros(shape)
    ph, pw = psf.shape
    padded[:ph, :pw] = psf
    padded = np.roll(padded, (-(ph // 2), -(pw // 2)), axis=(0, 1))
    return np.fft.fft2(padded)


# ---------------------------------------------------------------
# STEP 3: blur in the frequency domain (G = H F), then add noise
# ---------------------------------------------------------------
def make_blur_noise(img, length=MOTION_LENGTH, angle=MOTION_ANGLE,
                    sigma=GAUSS_SIGMA, seed=RNG_SEED):
    """Returns (blurred + noisy, blurred only, H)."""
    H = psf_to_otf(motion_psf(length, angle), img.shape)
    blurred = np.real(np.fft.ifft2(H * np.fft.fft2(img)))
    rng = np.random.default_rng(seed)
    noisy = blurred + rng.normal(0.0, sigma, img.shape)
    return noisy, blurred, H


# ---------------------------------------------------------------
# STEP 4: display and save
# ---------------------------------------------------------------
def show(img, sp, blur_only, blur_noise, psf):
    shown = [("Original", img),
             (f"Salt & pepper (density {SP_DENSITY})", sp),
             (f"Motion blur L={MOTION_LENGTH}, {MOTION_ANGLE} deg\n+ Gaussian noise sigma={GAUSS_SIGMA}",
              blur_noise)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    for col, (title, im) in enumerate(shown):
        axes[0, col].imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(title, fontsize=10)
        axes[1, col].imshow(np.clip(im[ZOOM], 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title("Zoom: " + title.split("\n")[0], fontsize=10)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Original and degraded images (bottom row: zoomed detail)", fontsize=13)
    fig.savefig(OUT_DIR / "q4_degraded.png", dpi=150, bbox_inches="tight")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    axes[0].imshow(psf, cmap="gray")
    axes[0].set_title(f"Motion blur kernel h(x,y)\n{psf.shape[0]}x{psf.shape[1]}, length {MOTION_LENGTH}, "
                      f"angle {MOTION_ANGLE} deg")
    axes[1].imshow(blur_only[ZOOM], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Zoom: blur only (h * f)")
    axes[2].imshow(np.clip(blur_noise[ZOOM], 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Zoom: blur + Gaussian noise (h * f + n), sigma={GAUSS_SIGMA}")
    for ax in axes:
        ax.axis("off")
    fig.savefig(OUT_DIR / "q4_blur_kernel.png", dpi=150, bbox_inches="tight")


def show_parameter_effect(img):
    """The degradations with other parameter values."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    for col, d in enumerate([0.01, 0.05, 0.20]):
        axes[0, col].imshow(make_salt_pepper(img, d)[ZOOM], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"Salt & pepper, density = {d}")
    for col, (length, sigma) in enumerate([(10, 0.01), (MOTION_LENGTH, 0.01), (MOTION_LENGTH, 0.05)]):
        g, _, _ = make_blur_noise(img, length, MOTION_ANGLE, sigma)
        axes[1, col].imshow(np.clip(g[ZOOM], 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"Motion L = {length}, Gaussian sigma = {sigma}")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Effect of the degradation parameters (zoomed)", fontsize=13)
    fig.savefig(OUT_DIR / "q4_parameters.png", dpi=150, bbox_inches="tight")


def mse(a, b):
    """MSE on the 0..255 scale."""
    return np.mean((255 * (a - b)) ** 2)


if __name__ == "__main__":
    image = load_gray(IMAGE_PATH)
    sp = make_salt_pepper(image)
    blur_noise, blur_only, _ = make_blur_noise(image)
    psf = motion_psf()

    print(f"Salt & pepper: {100 * np.mean(sp != image):.1f}% of pixels changed "
          f"(density {SP_DENSITY}), MSE = {mse(image, sp):.1f}")
    print(f"Motion blur only:          MSE = {mse(image, blur_only):.1f}")
    print(f"Motion blur + noise:       MSE = {mse(image, blur_noise):.1f}  "
          f"(noise std {GAUSS_SIGMA}, PSF size {psf.shape})")

    show(image, sp, blur_only, blur_noise, psf)
    show_parameter_effect(image)
    print(f"Figures saved in: {OUT_DIR}")
    plt.show()
