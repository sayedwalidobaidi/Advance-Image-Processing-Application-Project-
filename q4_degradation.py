"""
Assignment 2 - Part B - Question 4: Image degradation
  1. Salt-and-pepper noise
  2. Motion blur + additive Gaussian noise

Degradation model (lecture chapter 5):   g(x,y) = h(x,y) * f(x,y) + n(x,y)
In the frequency domain:                 G(u,v) = H(u,v) F(u,v) + N(u,v)

Run:  py q4_degradation.py
Q5 and Q6 import make_salt_pepper() and make_blur_noise() from this file,
so all questions use exactly the same degraded images.
"""

import numpy as np
import matplotlib.pyplot as plt

from q1_edge_detection import IMAGE_PATH, OUT_DIR, load_gray

# ---------------------------------------------------------------
# Settings (change these and re-run to see the effect)
# ---------------------------------------------------------------
SP_DENSITY = 0.05        # fraction of pixels hit by salt-and-pepper noise (lecture code uses 0.01)
MOTION_LENGTH = 28       # same as the lecture: h = fspecial('motion', 28, 15)
MOTION_ANGLE = 15        # direction of the motion in degrees (0 = horizontal)
GAUSS_SIGMA = 0.01       # standard deviation of the Gaussian noise (image range is 0..1)
RNG_SEED = 0             # fixed seed: the same "random" noise every run

ZOOM = (slice(60, 220), slice(170, 330))   # crop (rows, cols) used to show fine detail


# ---------------------------------------------------------------
# STEP 1: salt-and-pepper noise (impulse noise, chapter 3 slide 6)
# Each pixel is replaced by pure black (pepper, 0) with probability d/2,
# by pure white (salt, 1) with probability d/2, and is left alone otherwise.
# ---------------------------------------------------------------
def make_salt_pepper(img, density=SP_DENSITY, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    r = rng.random(img.shape)                  # one random number 0..1 per pixel
    noisy = img.copy()
    noisy[r < density / 2] = 0.0               # pepper
    noisy[(r >= density / 2) & (r < density)] = 1.0   # salt
    return noisy


# ---------------------------------------------------------------
# STEP 2: the motion-blur degradation function h(x,y) (the PSF)
# A camera moving L pixels in a straight line: every point of the scene is
# smeared into a short line. h is that line, with all weights adding up to 1
# (so the average brightness of the image does not change).
# ---------------------------------------------------------------
def motion_psf(length=MOTION_LENGTH, angle=MOTION_ANGLE):
    size = length if length % 2 == 1 else length + 1      # odd size -> has a centre pixel
    psf = np.zeros((size, size))
    c = size // 2
    theta = np.deg2rad(angle)
    for t in np.linspace(-(length - 1) / 2, (length - 1) / 2, 4 * length):
        x = c + t * np.cos(theta)
        y = c - t * np.sin(theta)                 # minus: image rows grow downwards
        psf[int(round(y)), int(round(x))] = 1.0
    return psf / psf.sum()


def psf_to_otf(psf, shape):
    """H(u,v): Fourier transform of h, padded to the image size and shifted so
    that the centre of the PSF sits at pixel (0,0) (otherwise the image would move)."""
    padded = np.zeros(shape)
    ph, pw = psf.shape
    padded[:ph, :pw] = psf
    padded = np.roll(padded, (-(ph // 2), -(pw // 2)), axis=(0, 1))
    return np.fft.fft2(padded)


# ---------------------------------------------------------------
# STEP 3: blur (G = H F) and add Gaussian noise (+ N)
# Blurring by multiplying in the frequency domain is the same as convolving
# with h in the spatial domain (lecture: convolution <-> multiplication).
# ---------------------------------------------------------------
def make_blur_noise(img, length=MOTION_LENGTH, angle=MOTION_ANGLE,
                    sigma=GAUSS_SIGMA, seed=RNG_SEED):
    """Returns (blurred + noisy image, blurred image without noise, H)."""
    H = psf_to_otf(motion_psf(length, angle), img.shape)
    blurred = np.real(np.fft.ifft2(H * np.fft.fft2(img)))
    rng = np.random.default_rng(seed)
    noisy = blurred + rng.normal(0.0, sigma, img.shape)   # mean 0, std sigma
    return noisy, blurred, H
# WHY the dark strip at the image border: multiplying in the frequency domain is a CIRCULAR
#     convolution, so the blur wraps around from one side of the image to the other.


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

    # The blur kernel itself, and blur vs blur+noise, to separate the two effects
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
    """Same image with weaker and stronger settings, for the 'which parameters
    control the amount of noise and blur' question."""
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
    """Chapter 3 slide 24, on the 8-bit scale 0..255."""
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
