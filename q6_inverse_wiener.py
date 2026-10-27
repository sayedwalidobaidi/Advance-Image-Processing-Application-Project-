"""
Assignment 2 - Part B - Question 6: Inverse and Wiener restoration (frequency domain)
Input: the motion blur + Gaussian noise image from Q4, with its known H(u,v).

Run:  py q6_inverse_wiener.py
Figures are shown on screen and saved in the "results" folder next to this file.
"""

import numpy as np
import matplotlib.pyplot as plt

from q1_edge_detection import IMAGE_PATH, OUT_DIR, load_gray
from q4_degradation import make_blur_noise, GAUSS_SIGMA, MOTION_LENGTH, MOTION_ANGLE, ZOOM
from q5_spatial_filters import mse, psnr

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
INVERSE_T = 0.1                         # inverse: use only |H| > T
INVERSE_T_TRIALS = [0.001, 0.1, 0.45]
WIENER_K = 0.01                         # Wiener: K ~ noise / signal power
WIENER_K_TRIALS = [0.0001, 0.01, 0.05]


# ---------------------------------------------------------------
# STEP 1: inverse filter, F = G / H where |H| > T (G / H = F + N / H)
# ---------------------------------------------------------------
def inverse_filter(g, H, threshold):
    G = np.fft.fft2(g)
    keep = np.abs(H) > threshold
    F_hat = np.zeros_like(G)
    F_hat[keep] = G[keep] / H[keep]
    return np.real(np.fft.ifft2(F_hat)), keep.mean()


# ---------------------------------------------------------------
# STEP 2: Wiener filter, F = G * conj(H) / (|H|^2 + K)
# ---------------------------------------------------------------
def wiener_filter(g, H, k):
    G = np.fft.fft2(g)
    F_hat = G * np.conj(H) / (np.abs(H) ** 2 + k)
    return np.real(np.fft.ifft2(F_hat))


# ---------------------------------------------------------------
# STEP 3: display and save
# ---------------------------------------------------------------
def show_im(ax, im, title, zoom=False):
    im = np.clip(im[ZOOM] if zoom else im, 0, 1)
    ax.imshow(im, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def show_main(img, g, inv, wie):
    panels = [("Original", img), (f"Degraded (blur + noise)\nPSNR {psnr(img, g):.1f} dB", g),
              (f"Inverse, |H| > {INVERSE_T}\nPSNR {psnr(img, inv):.1f} dB", inv),
              (f"Wiener, K = {WIENER_K}\nPSNR {psnr(img, wie):.1f} dB", wie)]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), layout="constrained")
    for col, (title, im) in enumerate(panels):
        show_im(axes[0, col], im, title)
        show_im(axes[1, col], im, "Zoom: " + title.split("\n")[0], zoom=True)
    fig.suptitle("Frequency-domain restoration of the motion-blurred + noisy image", fontsize=13)
    fig.savefig(OUT_DIR / "q6_restoration.png", dpi=150, bbox_inches="tight")


def show_spectrum(H, g):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    axes[0].imshow(np.log10(np.fft.fftshift(np.abs(H)) + 1e-6), cmap="gray")
    axes[0].set_title("log10 |H(u,v)|  (motion blur, centre = low frequencies)\n"
                      "dark lines = H close to 0 -> G/H divides by ~0")
    axes[1].imshow(np.log10(np.fft.fftshift(np.abs(np.fft.fft2(g))) + 1e-6), cmap="gray")
    axes[1].set_title("log10 |G(u,v)|  (degraded image spectrum)")
    for ax in axes:
        ax.axis("off")
    fig.savefig(OUT_DIR / "q6_spectrum.png", dpi=150, bbox_inches="tight")


def show_parameters(img, g, blur_only, H):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), layout="constrained")
    # Row 1: inverse filter (first panel without noise)
    clean_inv, _ = inverse_filter(blur_only, H, 1e-6)
    show_im(axes[0, 0], clean_inv,
            f"Inverse on blur ONLY (no noise)\nPSNR {psnr(img, clean_inv):.1f} dB", zoom=True)
    for col, t in enumerate(INVERSE_T_TRIALS, start=1):
        r, kept = inverse_filter(g, H, t)
        show_im(axes[0, col], r, f"Inverse, |H| > {t}  ({100 * kept:.0f}% of freqs kept)\n"
                                 f"PSNR {psnr(img, r):.1f} dB", zoom=True)
    # Row 2: Wiener filter
    show_im(axes[1, 0], g, f"Degraded\nPSNR {psnr(img, g):.1f} dB", zoom=True)
    for col, k in enumerate(WIENER_K_TRIALS, start=1):
        r = wiener_filter(g, H, k)
        show_im(axes[1, col], r, f"Wiener, K = {k}\nPSNR {psnr(img, r):.1f} dB", zoom=True)
    fig.suptitle("Effect of the main parameter: inverse threshold T (top), Wiener K (bottom) - zoomed",
                 fontsize=13)
    fig.savefig(OUT_DIR / "q6_parameters.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    image = load_gray(IMAGE_PATH)
    g, blur_only, H = make_blur_noise(image)
    print(f"Degradation: motion L={MOTION_LENGTH}, angle={MOTION_ANGLE}, Gaussian sigma={GAUSS_SIGMA}")
    print(f"|H| range: min = {np.abs(H).min():.2e}, max = {np.abs(H).max():.2f}, "
          f"{100 * np.mean(np.abs(H) < 0.01):.0f}% of frequencies have |H| < 0.01")

    print(f"\n{'Method':30s} {'MSE':>8s} {'PSNR (dB)':>10s}")
    print(f"{'Degraded (no filter)':30s} {mse(image, g):8.1f} {psnr(image, g):10.2f}")
    for t in [1e-6] + INVERSE_T_TRIALS:
        r, kept = inverse_filter(g, H, t)
        print(f"{f'Inverse |H| > {t:g} ({100 * kept:.0f}% kept)':30s} {mse(image, r):8.1f} {psnr(image, r):10.2f}")
    for k in WIENER_K_TRIALS:
        r = wiener_filter(g, H, k)
        print(f"{f'Wiener K = {k:g}':30s} {mse(image, r):8.1f} {psnr(image, r):10.2f}")

    inv, _ = inverse_filter(g, H, INVERSE_T)
    wie = wiener_filter(g, H, WIENER_K)
    show_main(image, g, inv, wie)
    show_spectrum(H, g)
    show_parameters(image, g, blur_only, H)
    print(f"\nFigures saved in: {OUT_DIR}")
    plt.show()
