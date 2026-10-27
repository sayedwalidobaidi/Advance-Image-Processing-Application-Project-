"""
Assignment 2 - Part B - Question 6: Inverse and Wiener restoration (frequency domain)
Input: the motion blur + Gaussian noise image from Q4, with its known H(u,v).

  Model:          G = H F + N
  Inverse filter: F_hat = G / H                       (only where |H| > threshold, as in the lecture)
  Wiener filter:  F_hat = G * conj(H) / (|H|^2 + K)   (lecture code: K = 0.05)

Run:  py q6_inverse_wiener.py
Figures are shown on screen and saved in the "results" folder next to this file.

VIDEO SCRIPT (about 60 seconds) - read the "# SAY:" comments from top to bottom.
  1. Inverse filter: divide by H; why the noise explodes (show the |H| spectrum).
  2. The lecture fix |H| > T.
  3. Wiener filter: the +K term; K = 0 gives the inverse filter again.
  4. Compare the four images; change T and K live; limitations.
"""

import numpy as np
import matplotlib.pyplot as plt

from q1_edge_detection import IMAGE_PATH, OUT_DIR, load_gray
from q4_degradation import make_blur_noise, GAUSS_SIGMA, MOTION_LENGTH, MOTION_ANGLE, ZOOM
from q5_spatial_filters import mse, psnr

# ---------------------------------------------------------------
# Settings (change these and re-run to see the effect)
# ---------------------------------------------------------------
INVERSE_T = 0.1                         # inverse filter: only use frequencies where |H| > T (best PSNR here)
INVERSE_T_TRIALS = [0.001, 0.1, 0.45]   # 0.45 is the value used in the lecture code
WIENER_K = 0.01                         # Wiener: K ~ noise power / signal power (best PSNR here)
WIENER_K_TRIALS = [0.0001, 0.01, 0.05]  # 0.05 is the value used in the lecture code


# ---------------------------------------------------------------
# STEP 1: inverse filtering
# G = H F + N   ->   G / H = F + N / H
# Where |H| is tiny, N / H becomes huge, so the noise explodes.
# The lecture's fix: keep only the frequencies where |H| > T, set the others to 0.
# ---------------------------------------------------------------
# SAY: "The blur multiplied every frequency by H, so the simplest restoration is to DIVIDE by
#       H. But G over H equals F PLUS N over H. Where H is almost zero, N over H becomes huge,
#       so the noise explodes. The lecture fix is to use only frequencies where |H| is larger
#       than a threshold T - in MATLAB, abs(H) > 0.45 - and set the others to zero."
def inverse_filter(g, H, threshold):
    G = np.fft.fft2(g)
    keep = np.abs(H) > threshold
    F_hat = np.zeros_like(G)
    F_hat[keep] = G[keep] / H[keep]
    return np.real(np.fft.ifft2(F_hat)), keep.mean()


# ---------------------------------------------------------------
# STEP 2: Wiener filtering (lecture chapter 5, with Sn/Sf replaced by a constant K)
#   F_hat = [ |H|^2 / (|H|^2 + K) ] * G / H  =  G * conj(H) / (|H|^2 + K)
# Where |H| is large  -> |H|^2 >> K -> behaves like the inverse filter (undo the blur)
# Where |H| is small  -> |H|^2 << K -> the gain goes to 0 (do not amplify noise)
# K = 0 gives back the plain inverse filter.
# ---------------------------------------------------------------
# SAY: "The Wiener filter minimises the mean square error. It is G times |H| squared over
#       |H| squared plus K, divided by H. Where H is large, K does not matter and it removes
#       the blur like the inverse filter. Where H is small, K wins and the gain goes to zero,
#       so the noise is NOT amplified. With K = 0 it becomes the inverse filter again."
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
    # SAY (show q6_restoration.png): "Original, degraded, inverse and Wiener side by side."
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
    """Why inverse filtering fails: |H| has many near-zero values (dark lines)."""
    # SAY (show q6_spectrum.png): "This is |H|. The dark lines are frequencies where H is almost
    #       zero - there the inverse filter divides by almost zero."
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
    # SAY (show q6_parameters.png): "The first panel is the proof: WITHOUT noise the inverse
    #       filter restores the image almost perfectly. So the problem is the noise, not the blur."
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), layout="constrained")
    # Row 1: inverse filter. First panel: no noise, to prove the noise is the problem.
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


# RESULT (cameraman, lecture blur L = 28 at 15 deg, sigma = 0.01) - SAY when comparing:
#   Degraded 21.4 dB. Inverse with ALL frequencies: 4.8 dB, the image is destroyed by noise.
#   Inverse on the blur WITHOUT noise: 76 dB, almost perfect -> the noise is the real problem.
#   Inverse T: 0.001 -> ~6.9 dB (noise explodes); 0.1 -> 22.9 dB (best); 0.45 (lecture) ->
#     21.6 dB - it throws away most frequencies, so the image stays blurred.
#   Wiener K: 0.0001 -> very grainy (almost the inverse filter); 0.01 -> 23.8 dB (best);
#     0.05 (lecture) -> 22.5 dB, smoother but more blur.
#   SAY: "The Wiener filter restores the blurred image best: 23.8 dB against 22.9 for the
#         inverse filter, and it has less ringing - the inverse filter's hard on/off cut in
#         frequency makes ripples around the head."
#   SAY: "Limitations of both: they need H exactly - the blur length and angle; they cannot
#         recover frequencies where H is zero; ringing remains; Wiener also needs K, which I
#         found by trying values against the original - impossible with a real photo."
# LIVE DEMO: set GAUSS_SIGMA = 0 in q4_degradation.py -> both filters become almost perfect.
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
