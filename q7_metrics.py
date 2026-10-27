"""
Assignment 2 - Part C - Question 7: Quantitative and visual comparison
MSE and PSNR of every restoration method from Q5 and Q6, with the original as reference.

  MSE  = (1 / (M N)) * sum (I(i,j) - I_hat(i,j))^2        chapter 3 slide 24, 0..255 scale
  PSNR = 10 * log10((2^n - 1)^2 / MSE), n = 8 bits          higher is better (dB)

Run:  py q7_metrics.py
Saves the table as results/q7_table.csv (opens in Excel/Word) and a figure for the report.

VIDEO SCRIPT (about 35 seconds) - read the "# SAY:" comments from top to bottom.
  1. What the table contains and the two formulas.
  2. Best method for each degradation.
  3. Do the numbers agree with what we see?
  4. Why a filter good for one noise fails on the other.
"""

import csv

import numpy as np
import matplotlib.pyplot as plt

from q1_edge_detection import IMAGE_PATH, OUT_DIR, load_gray
from q4_degradation import make_salt_pepper, make_blur_noise, ZOOM
from q5_spatial_filters import (average_filter, median_filter_n, spatial_wiener,
                                mse, psnr, MAIN_SIZE)
from q6_inverse_wiener import inverse_filter, wiener_filter, INVERSE_T, WIENER_K

# SAY: "Question 7 puts all restoration results in one table. Every method is run again with
#       the parameters from questions 5 and 6, and compared with the ORIGINAL image using
#       MSE - lower is better - and PSNR in decibels - higher is better."


# ---------------------------------------------------------------
# STEP 1: run every method with the parameters chosen in Q5 and Q6
# ---------------------------------------------------------------
def run_all(img):
    n = MAIN_SIZE
    sp = make_salt_pepper(img)
    g, _, H = make_blur_noise(img)
    rows = [
        ("Salt-and-pepper noise", "None (degraded image)", "-", sp),
        ("Salt-and-pepper noise", "Average", f"{n}x{n}", average_filter(sp, n)),
        ("Salt-and-pepper noise", "Median", f"{n}x{n}", median_filter_n(sp, n)),
        ("Salt-and-pepper noise", "Spatial Wiener", f"{n}x{n}", spatial_wiener(sp, n)[0]),
        ("Motion blur + Gaussian noise", "None (degraded image)", "-", g),
        ("Motion blur + Gaussian noise", "Inverse", f"|H| > {INVERSE_T}", inverse_filter(g, H, INVERSE_T)[0]),
        ("Motion blur + Gaussian noise", "Frequency-domain Wiener", f"K = {WIENER_K}", wiener_filter(g, H, WIENER_K)),
    ]
    return [(deg, method, param, im, mse(img, im), psnr(img, im)) for deg, method, param, im in rows]


# ---------------------------------------------------------------
# STEP 2: print and save the table
# ---------------------------------------------------------------
def save_table(rows):
    print(f"{'Degraded image':30s} {'Restoration method':26s} {'Parameter':12s} {'MSE':>8s} {'PSNR (dB)':>10s}")
    for deg, method, param, _, e, p in rows:
        print(f"{deg:30s} {method:26s} {param:12s} {e:8.1f} {p:10.2f}")
    with open(OUT_DIR / "q7_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Degraded image", "Restoration method", "Parameter", "MSE", "PSNR (dB)"])
        for deg, method, param, _, e, p in rows:
            w.writerow([deg, method, param, f"{e:.1f}", f"{p:.2f}"])


def best_per_degradation(rows):
    for deg in dict.fromkeys(r[0] for r in rows):
        group = [r for r in rows if r[0] == deg and not r[1].startswith("None")]
        best = max(group, key=lambda r: r[5])
        print(f"Best for {deg}: {best[1]} ({best[2]}), PSNR {best[5]:.2f} dB")


# ---------------------------------------------------------------
# STEP 3: the results side by side, so numbers and visual quality can be compared
# ---------------------------------------------------------------
def show(img, rows):
    groups = [[r for r in rows if r[0] == deg] for deg in dict.fromkeys(r[0] for r in rows)]
    fig, axes = plt.subplots(2, 4, figsize=(18, 10), layout="constrained")
    for row, group in enumerate(groups):
        for col, (deg, method, param, im, e, p) in enumerate(group):
            axes[row, col].imshow(np.clip(im[ZOOM], 0, 1), cmap="gray", vmin=0, vmax=1)
            name = "Degraded" if method.startswith("None") else f"{method} ({param})"
            axes[row, col].set_title(f"{name}\nMSE {e:.1f}   PSNR {p:.2f} dB", fontsize=10)
        for ax in axes[row]:
            ax.axis("off")
        axes[row, 0].text(-0.08, 0.5, group[0][0], rotation=90, va="center", ha="right",
                          transform=axes[row, 0].transAxes, fontsize=11)
    fig.suptitle("Q7: restoration results with MSE and PSNR (zoomed)", fontsize=13)
    fig.savefig(OUT_DIR / "q7_comparison.png", dpi=150, bbox_inches="tight")


# RESULT (cameraman) - the completed table:
#   Salt & pepper:  none 1076.6 / 17.81 dB | average 3x3 209.4 / 24.92 | MEDIAN 3x3 63.5 / 30.10
#                   | spatial Wiener 3x3 673.3 / 19.85
#   Blur + noise:   none 469.5 / 21.41 dB | inverse |H|>0.1 336.2 / 22.87
#                   | frequency-domain WIENER K=0.01 268.7 / 23.84
#   SAY: "For salt and pepper the median filter is clearly best, plus 12 dB. For blur plus noise
#         the frequency-domain Wiener filter is best, plus 2.4 dB, ahead of the inverse filter."
#   SAY: "The numbers mostly agree with what we see. But not always: the average filter scores
#         25 dB although it looks blurred and blotchy, and the inverse result has ringing that
#         the small PSNR difference does not show. PSNR treats every pixel error the same and
#         does not measure sharpness or patterns."
#   SAY: "A filter only works when its model matches the noise. The median ignores OUTLIERS -
#         salt and pepper is outliers, but blur is not. Inverse and Wiener UNDO a known H -
#         salt and pepper has no H to undo. The spatial Wiener filter assumes small Gaussian
#         noise, so it keeps the big impulses as if they were detail."
if __name__ == "__main__":
    image = load_gray(IMAGE_PATH)
    rows = run_all(image)
    save_table(rows)
    print()
    best_per_degradation(rows)
    show(image, rows)
    print(f"\nTable saved as {OUT_DIR / 'q7_table.csv'}")
    plt.show()
