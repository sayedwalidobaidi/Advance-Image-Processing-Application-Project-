# Advanced Image Processing Applications – Assignment 2

Edge detection, thresholding and image restoration in Python, applied to one grayscale image
(the "cameraman" test image by default; set `IMAGE_PATH` in `q1_edge_detection.py` to use another).

| File | Question |
|---|---|
| `q1_edge_detection.py` | Q1 – Roberts, Prewitt, Sobel and Kirsch edge detection |
| `q2_laplacian.py` | Q2 – Laplacian masks, zero crossings, noise sensitivity |
| `q3_thresholding.py` | Q3 – Manual, histogram-based and local max–min thresholding |
| `q4_degradation.py` | Q4 – Salt-and-pepper noise; motion blur + Gaussian noise |
| `q5_spatial_filters.py` | Q5 – Average, median and spatial Wiener filters |
| `q6_inverse_wiener.py` | Q6 – Inverse and frequency-domain Wiener restoration |
| `q7_metrics.py` | Q7 – MSE / PSNR comparison table |
| `q8_edges_after_restoration.py` | Q8 – Sobel edges before and after restoration |

Figures are saved in `results/`.

## Run

```bash
py -m pip install numpy scipy scikit-image matplotlib
py q1_edge_detection.py
```

The files import from each other, so run them from this folder.
