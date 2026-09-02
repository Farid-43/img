# 4_frequency_domain_filtering.py
"""
Frequency Domain Filtering — ALL filters built manually from equations
========================================================================
Filters (all built with manual for-loops, no vectorized shortcuts):

  Gaussian:      GLPF, GHPF, GBRF, Gaussian Notch-Reject
  Butterworth:   BLPF, BHPF, BBRF, Butterworth Notch-Reject

Pipeline:
  1. FFT → shift to center
  2. Build H(u,v) filter mask manually (nested loops)
  3. G(u,v) = |F(u,v)| * H(u,v)
  4. Recombine with phase: G(u,v) * e^(j*angle)
  5. Inverse FFT → output image
"""

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════
# 1. GAUSSIAN LOW-PASS FILTER (GLPF)
#    H(u,v) = exp( -D(u,v)² / (2·D0²) )
# ═══════════════════════════════════════════════════════════════
def make_gaussian_lpf(shape, D0):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    H = np.zeros((rows, cols), dtype=np.float64)

    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center_row)**2 + (v - center_col)**2)
            H[u, v] = np.exp(-(D**2) / (2 * D0**2))

    return H


# ═══════════════════════════════════════════════════════════════
# 2. GAUSSIAN HIGH-PASS FILTER (GHPF)
#    H(u,v) = 1 - GLPF(u,v)
# ═══════════════════════════════════════════════════════════════
def make_gaussian_hpf(shape, D0):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    H = np.zeros((rows, cols), dtype=np.float64)

    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center_row)**2 + (v - center_col)**2)
            H[u, v] = 1 - np.exp(-(D**2) / (2 * D0**2))

    return H


# ═══════════════════════════════════════════════════════════════
# 3. GAUSSIAN BAND-REJECT FILTER (GBRF)
#    H(u,v) = 1 - exp( -0.5 * [ (D² - D0²) / (D·W) ]² )
#    D0 = center frequency,  W = bandwidth
# ═══════════════════════════════════════════════════════════════
def make_gaussian_brf(shape, D0, W):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    H = np.zeros((rows, cols), dtype=np.float64)

    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center_row)**2 + (v - center_col)**2)
            if D == 0:
                H[u, v] = 0.0
            else:
                val = (D**2 - D0**2) / (D * W)
                H[u, v] = 1 - np.exp(-0.5 * val**2)

    return H


# ═══════════════════════════════════════════════════════════════
# 4. GAUSSIAN NOTCH-REJECT FILTER
#    For each notch point (u_k, v_k) and its symmetric point:
#    H(u,v) = ∏ [ 1 - exp(-D1²/(2·d0²)) ] * [ 1 - exp(-D2²/(2·d0²)) ]
# ═══════════════════════════════════════════════════════════════
def make_gaussian_notch_reject(shape, d0, u_k, v_k):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    H = np.ones((rows, cols), dtype=np.float64)

    # symmetric point
    sym_u = 2 * center_row - u_k
    sym_v = 2 * center_col - v_k

    for u in range(rows):
        for v in range(cols):
            D1 = np.sqrt((u - u_k)**2 + (v - v_k)**2)
            D2 = np.sqrt((u - sym_u)**2 + (v - sym_v)**2)
            H[u, v] = (1 - np.exp(-(D1**2) / (2 * d0**2))) * \
                       (1 - np.exp(-(D2**2) / (2 * d0**2)))

    return H


# ═══════════════════════════════════════════════════════════════
# 5. BUTTERWORTH LOW-PASS FILTER (BLPF)
#    H(u,v) = 1 / ( 1 + (D/D0)^(2n) )
# ═══════════════════════════════════════════════════════════════
def make_butterworth_lpf(shape, D0, n):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    H = np.zeros((rows, cols), dtype=np.float64)

    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center_row)**2 + (v - center_col)**2)
            if D == 0:
                H[u, v] = 1.0
            else:
                H[u, v] = 1 / (1 + (D / D0)**(2 * n))

    return H


# ═══════════════════════════════════════════════════════════════
# 6. BUTTERWORTH HIGH-PASS FILTER (BHPF)
#    H(u,v) = 1 / ( 1 + (D0/D)^(2n) )
# ═══════════════════════════════════════════════════════════════
def make_butterworth_hpf(shape, D0, n):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    H = np.zeros((rows, cols), dtype=np.float64)

    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center_row)**2 + (v - center_col)**2)
            if D == 0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1 / (1 + (D0 / D)**(2 * n))

    return H


# ═══════════════════════════════════════════════════════════════
# 7. BUTTERWORTH BAND-REJECT FILTER (BBRF)
#    H(u,v) = 1 / ( 1 + [ (D·W) / (D² - D0²) ]^(2n) )
# ═══════════════════════════════════════════════════════════════
def make_butterworth_brf(shape, D0, W, n):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    H = np.zeros((rows, cols), dtype=np.float64)

    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center_row)**2 + (v - center_col)**2)
            denom = D**2 - D0**2
            if denom == 0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1 / (1 + ((D * W) / denom)**(2 * n))

    return H


# ═══════════════════════════════════════════════════════════════
# 8. BUTTERWORTH NOTCH-REJECT FILTER
#    For each notch (u_k, v_k) and its symmetric point:
#    H(u,v) = ∏ [ 1/(1+(d0/D1)^(2n)) ] * [ 1/(1+(d0/D2)^(2n)) ]
# ═══════════════════════════════════════════════════════════════
def make_butterworth_notch_reject(shape, d0, n, u_k, v_k):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    H = np.ones((rows, cols), dtype=np.float64)

    sym_u = 2 * center_row - u_k
    sym_v = 2 * center_col - v_k

    for u in range(rows):
        for v in range(cols):
            D1 = np.sqrt((u - u_k)**2 + (v - v_k)**2)
            D2 = np.sqrt((u - sym_u)**2 + (v - sym_v)**2)

            if D1 == 0 or D2 == 0:
                H[u, v] = 0.0
            else:
                H[u, v] = (1 / (1 + (d0 / D1)**(2 * n))) * \
                           (1 / (1 + (d0 / D2)**(2 * n)))

    return H


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

# take input
img_input = cv2.imread('images/pnois2.jpg', 0)

if img_input is None:
    print("Error: 'images/pnois2.jpg' not found!")
else:
    img = img_input.copy()

    # ── Step 1: Fourier Transform ──
    ft = np.fft.fft2(img)
    ft_shift = np.fft.fftshift(ft)

    # ── Step 2: Magnitude & Phase ──
    magnitude_spectrum_ac = np.abs(ft_shift)
    ang = np.angle(ft_shift)

    # For visualization
    magnitude_spectrum = 20 * np.log(np.abs(ft_shift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255,
                                       cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    img_shape = ft_shift.shape

    # ── Parameters ──
    D0 = 30          # cutoff frequency
    W = 10           # bandwidth (for band-reject)
    n = 2            # Butterworth order
    d0_notch = 10    # notch radius
    u_k, v_k = 261, 261  # noise spike coordinate

    # ══════════════════════════════════════════════
    # Choose which filter to apply (change this!)
    # ══════════════════════════════════════════════
    # Options: 'GLPF', 'GHPF', 'GBRF', 'G_NOTCH',
    #          'BLPF', 'BHPF', 'BBRF', 'BW_NOTCH'

    filter_choice = 'GLPF'

    print(f"Building filter: {filter_choice} ...")

    if filter_choice == 'GLPF':
        H = make_gaussian_lpf(img_shape, D0)
        title = f"Gaussian Low-Pass (D0={D0})"

    elif filter_choice == 'GHPF':
        H = make_gaussian_hpf(img_shape, D0)
        title = f"Gaussian High-Pass (D0={D0})"

    elif filter_choice == 'GBRF':
        H = make_gaussian_brf(img_shape, D0, W)
        title = f"Gaussian Band-Reject (D0={D0}, W={W})"

    elif filter_choice == 'G_NOTCH':
        H = make_gaussian_notch_reject(img_shape, d0_notch, u_k, v_k)
        title = f"Gaussian Notch-Reject (d0={d0_notch})"

    elif filter_choice == 'BLPF':
        H = make_butterworth_lpf(img_shape, D0, n)
        title = f"Butterworth Low-Pass (D0={D0}, n={n})"

    elif filter_choice == 'BHPF':
        H = make_butterworth_hpf(img_shape, D0, n)
        title = f"Butterworth High-Pass (D0={D0}, n={n})"

    elif filter_choice == 'BBRF':
        H = make_butterworth_brf(img_shape, D0, W, n)
        title = f"Butterworth Band-Reject (D0={D0}, W={W}, n={n})"

    elif filter_choice == 'BW_NOTCH':
        H = make_butterworth_notch_reject(img_shape, d0_notch, n, u_k, v_k)
        title = f"Butterworth Notch-Reject (d0={d0_notch}, n={n})"

    print("Filter built. Applying...")

    # ── Step 3: Apply filter  G(u,v) = |F(u,v)| * H(u,v) ──
    filtered_spectrum = magnitude_spectrum_ac * H

    # ── Step 4: Recombine with phase  G(u,v) = |G(u,v)| * e^(j·θ) ──
    final_result = np.multiply(filtered_spectrum, np.exp(1j * ang))

    # ── Step 5: Inverse Fourier Transform ──
    img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
    img_back_scaled = cv2.normalize(img_back, None, 0, 255,
                                    cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # ── Display ──
    H_display = cv2.normalize(H, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imshow("1. Input Image", img_input)
    cv2.imshow("2. Magnitude Spectrum", magnitude_spectrum)
    cv2.imshow(f"3. Filter H(u,v) — {title}", H_display)
    cv2.imshow(f"4. Output — {title}", img_back_scaled)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
