# 📄 DocScan v3.0 — The Professional Document Scanner

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)](https://opencv.org/)

**DocScan** takes a skewed photo of a book, receipt, whiteboard, or envelope and flattens it into a perfectly straight, auto-cropped rectangle — **at full original resolution**, in under 2 seconds.

> Unlike naive implementations that warp a downscaled preview (destroying text legibility), this pipeline runs edge-detection on a low-res copy for speed, but applies the perspective warp to the **original high-resolution image**.

---

## 🚀 What Makes This Version Different (v3.0)

| Feature | Why It Matters |
| :--- | :--- |
| **Full-Resolution Warp** | Detection runs on a downscaled copy (max 500px) for speed, but the warp applies to the **ORIGINAL** image. Your scanned documents remain crisp and readable. |
| **Angle-Based Corner Ordering** | The classic `sum/diff` heuristic fails when a document is rotated steeply (~45-90°), causing the warp to collapse into a zero-width strip. This version sorts corners by **angle around the centroid**, guaranteeing consistent ordering for *any* rotation. |
| **Multi-Epsilon Search + Convex Hull** | Real-world photos often produce fragmented contours (5-17 points). This version takes the **Convex Hull** first, then searches a range of epsilon values (0.01→0.05) until exactly 4 corners are found. This is the standard robust approach used in production pipelines. |
| **Built-in Debug Mode** | Pass `debug=True` to visualize the detected green contour and red corner points instantly. No more silent failures—you see exactly why the scanner succeeded or failed. |
| **Manual Rotation Parameter** | The scanner detects a rectangle, but it cannot magically know which side is "up" (especially if the document is photographed diagonally). Tesseract OSD fails on low-contrast text. Instead of pretending, this exposes a manual `rotate` parameter (90/180/270) — honest, simple, and easy to wire into a UI button. |
| **Portable Upload** | Works in Google Colab (file picker) AND local environments (loads from a `test_images/` folder). No hard dependency on Colab. |
| **Synthetic Test Suite** | Includes **Mild Skew**, **Harsh Perspective**, and **Low Contrast** tests. This ensures the detector is validated against real-world edge cases before you even upload a photo. |

---

## 🧠 How It Works (The Pipeline)

1. **Downscaled Detection**: The image is resized to max 500px. Edges are detected via Canny (50/150) and dilated to close small gaps.
2. **Contour Filtering**: Only the top 6 largest contours are kept; any contour smaller than 8% of the image area is discarded (this filters out text and noise).
3. **Convex Hull + Multi-Epsilon**: The convex hull is computed to smooth over gaps. `approxPolyDP` is applied with epsilons ranging from 1% to 5% of the perimeter until exactly 4 points are found.
4. **Angle-Based Ordering**: Points are sorted by angle around the centroid, then rotated to start at the top-left corner. This fixes the "collapsed warp" bug found in 90% of naive doc-scanners.
5. **Full-Resolution Warp**: The 4 corners are scaled back to original coordinates. `getPerspectiveTransform` and `warpPerspective` are applied to the **full-resolution original**, preserving text clarity.
6. **Auto-Crop**: Black borders are removed automatically.
7. **(Optional) Manual Rotation**: If the output is upside-down, rotate it via the `rotate` parameter (no unreliable auto-rotation).

---

## 🛠️ Installation & Usage

### Clone & Install

```bash
git clone https://github.com/[YOUR_USERNAME]/DocScan.git
cd DocScan
pip install opencv-python numpy matplotlib
