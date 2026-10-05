# 📄 DocScan v3.0 — The Professional Document Scanner

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)](https://opencv.org/)

**DocScan** takes a skewed photo of a book, receipt, whiteboard, or envelope and flattens it into a straight, document-only rectangle — **at full original resolution**.

> Unlike naive implementations that warp a downscaled preview (destroying text legibility), this pipeline runs edge-detection on a low-res copy for speed, but applies the perspective warp to the **original high-resolution image**.

---

## 🚀 What Makes This Version Different (v3.0)

| Feature | Why It Matters |
| :--- | :--- |
| **Full-Resolution Warp** | Detection runs on a downscaled copy (max 500px) for speed, but the warp applies to the **ORIGINAL** image. Your scanned documents remain crisp and readable. |
| **Angle-Based Corner Ordering** | The classic `sum/diff` heuristic fails when a document is rotated steeply (~45-90°): the same point can win more than one role, so one corner is assigned twice and the warp collapses into a near-zero-width strip, with no error message. This version sorts corners by **angle around the centroid**, then checks the winding direction with a signed-area test so the result is never mirrored. |
| **Multi-Epsilon Search + Convex Hull** | Real-world photos often produce fragmented contours (5-17 points). This version takes the **Convex Hull** first, then searches a range of epsilon values (1%→5% of the perimeter) until exactly 4 corners are found. This is a common robust approach for document scanners. |
| **Built-in Debug Mode** | Pass `debug=True` to visualize the detected green contour and red corner points. If detection fails, it shows the input next to the edge map, so you can see why instead of getting a silent failure. |
| **Manual Rotation Parameter** | The scanner finds the document's rectangle, but it cannot know which side is "up". Tesseract's orientation detection (OSD) was tested and rejected: its confidence stayed near zero even on correctly oriented images of this content. Instead, a manual `rotate` parameter (90/180/270) is exposed — simple, honest, and easy to wire into a UI button. |
| **Portable Upload** | Works in Google Colab (file picker) AND local environments (loads from a `test_images/` folder). No hard dependency on Colab. |
| **Synthetic Test Suite** | Includes **Mild Skew**, **Harsh Perspective**, and **Low Contrast** tests, with the test document placed on a larger background so a closed outline exists to detect. This validates the detector against edge cases before you upload a photo. |

---

## 🧠 How It Works (The Pipeline)

1. **Downscaled Detection**: The image is resized so its longest side is at most 500px, converted to grayscale and blurred (Gaussian 5×5). Edges are detected via Canny (50/150) and dilated (3×3 kernel, 2 iterations) to close small gaps.
2. **Contour Filtering**: Only the 6 largest contours are kept; any contour smaller than 8% of the image area is discarded (this filters out text and noise).
3. **Convex Hull + Multi-Epsilon**: The convex hull is computed to smooth over gaps. `approxPolyDP` is applied with epsilons ranging from 1% to 5% of the perimeter until exactly 4 points are found.
4. **Angle-Based Ordering**: Points are sorted by angle around the centroid, the winding direction is checked with a signed-area test (reversing the order if needed so the warp is never mirrored), and the sequence is rotated to start at the top-left corner.
5. **Full-Resolution Warp**: The 4 corners are scaled back to original coordinates. `getPerspectiveTransform` and `warpPerspective` are applied to the **full-resolution original**. The output size comes directly from the detected corners (width = the longer of the top and bottom edges, height = the longer of the left and right edges), so the result contains only the document and no separate crop step is needed.
6. **(Optional) Manual Rotation**: If the output is upside-down, rotate it via the `rotate` parameter (no unreliable auto-rotation).

---

## 🛠️ Installation & Usage

### Clone & Install

```bash
git clone https://github.com/tolkinovmuhammadkodir-prog/Docscan.git
cd Docscan
pip install opencv-python numpy matplotlib
```

### Run the demo

```bash
python docscan_v2.py
```

This runs the three synthetic tests (mild skew, harsh perspective, low contrast), then scans any images you put in a `test_images/` folder. In Google Colab, it opens a file-upload picker instead.

### Use it in your own code

```python
import cv2
from docscan_v2 import scan_document

image = cv2.imread("photo.jpg")
scanned = scan_document(image, debug=True)  # returns None if no 4-corner document is found

if scanned is not None:
    cv2.imwrite("scan.png", scanned)
```

If the output is upside-down, pass `rotate=90`, `180`, or `270`.

if scanned is not None:
    cv2.imwrite("scan.png", scanned)
```

If the output is upside-down, pass `rotate=90`, `180`, or `270`.
