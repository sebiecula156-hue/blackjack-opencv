# 🃏 Blackjack Computer Vision System (C++/OpenCV)

A sophisticated real-time playing card detection and identification system built with **C++** and **OpenCV**. This application is designed to isolate, stabilize, and extract card rank data from a live camera feed for use in a Blackjack automation engine.

---

## 🚀 Advanced Technical Features

* **Temporal Stability Logic:** Implements an **Intersection over Union (IoU)** algorithm to track card positions across frames. A 3-second stability timer ensures data is only captured when the card is held steady.
* **Intelligent Image Pre-processing:** Utilizes a custom pipeline consisting of **Gaussian Blurring**, **Canny Edge Detection**, and **Morphological Closing** to eliminate background noise and bridge contour gaps caused by fingers.
* **Geometric Rectification:** Features a high-precision **Perspective Warp** transformation that "unwarps" the card from any skewed angle into a normalized 200x300 top-down view.
* **Adaptive Feature Extraction:** Automatically extracts a specific **Region of Interest (ROI)** from the card's corner using **Adaptive Gaussian Thresholding**, preparing the rank and suit for template matching.
* **Contour Filtering:** Employs **Extent (Solidity)** and **Aspect Ratio** filters to distinguish playing cards from other rectangular objects or human hands in the frame.

---

## 🛠️ Project Architecture

* **`main.cpp`**: The core engine. Contains the frame-by-frame processing logic, IOU stabilization, and the main CV pipeline.
* **`CMakeLists.txt`**: Handles cross-platform build configuration and links necessary OpenCV modules.
* **Template Database**: A directory-based system (`/templates`) used for storing and comparing extracted card ranks.


---

## 📦 Requirements & Build

* **Prerequisites:**
   * C++ Compiler (supporting C++17 or higher)
   * **OpenCV 4.x** installed and configured in your system path.
   * **CMake** 3.10 or higher.
