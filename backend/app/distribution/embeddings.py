"""
Environmental and Visual Feature Extractor for Distribution Analysis
Extracts brightness, contrast, color statistics, sharpness, and spatial frequencies.
"""
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import cv2

def extract_image_distribution_metrics(image_path: str) -> Dict[str, float]:
    """
    Extract comprehensive numerical visual features from an image:
    - Brightness (mean grayscale intensity)
    - Contrast (standard deviation of grayscale intensity)
    - Sharpness (Laplacian variance)
    - Colorfulness / Saturation
    - Mean Red, Green, Blue
    - High-frequency spatial energy (FFT)
    """
    try:
        with Image.open(image_path) as im:
            im_rgb = im.convert("RGB").resize((128, 128))
            arr = np.array(im_rgb, dtype=np.float32) / 255.0

            # RGB channels
            r_mean = float(np.mean(arr[:, :, 0]))
            g_mean = float(np.mean(arr[:, :, 1]))
            b_mean = float(np.mean(arr[:, :, 2]))

            # Grayscale
            gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))

            # Sharpness via Laplacian variance
            gray_uint8 = (gray * 255).astype(np.uint8)
            laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
            sharpness = float(np.var(laplacian))

            # Saturation via HSV
            arr_uint8 = (arr * 255).astype(np.uint8)
            hsv = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2HSV)
            saturation = float(np.mean(hsv[:, :, 1]) / 255.0)

            # High frequency energy (2D FFT)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-7)
            # Sample corner high frequencies
            h, w = magnitude_spectrum.shape
            high_freq_energy = float(np.mean(magnitude_spectrum[:h//4, :w//4]))

            return {
                "brightness": round(brightness, 4),
                "contrast": round(contrast, 4),
                "sharpness": round(sharpness, 2),
                "saturation": round(saturation, 4),
                "r_mean": round(r_mean, 4),
                "g_mean": round(g_mean, 4),
                "b_mean": round(b_mean, 4),
                "high_freq_energy": round(high_freq_energy, 3)
            }
    except Exception:
        return {
            "brightness": 0.5,
            "contrast": 0.2,
            "sharpness": 100.0,
            "saturation": 0.3,
            "r_mean": 0.5,
            "g_mean": 0.5,
            "b_mean": 0.5,
            "high_freq_energy": 50.0
        }
