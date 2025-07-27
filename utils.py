import cv2
import numpy as np

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (200, 200))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    avg_hsv = cv2.mean(hsv)[:3]
    h, s, v = round(avg_hsv[0]), round(avg_hsv[1]), round(avg_hsv[2])

    lower_yellow = (20, 100, 100)
    upper_yellow = (40, 255, 255)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_pixels = cv2.countNonZero(mask)
    total_pixels = hsv.shape[0] * hsv.shape[1]
    yellow_ratio = yellow_pixels / total_pixels

    return [h, s, v, round(yellow_ratio, 4)]

def generate_yellow_focus_image(image_path, save_path):
    import cv2
    img = cv2.imread(image_path)
    img = cv2.resize(img, (200, 200))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold warna kuning
    lower_yellow = (20, 100, 100)
    upper_yellow = (40, 255, 255)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Overlay ke gambar asli
    yellow_area = cv2.bitwise_and(img, img, mask=mask)

    # Gabungkan original dan mask (optional)
    combined = cv2.addWeighted(img, 0.6, yellow_area, 0.8, 0)

    # Simpan gambar hasil
    cv2.imwrite(save_path, combined)
