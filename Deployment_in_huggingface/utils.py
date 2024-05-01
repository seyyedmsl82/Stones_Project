# Import necessary libraries
import cv2
from PIL import Image
import numpy as np
import torch
from torch import nn


def calculator(x1, y1, x2, y2, w, h):
    # Calculate the slope and intercept of the line
    slope = (y2 - y1 + 1e-5) / (x2 - x1 + 1e-5)
    intercept = y1 - slope * x1

    # Set the starting and ending points for the line
    x1 = w
    y1 = int(slope * x1 + intercept)

    x2 = int((h - intercept) // slope)
    y2 = h

    return x1, y1, x2, y2, slope, intercept


def image_cropper(image):
    h, w = image.shape[:2]
    w, h = w // 6, h // 6
    image = cv2.resize(image, (w, h))

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    g_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # if gray_image.dtype != np.uint8:
    #     gray_image = gray_image.astype(np.uint8)

    edges = cv2.Canny(g_image, threshold1=100, threshold2=300)

    # Define the contour
    contour = np.array([[w // 4, h // 8], [3 * (w // 4), h // 8], [3 * (w // 4), 7 * (h // 8)], [w // 4, 7 * (h // 8)]])
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.int32([contour]), 255)
    outside_mask = cv2.bitwise_not(mask)

    # Apply the outside_mask to the edges image to keep the portion outside the contour
    masked_edges = cv2.bitwise_and(edges, outside_mask)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=70,
        maxLineGap=10
    )

    if lines is not None:
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x1 > 3 * (w // 4) and y1 < 7 * (h // 8):
                right_lines.append(line)
            elif x1 < w // 4 and y1 < 7 * (h // 8):
                left_lines.append(line)

        x1_l = 0
        x1_r = x2_r = w
        y_u = h // 2
        y_d = h

        if len(left_lines) > 0:
            left_line_avg = np.mean(left_lines, axis=0, dtype=np.int32)
            x1, y1, x2, y2 = left_line_avg[0]
            x1, y1, x2, y2, slope, intercept = calculator(x1, y1, x2, y2, w, h)
            x1_l = int((h // 2 - intercept) // slope)

            # cv2.line(image, (x1_l, h//2), (x2_l, h), (0, 0, 255), 2)

        if len(right_lines) > 0:
            right_line_avg = np.mean(right_lines, axis=0, dtype=np.int32)
            x1, y1, x2, y2 = right_line_avg[0]
            x1, y1, x2, y2, slope, intercept = calculator(x1, y1, x2, y2, w, h)
            x1_r = int((h // 2 - intercept) // slope)
            x2_r = int((h - intercept) // slope)

            # cv2.line(image, (x1_r, h//2), (x2_r, h), (0, 0, 255), 2)

        # original_points = np.float32([[x1_l, y_u], [x1_r, y_u], [x2_r, y_d], [x2_l, y_d]])
        original_points = np.float32([[x1_l, y_u], [x1_r, y_u], [x2_r, y_d]])
        target_points = np.float32([[0, 0], [w, 0], [w, h]])

        affine_matrix = cv2.getAffineTransform(original_points, target_points)

        # Apply the affine transformation to the image
        affine_image = cv2.warpAffine(image, affine_matrix, (image.shape[1], image.shape[0]))
        equalized_image = histogram_equalization_color(affine_image)

        return equalized_image

    image = image[h // 2:h, :]
    image = histogram_equalization_color(image)

    return image


def histogram_equalization_color(image):
    # Split the color image into individual channels
    b, g, r = cv2.split(image)

    # Apply histogram equalization to each channel separately
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)

    # Merge the equalized channels back into a color image
    equalized_image = cv2.merge([b_eq, g_eq, r_eq])

    return equalized_image
