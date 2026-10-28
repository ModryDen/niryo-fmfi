from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Workspace:
    corners: np.ndarray
    mask: np.ndarray

    def relative_position(self, pixel):
        destination = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(
            self.corners.astype(np.float32), destination
        )
        point = np.asarray([[pixel]], dtype=np.float32)
        relative = cv2.perspectiveTransform(point, transform)[0, 0]
        return float(relative[0]), float(relative[1])


def find_ring_markers(gray):
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 25
    )
    low_contrast = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    _, global_threshold = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    candidates = []
    for threshold in (adaptive, low_contrast, global_threshold):
        contours, hierarchy = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None:
            continue
        for index, contour in enumerate(contours):
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if not 7 <= radius <= min(gray.shape) * 0.12:
                continue
            center = np.array([cx, cy])
            nested = 0
            current = index
            while current >= 0:
                ring = contours[current]
                area = cv2.contourArea(ring)
                perimeter = cv2.arcLength(ring, True)
                if perimeter == 0:
                    break
                circularity = 4 * np.pi * area / perimeter ** 2
                if circularity < (0.83 if nested == 0 else 0.70):
                    break
                (rx, ry), _ = cv2.minEnclosingCircle(ring)
                if np.linalg.norm(np.array([rx, ry]) - center) > max(2, radius * 0.15):
                    break
                nested += 1
                current = hierarchy[0, current, 2]
            if nested >= 3:
                candidates.append((center, radius))

    # Each ring and threshold can produce the same marker; keep its largest contour.
    markers = []
    for center, radius in sorted(candidates, key=lambda item: item[1], reverse=True):
        if not any(
            np.linalg.norm(center - other) <= max(3, other_radius * 0.25)
            for other, other_radius in markers
        ):
            markers.append((center, radius))
    return markers


def detect_workspace(frame):
    """Locate exactly four Niryo ring markers without guessing missing corners."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    markers = find_ring_markers(gray)
    if len(markers) != 4:
        return None

    centers = np.asarray([center for center, _ in markers], dtype=np.float32)
    hull = cv2.convexHull(centers).reshape(-1, 2)
    if len(hull) != 4 or cv2.contourArea(hull) < 2500:
        return None
    corners = np.int32(np.round(order_corners(hull)))
    sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    radii = [radius for _, radius in markers]
    if min(sides) < 4 * max(radii) or max(sides) / min(sides) > 4:
        return None

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners, 255)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    for center, radius in markers:
        point = tuple(map(int, np.round(center)))
        cv2.circle(mask, point, int(np.ceil(radius)) + 3, 0, -1)
    return Workspace(corners, mask)


def order_corners(points):
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    start = np.argmin(points[:, 0] + points[:, 1])
    return np.roll(points, -start, axis=0)
