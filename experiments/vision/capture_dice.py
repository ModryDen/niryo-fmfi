
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from pyniryo import NiryoRobot


ROBOT_IP = "169.254.200.200"
SAVE_DIR = Path("dice_photos")

MIN_AREA = 300
MAX_AREA = 50000
MIN_ASPECT = 0.65
MAX_ASPECT = 1.35


def get_camera_image(robot):
    compressed = robot.get_img_compressed()

    if isinstance(compressed, tuple):
        compressed = compressed[-1]

    if isinstance(compressed, np.ndarray):
        compressed = compressed.tobytes()

    data = np.frombuffer(compressed, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError("Nepodarilo sa načítať obraz z kamery.")

    return image


def detect_cubes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekcia hrán – iba na rozpoznávanie objektov.
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_AREA or area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        polygon = cv2.approxPolyDP(
            contour,
            0.04 * perimeter,
            True
        )

        # Kocka musí mať približne štyri viditeľné strany.
        if len(polygon) != 4:
            continue

        x, y, w, h = cv2.boundingRect(polygon)

        if h == 0:
            continue

        aspect = w / h

        if not MIN_ASPECT <= aspect <= MAX_ASPECT:
            continue

        # Kontrola približne štvorcového tvaru.
        rect_area = w * h

        if rect_area == 0:
            continue

        fill_ratio = area / rect_area

        if fill_ratio < 0.45:
            continue

        detections.append((x, y, w, h))

    # Odstránenie duplicitných a prekrývajúcich sa detekcií.
    detections.sort(key=lambda item: item[2] * item[3], reverse=True)

    filtered = []

    for candidate in detections:
        x, y, w, h = candidate

        duplicate = False

        for fx, fy, fw, fh in filtered:
            ix1 = max(x, fx)
            iy1 = max(y, fy)
            ix2 = min(x + w, fx + fw)
            iy2 = min(y + h, fy + fh)

            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

            if intersection > 0.5 * min(w * h, fw * fh):
                duplicate = True
                break

        if not duplicate:
            filtered.append(candidate)

    return filtered


def save_cubes(frame, detections):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    for index, (x, y, w, h) in enumerate(detections, start=1):
        # Malá rezerva okolo kocky.
        padding = 8

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        # Výrez z pôvodnej farebnej fotografie.
        cube_photo = frame[y1:y2, x1:x2]

        filename = SAVE_DIR / f"cube_{timestamp}_{index}.png"

        cv2.imwrite(str(filename), cube_photo)

        print(f"Uložená kocka: {filename}")


def main():
    robot = None

    try:
        robot = NiryoRobot(ROBOT_IP)
        print("ROBOT CONNECTED")

        print("S = uložiť nájdené kocky")
        print("Q = ukončiť")

        while True:
            frame = get_camera_image(robot)

            detections = detect_cubes(frame)

            preview = frame.copy()

            for index, (x, y, w, h) in enumerate(detections, start=1):
                cv2.rectangle(
                    preview,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    preview,
                    f"Cube {index}",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                preview,
                f"Detected cubes: {len(detections)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.imshow("Niryo - Cube Detection", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                save_cubes(frame, detections)

            elif key == ord("q"):
                break

    except Exception as error:
        print("ERROR:")
        print(f"{type(error).__name__}: {error}")

    finally:
        cv2.destroyAllWindows()

        if robot is not None:
            robot.close_connection()

        print("Connection closed.")


if __name__ == "__main__":
    main()
