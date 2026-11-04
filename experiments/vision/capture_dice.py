
from pyniryo import NiryoRobot
from pyniryo.vision import uncompress_image

import cv2
import time
from pathlib import Path
from datetime import datetime

ROBOT_IP = "169.254.200.200"
OUTPUT_DIR = Path("captured_dice")

MIN_AREA = 1500
MAX_AREA = 100000
SAVE_INTERVAL = 1.5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

robot = None
last_save_time = 0


def find_dice_candidates(img):
    """
    Find approximately square/rectangular objects.
    Detection uses a processed copy; the original image is unchanged.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if not MIN_AREA <= area <= MAX_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, 0.04 * perimeter, True
        )

        if len(approx) < 4 or len(approx) > 6:
            continue

        x, y, w, h = cv2.boundingRect(approx)

        if h == 0 or w == 0:
            continue

        aspect_ratio = w / h

        if not 0.65 <= aspect_ratio <= 1.45:
            continue

        candidates.append((x, y, w, h))

    return candidates


try:
    robot = NiryoRobot(ROBOT_IP)
    print("Connected to Niryo NED.")
    print("Saving raw images when dice-like objects are detected.")
    print("Press Q to quit.")

    while True:
        img_compressed = robot.get_img_compressed()
        img_raw = uncompress_image(img_compressed)

        if img_raw is None:
            print("Could not decode camera image.")
            time.sleep(0.2)
            continue

        candidates = find_dice_candidates(img_raw)
        now = time.time()

        if candidates and now - last_save_time >= SAVE_INTERVAL:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            frame_path = OUTPUT_DIR / f"dice_{timestamp}.png"

            # Save the original, unmodified camera frame.
            cv2.imwrite(str(frame_path), img_raw)

            # Save individual candidate crops from the original frame.
            for index, (x, y, w, h) in enumerate(candidates):
                crop = img_raw[y:y + h, x:x + w]

                if crop.size:
                    crop_path = OUTPUT_DIR / (
                        f"dice_{timestamp}_crop_{index}.png"
                    )
                    cv2.imwrite(str(crop_path), crop)

            print(
                f"Detected {len(candidates)} candidate(s). "
                f"Saved: {frame_path}"
            )

            last_save_time = now

        # Display a copy with candidate boxes; never save this display.
        display = img_raw.copy()

        for x, y, w, h in candidates:
            cv2.rectangle(
                display, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )

        cv2.imshow("Dice detection (Q to quit)", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    cv2.destroyAllWindows()

    if robot is not None:
        robot.close_connection()

    print("Capture stopped.")
