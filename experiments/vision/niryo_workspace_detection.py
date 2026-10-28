
import cv2
from pyniryo import NiryoRobot, vision

ROBOT_IP = "169.254.200.200"
WORKSPACE_RATIO = 1.0


def main():
    robot = None

    try:
        robot = NiryoRobot(ROBOT_IP)
        print("ROBOT CONNECTED")
        print("Workspace detection started. Press Q to quit.")

        intrinsics = robot.get_camera_intrinsics()

        while True:
            # Use the existing camera-reading helper
            from dice_camera_calibration import read_frame

            frame = read_frame(robot, intrinsics)

            # Detect and display workspace markers
            markers_found, marker_debug = vision.debug_markers(
                frame,
                workspace_ratio=WORKSPACE_RATIO,
            )

            cv2.imshow("Niryo Workspace Markers", marker_debug)

            if markers_found:
                workspace = vision.extract_img_workspace(
                    frame,
                    workspace_ratio=WORKSPACE_RATIO,
                )

                if workspace is not None:
                    cv2.imshow("Extracted Workspace", workspace)
            else:
                print("Workspace markers not detected.")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as error:
        print(f"WORKSPACE DETECTION FAILED: {type(error).__name__}: {error}")

    finally:
        cv2.destroyAllWindows()

        if robot is not None:
            robot.close_connection()

        print("Connection closed.")


if __name__ == "__main__":
    main()
