# Niryo Dice Counting & Sorting

Project for **Praktický robotický seminár** (*Practical Robotics Seminar*) at FMFI UK.

Using computer vision and image processing to detect, count, and sort dice using the Niryo Ned robot.

## Setup

```bash
pip install pyniryo==1.1.2 numpy opencv-python
```

After powering on the robot, run `calibration.py` to calibrate it before operating the robot!

## Selecting a die and teaching camera close-ups

The rectified detector supports click selection and **operator-confirmed** physical
close-ups. No movement is performed merely by starting the detector or clicking.
Use the project interpreter in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\experiments\vision\dice_object_detection-rectified.py --calibrate
```

This camera calibration is separate from the robot's joint calibration. A saved
overview joint position alone cannot convert camera pixels into robot coordinates.
The teaching command reads camera images and tool poses; it does not move the arm.

1. Manually put the robot at the saved overview pose, looking down at the board.
   Fix the board, camera mounting, and tool for the entire calibration and use.
2. Place five identifiable dice/targets of the same height as the working dice.
   Four should surround the working area; the fifth must be inside it.
3. Type `CALIBRATE`. On the frozen, undistorted overview, click the four outer
   target centres, then the interior target centre. Press Enter (`R` clears the
   points, `Q` cancels).
4. For each numbered target, use Niryo Studio to move the camera manually until
   that target is centred on the live crosshair. Choose an unobstructed, safe
   close-up height where the die is in focus and no other die is visible.
   **Keep exactly the same roll/pitch/yaw as the overview and the same tool Z
   height for all five close-ups.** Stop manual control before pressing `C` to
   record each pose. The script cannot determine safe clearance for you.
5. The script fits four pixel-to-camera-centred-tool correspondences and checks
   the independent fifth point (maximum error 5 mm). Different close-up heights
   over 2 mm or orientation differences over 0.02 rad are rejected. The resulting
   local JSON file is ignored by Git.
6. Manually return to the **original overview pose**, then start normally:

```powershell
.\.venv\Scripts\python.exe .\experiments\vision\dice_object_detection-rectified.py
```

Click inside a detected die (yellow outline), then press `M`. The console prints
the die's undistorted pixel centre and planned **tool poses**, not a measured 3D
die position. Inspect the full arm/tool path and type `MOVE` in the terminal to
confirm. The camera first travels horizontally at overview Z, then vertically
down to the taught close-up Z. Outside-area targets, changed camera resolution or
intrinsics, and an incorrect starting pose block movement. Velocity is set to
10% and remains at 10% afterwards.

At the close-up, the main window shows the actual live camera image. Press `B`
and confirm `MOVE` to rise vertically before returning horizontally to the
overview. `Q` exits **without moving the robot back**. A failed motion stops the
session; inspect the robot and return manually before restarting.

**Safety and limits:** keep an emergency stop accessible and verify clearance
for the entire arm, not just the camera. This empirical planar mapping is not
collision avoidance. Do not move dice while confirming a motion. Re-teach after
changing the overview, board plane, die height, camera mounting, tool, or close-up
distance. Targets must remain within the calibrated region and detector ROI.
The taught framing determines magnification: one isolated die in view is not
guaranteed if neighbours are too close, the die height changes, or the teaching
is inaccurate. Check the live view; the script never descends further on its own.

Offline calibration/confirmation tests (no robot connection or movement):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_dice_camera_calibration.py
```

## Structure

- `src/`: main application (WIP)
- `experiments/`: prototypes and tests for vision algorithms and robot movement
- `diagnostics/`: connection, calibration, camera and robot position tools
- `tests/`: tests for vision and counting using saved images (WIP)
- `data/`: sample images for testing (WIP)
- `docs/`: assignment instructions and project notes

## Resources

- [Assignment instructions](docs/kocky.md)
- [Niryo Ned documentation](https://docs.niryo.com/robots/ned/)

Work in progress...
