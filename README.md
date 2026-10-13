# Niryo Dice Counting & Sorting

Project for **Praktický robotický seminár** (*Practical Robotics Seminar*) at FMFI UK.

Using computer vision and image processing to detect, count, and sort dice using the Niryo Ned robot.

## Setup

```bash
pip install pyniryo==1.1.2 numpy opencv-python
```

After powering on the robot, run `calibration.py` to calibrate it before operating the robot!

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

Work in progress.
