# Niryo Dice Counting & Sorting

Project for **Praktický robotický seminár** (*Practical Robotics Seminar*) at FMFI UK.

Using computer vision and image processing to detect, count, and sort dice using the Niryo Ned robot.

## Setup

```bash
pip install pyniryo==1.1.2 numpy opencv-python
```

After powering on the robot, run `calibration.py` to calibrate it before operating the robot!

## Automatic board detection

The rectified detector locates the four concentric circular Niryo markers in each
undistorted camera frame, checking nested, approximately concentric round contours
with adaptive and global thresholding, including a lower-contrast Gaussian
threshold for slightly blurred rings. This avoids treating a square die with a
central pip as a ring marker, which the installed SDK candidate finder can do.
There is no fixed rectangular ROI. The cyan outline follows the usable interior
of the marker quadrilateral; yellow `M1`-`M4` labels show the detected corners.
The marker rings and a small border margin are excluded from dice detection.
Only dice contours entirely inside this mask are accepted. Coordinates remain
in the original undistorted image so click selection and camera calibration use
the same pixel coordinate system.

Keep all four markers visible, in focus and uncovered, with sufficient contrast.
If fewer or more than four marker candidates are found, or their geometry is
invalid, the display says `Workspace NOT found` and `Dice: --`; detection and
approach requests are blocked instead of reusing an old region or guessing a
missing corner. The close-up view and confirmed return do not require the board
markers to remain in view. This detector is intended for the Niryo ring-marker
board, not arbitrary boards or generic object recognition.
The missing-workspace warning is drawn at the bottom of the view so it does not
cover the upper markers in diagnostic screenshots.

### Automatic positions without robot movement (default)

Run the rectified detector normally, including with VS Code's Run Python File:

```powershell
.\.venv\Scripts\python.exe .\experiments\vision\dice_object_detection-rectified.py
```

No five-point manual teaching or saved Workspace in Studio is needed for this
mode. A perspective transform from the four detected marker centres defines
normalised board coordinates for each die's image centre:
`M1=(0,0)`, `M2=(1,0)`, `M3=(1,1)`, `M4=(0,1)`.
The origin is the image's upper-left marker, X runs towards M2 and Y towards M4.
Each die shows `x` and `y`; clicking it prints more precise coordinates and its
original pixel centre in the terminal.

These are estimated planar image positions, **not millimetres, depth or robot
XYZ**. Object height introduces parallax. Corner labels follow image orientation,
not encoded physical marker identities; rotating the board can change the origin
and axes. Die numbers are assigned each frame, not persistent tracking IDs.
Positions are recomputed from the current frame and disappear if markers are
lost. The default mode never enables robot motion, does not load a saved motion
calibration, and blocks both `M` and `B`.

Automatic board detection is **not robot-coordinate calibration** and does not
make it safe to move the board after teaching. Re-teach the camera mapping after
changing the board or overview pose. Motion still requires a valid taught
calibration and explicit confirmation.

## Non-contact camera-to-arm calibration (read-only)

Run [dice_hand_eye_calibration.py](experiments/vision/dice_hand_eye_calibration.py)
with VS Code's Run Python File, or:

```powershell
.\.venv\Scripts\python.exe .\experiments\vision\dice_hand_eye_calibration.py --board-size-mm 170
```

The measured distance **between marker centres** is 170 mm on each side of the
current square board. Change the argument if using a different square board.
No calibration tip or precise centring over dice is needed. This is not
single-image auto-calibration: the camera-to-arm transform needs diverse robot
poses. The script sends **no motion commands**, including after a successful fit.

1. Keep the empty board fixed. Start in a clear overview pose with all four
   markers visible. Keep the camera and tool mounting rigid throughout.
2. Type `CAPTURE` in the terminal. This starts a new observation session;
   previous draft observations are replaced. An existing checked result is
   replaced only after another successful fit.
3. Use Studio to choose safe overview poses, never descend towards the board.
   Keep **joint 6 fixed at its initial angle**: the camera can be attached before
   this joint, so an effective camera-to-tool transform is valid only at that
   angle. This restriction also applies to any future use of the result.
4. Stop manual control, wait at least one second, then press `C` in the image
   window. Collect at least **8 distinct views**. The first six (or more) must
   include changes of orientation about **two different axes**, not just X/Y
   shifts or rotation around one axis. Their total rotational span must reach
   at least 15 degrees; do not make a move unless you can verify clearance.
   There are no prescribed joint targets or automatic rotations.
5. Keep labels M1-M4 on the same physical markers in every view; avoid large
   camera roll or crossing to the other side of the board. The ring detector's
   labels are image-oriented, not persistent marker IDs.
6. The final **two views** are independent checks and are not used in fitting.
   They should also be different overview poses. `U` removes the last view.
   `F` solves and checks the geometry; `Q` leaves without fitting.

Captures reject motion, unstable images, missing markers, duplicate poses,
changed joint 6, ambiguous planar pose, and board reprojection RMSE over 3 px.
The rigid hand-eye equation `A X = X B` is solved using rotation-vector SVD and
translation least squares (the installed OpenCV 5 build does not expose
`calibrateHandEye`). Both training and held-out views must agree on all four
board corners within 5 mm and orientation within 2 degrees.

Unvalidated observations are saved to `dice_hand_eye_samples.json`; there is no
automatic resume. A successful fit writes `dice_hand_eye_calibration.json` with
explicitly named **camera-to-tool** and **board-to-base** transforms, intrinsics,
joint-6 restriction, raw samples and per-view check errors. Both files are local
and ignored by Git. Size measurement error, camera calibration error and robot
kinematic errors can bias all views together: these consistency checks are
**not a guarantee of absolute accuracy or collision safety**.

**The result is not yet connected to motion in the dice detector.** It does not
create a Studio Workspace, replace the older five-point file, or enable `M`.
Independent physical validation and a safe approach planner are required before
using this geometry for movement.

## Optional: selecting a die and teaching camera close-ups

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
6. Manually return to the **original overview pose**, then explicitly enable motion:

```powershell
.\.venv\Scripts\python.exe .\experiments\vision\dice_object_detection-rectified.py --enable-motion
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
distance. Targets must remain within the calibrated region and detected board interior.
The taught framing determines magnification: one isolated die in view is not
guaranteed if neighbours are too close, the die height changes, or the teaching
is inaccurate. Check the live view; the script never descends further on its own.

Offline calibration/confirmation tests (no robot connection or movement):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_dice*calibration.py"
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

.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_dice_camera_calibration.py

3. Priprav kalibračné body
V Niryo Studio ručne nastav robota do bezpečnej hornej polohy nad plochou — to bude overview poloha.
Priprav p päť nehybných, dobre rozpoznateľných bodov alebo kociek: štyri rozmiestni okolo okrajov pracovnej oblasti a piaty dovnútra. Všetky musia byť v rovnakej výške ako kocky pri neskoršom použití.
Pre každú z piatich polôh musí byť možné bezpečne priblížiť kameru. Zvoľ jednu bezpečnú výšku priblíženia a rovnakú orientáciu kamery pre všetky body.
4. Spusti kalibráciu kamery
Z koreňového priečinka projektu spusti:
.\.venv\Scripts\python.exe .\experiments\vision\dice_object_detection-rectified.py --calibrate

Potom:

V termináli napíš CALIBRATE a stlač Enter.
V zobrazenom nehyblivom overview obrázku klikni na stredy štyroch vonkajších bodov a potom na stred vnútorného bodu.
Po piatom bode stlač Enter. R vymaže výber bodov; Q kalibráciu zruší.
Program ťa bude postupne vyzývať na priblíženie ku každému bodu. V Niryo Studio ručne nastav robota tak, aby cieľ ležal pod červeným krížom živého obrazu.
Pri všetkých piatich bodoch zachovaj rovnakú orientáciu aj výšku nástroja. Zastav ručné ovládanie a až potom stlač C, aby sa zaznamenala poloha.
Po poslednom bode program uloží kalibráciu do experiments\vision\dice_camera_calibration.json.
Ak kalibrácia zahlási chybu, skontroluj výber bodov, rovnakú výšku a orientáciu. Piata poloha musí byť vnútri prvých štyroch a jej kontrolná chyba musí byť najviac 5 mm. Neobchádzaj túto kontrolu použitím nepresnej kalibrácie.

5. Vráť robota do overview polohy
Po kalibrácii program nevracia robota automaticky. Ručne vráť ho do pôvodnej overview polohy, z ktorej si kalibráciu začal. Kalibračný súbor si ponechaj. Kalibráciu treba zopakovať, ak zmeníš polohu kamery, nástroja alebo dosky, výšku kociek, overview polohu či pracovnú oblasť.

6. Spusti detekciu
V PowerShelli spusti skript bez prepínača --calibrate.
Predvolene ide o rezim bez pohybu; na potvrdene priblizenie treba --enable-motion:

.\.venv\Scripts\python.exe .\experiments\vision\dice_object_detection-rectified.py

Zobrazí sa obraz kamery a okná s detekovanými kockami a relativnymi suradnicami.
Bez --enable-motion je pohyb vzdy zablokovany. Aj s tymto prepinacom je potrebna
platna kalibracia a potvrdenie MOVE.

7. Ovládanie
Klikaj do hlavného okna Dice Detection. Prikazy M a B vyzaduju --enable-motion:

Klik na detegovanú kocku — vyberie ju.
M — požiada o priblíženie k vybranej kocke. Skontroluj naplánované polohy a celú dráhu ramena. V termináli napíš MOVE iba vtedy, ak je pohyb bezpečný; čokoľvek iné pohyb zruší.
B — z detailného pohľadu požiada o návrat: najprv nahor a potom do overview polohy. Aj tento pohyb treba potvrdiť napísaním MOVE v termináli.
Q — ukončí program, ale robota nevráti. Ak je robot v detailnej polohe, vráť ho bezpečne ručne.
R — resetuje referenčný obrázok na zarovnanie.
S — uloží binárne obrázky detegovaných kociek do priečinka dice_dataset v aktuálnom pracovnom priečinku.
Skript nastavuje rýchlosť ramena na 10 % a po skončení ju neobnovuje. Priblíženie je určené na kontrolu obrazu, nie na uchopenie ani triedenie kociek. Tento postup nepredstavuje detekciu kolízií — bezpečnosť celej dráhy musíš vždy overiť ty.