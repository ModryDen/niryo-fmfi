# Niryo Dice Counting & Sorting

Project for **Praktický robotický seminár** (*Practical Robotics Seminar*) at FMFI UK.

Using computer vision and image processing to detect, count, and sort dice using the Niryo Ned robot.

## Setup

```bash
pip install -r requirements.txt
```

After powering on the robot, run `calibration.py` to calibrate it before operating the robot!

## Pick a die anywhere inside the taught area

Use the validated `dice_pick_calibration.json` for a general image-to-tool mapping.
The four outer taught targets bound the usable region; the fifth is an independent
check and is not fitted. A fifth-point error above 5 mm disables this mapping;
the unvalidated draft is never silently used instead. The detector outlines the
calibrated region in orange and shows the selected tool XYZ. There is no fixed
cross in this mode: place same-height dice anywhere inside the orange boundary,
then click a detected die. The pose is calculated for that die, not taught point 5.

To select the full usable platform rather than the smaller four-target hull,
add `--board-area`. The orange outline then follows M1-M4; marker rings and
edge exclusions from the dice detector remain excluded. This explicitly extends
the same planar homography beyond its original training hull, without refitting
or modifying the independent fifth-point result. Projection singularities and
invalid board geometry are rejected. Edge accuracy is not established by the
interior fifth-point check: verify new edge positions with `--preview-pick`
before any descent. Positions beyond M1-M4 remain blocked.

First verify an upper approach without descent or gripper actuation:

```powershell
.\.venv\Scripts\python.exe .\experiments\vision\dice_object_detection-rectified.py --enable-motion --preview-pick
```

Click a die inside the orange outline, press P in the image window, inspect the
printed route, then type PREVIEW in the terminal. The arm moves above that die
at the taught overview height and returns. No gripper commands are sent.

After physical calibration and hardware faults have been resolved, full transfers use:

```powershell
.\.venv\Scripts\python.exe .\experiments\vision\dice_object_detection-rectified.py --enable-motion
```

Click a die, press P, and confirm PICK to pick and place it into the taught drop.
Do not add `--taught-target 5`: that option deliberately restricts picking to a
single fixed taught position. Each cycle requires confirmation. The tool keeps
the taught pickup orientation and height; individual die yaw is not followed.
Use isolated dice with clearance for the open fingers and the whole arm path.

If only the camera's returned overview view has small drift, add
`--refresh-area-view` and confirm `BOARD FIXED` only when the board, robot base,
camera/tool mounting and object height are unchanged. Unlike fixed-target refresh,
dice may occupy new positions. This records the current marker positions and
actual overview for the session, while preserving the ORIGINAL marker reference
and homography for XY conversion; calibration files are unchanged.

Detected marker correspondences register a selected current-frame pixel into the
original calibration image before calculating robot XY. The 3-pixel board drift
limit, overview pose/intrinsics checks, live-frame verification, scene recheck and
hardware-error blocking remain active. This registration does not permit moving
the physical board after teaching. Targets outside the four-point hull are rejected,
even if they lie on the larger board. Object height must match teaching. A passed
numeric fit and an upper preview do not establish grasp accuracy or collision safety.
No real robot motion was used for the software tests.

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

### Automatic recording, no fixed Joint 6

```powershell
.\.venv\Scripts\python.exe .\experiments\vision\dice_hand_eye_calibration.py --auto
```

Type `CAPTURE`, then use hand-guiding/Studio to choose safe overview poses.
The script does **not** move the arm or enable learning mode. Stop and release
manual control for at least 1.5 seconds at each pose; it automatically records
a stable view and, after at least eight views, attempts a fit. No `C`/`F` is
needed. You still need genuinely different orientations about two axes, with
all four markers visible and on the same labels; keep the board fixed and stay
clear of it. Recording pauses/rejects observations when visibility, image quality
or stability is insufficient.

Joint 6 is the last wrist rotation before the gripper, not gripper opening.
You do not need to hold it fixed in `--auto`. Instead, the program reads the
robot's actual URDF through the ROS bridge (`websocket-client`, TCP 9090),
computes the wrist pose, and checks its tool kinematics against the reported
robot pose (2 mm / 0.02 rad maximum disagreement). It also verifies the camera
stream is not already undistorted. Missing/incompatible models or ROS failures
block calibration; there is no hardcoded camera offset or automatic fallback.

The program tries a camera fixed relative to `tool_link` and one fixed relative
to the parent of `joint_6`. It only selects a mount if **exactly one** passes all
geometry/held-out checks. To distinguish these hypotheses it requires at least
12 degrees of joint-6 span in the training views and 8 degrees between the last
two check views. This is measured automatically; no precise angle setting is
needed. If more variety is needed, the console explains why; continue taking
different safe overview poses. `U` removes the last observation, `Q` exits.

Auto results use schema version 2: `reference_frame` names the selected link and
`reference_from_camera` maps camera coordinates into it. The exact robot model
is stored with the result, alongside the board pose, observations and validation
errors. Camera mounting must not change. This supports only the two tested
mount hypotheses; inconsistent or ambiguous data are rejected. Geometric
consistency is not proof of absolute accuracy or collision safety.

**Automatic recording/calculation is not automatic robot movement.** The result
remains `motion_enabled: false` and is not wired into the detector's movement.

### Legacy manual recording

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
## Pick a selected die into one taught drop location

The detector shows marker-centre edges and S1-S4: perspective projections of
side midpoints. These are calculated points, not independently detected markers
or the physical outer edges of the platform.

Run pickup teaching (separate from camera close-up calibration):

```powershell
.\.venv\Scripts\python.exe .\experiments\vision\dice_object_detection-rectified.py --teach-pick
```

Configure a Niryo gripper in Studio. Put five stationary dice of the working
height on the board: four outer targets and an interior check target. Start at
an overview pose with all four markers visible, whose Z clears the entire
transfer route and drop area. Type TEACH PICK; click the five die centres in
that order, then Enter. In Studio manually centre the GRIPPER at each pickup
position, keeping the same orientation and pickup Z. Release manual control
and type RECORD for each target. Teach one drop pose with the same orientation,
then RECORD DROP. Return manually to the original overview and type CHECK.
Teaching sends no arm or gripper commands. The independent fifth-point error
must be at most 5 mm. The local dice_pick_calibration.json is ignored by Git.

```powershell
.\.venv\Scripts\python.exe .\experiments\vision\dice_object_detection-rectified.py --enable-motion
```

Click a detected die, press P, check the printed route and type PICK. The program
rechecks the starting pose, gripper, board and dice before actuating. It opens,
travels above the die, descends, closes, lifts, travels above the drop, descends,
opens, lifts and returns to overview. Horizontal travel uses overview Z. Each
cycle needs selection and confirmation; speed stays at 10%. Camera close-ups
with M still require their own --calibrate teaching.

Keep the board fixed; marker drift above 3 pixels blocks picking. Re-teach after
changing the tool, camera, die height or setup. Gripper orientation is fixed,
not adapted to individual die angles. Use isolated dice with room for the open
fingers, verify clearance for the whole arm, and clear the drop before each
cycle. This does not plan collisions or stacking. It does not verify that a die
was actually grasped or released. A command failure stops without automatic
recovery or release. Physical accuracy needs validation on the robot; tests
use a simulated robot and send no hardware commands.

Pickup teaching now saves every recorded pose to dice_pick_draft.json before
validation, including rejected sessions. This draft does not enable motion.
Orientation checks compare physical rotations rather than separate Euler angles
to handle equivalent roll/pitch/yaw values near a vertical tool orientation.

Use --repair-pick to correct a complete saved draft without re-clicking image
points. This mode backs up the original draft and sends no motion commands.
Choose 1-5 or D, manually position the gripper in Studio, then RECORD. Invalid
orientation/height leaves the previous record intact. Corrections save after
each accepted record. With board and dice still fixed, return to the original
overview and choose CHECK to validate and save the motion calibration.

In --repair-pick, IMAGE lets you reselect the five original die image centres
from a fresh frozen frame at the original overview, preserving every robot
pose and the drop location. Board markers must match the reference and dice
must not have moved. Keep the original numbering; orange old labels are hints,
not new detections. Enter saves five new clicks; R resets; Q cancels. CHECK
still enforces the independent 5 mm error limit before enabling the mapping.

VIEW in pickup repair opens a diagnostic frozen camera frame at any robot
pose without changing or saving calibration. Original pixel labels are shown
as hints only; camera pose differences can shift them. Pose errors now report
position and physical rotation differences, limits, and actual/required poses.

Image repair saves new die centres and marker corners from the same frozen
frame, preserving the robot teaching. Older repaired mappings should repeat
IMAGE and CHECK once to refresh this reference. Preflight validation refusals
after PICK now keep the camera session open without actuating; errors after
actuation still terminate the session.

REFRESH in repair explicitly re-observes the image mapping when old marker
image coordinates prevent IMAGE. It requires the original robot overview, all
four detected markers, and typing SETUP UNCHANGED to attest that the board,
five dice, robot base and camera/tool mounting have not moved. It saves newly
clicked centres with marker corners from the same frame; all robot poses stay
unchanged. Cancel if the physical setup moved. It does not enable motion: CHECK
still enforces orientation, pickup height, independent fifth-point error, pose
and marker checks. Normal runtime marker drift protection remains unchanged.

REFRESH can now record the actual nearby camera overview even when its
orientation differs from the old camera reference. Position must remain within
2 mm of the previous overview. The original pickup orientation is retained
separately as pickup_reference_pose; taught grip/drop checks remain unchanged.
The robot must stay at the new overview during capture and CHECK. Starting
motion checks this newly recorded pose; pickup targets retain the taught grip
orientation, with orientation changes occurring on the printed approach route.
This does not solve robot repeatability or collision planning; inspect the route.

For a first physical trial of one unmoved taught die, --enable-motion
--taught-target 5 uses target 5 directly from the saved pickup draft. This is
a fixed-position trial, not a validated image-to-robot mapping. The clicked
die must remain within 5 pixels of its taught image position. Starting pose,
board reference, intrinsics, configured gripper and post-confirmation scene
checks remain active. It uses the full recorded pickup pose, including its
orientation, and the recorded drop. Click the central die, P, then inspect
the route and type PICK. Physical grasp/release is not automatically verified.
Arbitrary die positions still require a validated five-point calibration.

The linear-move adapter handles older Niryo servers that reject the installed
SDK extra empty frame name with "6 parameters expected but 7 given". Only that
explicit parameter rejection triggers a six-coordinate MOVE_LINEAR_POSE retry;
the successful legacy format is reused for the connection. Other motion,
kinematics and network errors are never retried. SDK packet tests intercept
the transport and do not connect to or move a robot.

After each linear move, pose feedback is polled for up to ten seconds,
requiring two consecutive readings within the original 2 mm / 0.02 rad limits.
No motion is resent while waiting. Pick/place also checks hardware status
before actuating and after each gripper command; reported motor errors block
further movement. Clear hardware faults in Studio before retrying a cycle.

Pose settling waits up to 10 seconds with the same accuracy limits.
The trial gripper settings are speed 300, peak effort 30%, holding effort 10%,
to reduce effort compared with SDK defaults. A successful grasp must still be
verified physically; these settings are not a guarantee of grip strength.
Hardware status is checked after every settled arm move as well as gripper
commands; a delayed reported overload stops descent or the next action.

After joint recalibration, diagnostics/check_empty_approach.py offers an empty
arm test above taught target 5 and back. It requires a clear overview above
Z=0.30 m within 20 mm of the saved overview, healthy hardware, Learning Mode
off, and typing TEST EMPTY ARM. Both endpoints use the starting Z. It sends no
gripper commands and never descends. Endpoint checks retain 2 mm / 0.02 rad
limits. A failure stops without an automatic return; results are saved to the
ignored diagnostics/empty_approach_report.json. Recalibration does not by itself
validate old pickup calibration or guarantee clearance for the entire arm.

The camera overview start/return check allows up to 2 degrees of orientation
difference while retaining the 2 mm position limit. This applies only to the
upper camera pose: pickup, drop, and the approach before descent retain the
original 0.02 rad angular check. Board-image validation remains mandatory
before pickup. The empty approach report records each stage angular limit.

The 2-degree angular limit now applies consistently to all upper approaches,
lifts, transfers and camera returns, including both legs of the empty-arm test.
Pickup and drop endpoints retain the strict 0.02 rad check immediately before
gripper actuation. The position limit remains 2 mm everywhere. Pick/place prints
each movement stage so a stopped cycle can be identified without guessing.

Before pickup planning and again after confirmation, the detector verifies
the ROS video stream is active and receives two distinct frame headers. A
cached TCP JPEG is not accepted as proof of live video. This read-only check
uses websocket-client and ROS bridge port 9090; failure blocks pickup without
actuating. If streaming is off, restore the camera connection rather than
moving the board or dice to match an old displayed frame.

For fixed taught-target trials only, --refresh-fixed-view asks for FIXED SETUP
to attest that the physical board, robot base, camera mounting and taught die
position are unchanged. It requires healthy hardware, the recorded overview
pose and a fresh video stream, then observes all four markers as the reference
for this run. It changes no calibration file or pickup/drop pose. Normal 3-pixel
marker drift protection applies against this session reference; the selected
die must still be within 5 pixels of its taught position. It cannot repair a
physically moved board or recalibrate the camera-to-robot mapping.

Strong glare can make a black die appear white and leave only a side/shadow
in the threshold mask. The detector now looks for a closed convex quadrilateral
edge outline in these cases. Recovered centres are labelled GLARE and remain
estimates: preview is allowed, full pick is blocked until lighting is corrected.
The regression fixture captures the actual reflective die; its estimated centre
shifts away from the old shadow contour. This does not correct robot-coordinate
extrapolation errors near board edges, which require additional physical checks.

### Lower inspection and experimental cube rotation

Add --preview-clearance-mm 80 to --preview-pick for a staged inspection:
high approach, vertical descent to taught grip Z + 80 mm, pause, vertical rise,
and return. Minimum permitted clearance is 30 mm; this is relative to tool grip
height, not measured table clearance. Confirm PREVIEW after checking the entire
arm path. At the inspection pose type RETURN to rise and go back. Anything else
ends the session at that pose without recovery. No gripper commands are sent.
The displayed overview image is frozen during these blocking operations; inspect
the real arm. This diagnostic does not correct an unknown XY offset.

--auto-yaw currently works only in preview mode. First position a reference cube
so an edge is parallel to the actual open jaw closing axis, re-select it, press Y,
and type ALIGNED. The robot stays at its overview pose while the reference is
recorded. Do not guess jaw alignment from a raw image angle. Reference is saved
to ignored dice_grip_orientation.json and bound to the mapping and gripper.
Later chosen cube edges are mapped through the board/image homography into robot
XY. The inferred jaw axis is aligned by the nearest square-equivalent world-Z
rotation (at most 45 degrees), preserving roll/pitch and XYZ. The rotation is
performed at upper approach height before any inspection descent. Glare-uncertain
outlines are blocked for rotation; changes during confirmation also block motion.
Full gripping with automatic rotation is not enabled yet: the orientation model
and actual TCP/grasp-centre offset must be verified physically first, especially
with a tilted tool. This feature does not claim to compensate that offset.

In general pickup mode, V refreshes the session camera view without restarting
the application. It reuses the full pose, hardware and fresh-video checks and
requires BOARD FIXED confirmation; no movement or calibration-file changes occur.
Do not confirm if the physical board, base or camera/tool mounting moved. A bad
return pose still blocks refresh. The original mapping remains unchanged.
Glare recovery also handles a small extra corner from a shadow and can refine
a flagged centre from one dominant connected bright top-face component. These
remain GLARE estimates, blocked for full gripping and auto rotation.

Explicit BOARD FIXED area-view teaching accepts up to 10 degrees of camera
overview rotation while retaining the 2 mm XYZ limit, so the current view can
be registered through M1-M4. It does not send motion. All four markers, live
frames, unchanged physical setup confirmation and hardware checks are required.
The robot must stay still during capture. Only the session camera overview is
updated; grip/drop orientation, the canonical mapping and saved files remain
unchanged. Normal movement angular tolerances are not widened.

--preview-speed 50 sets upper preview approach speed. Lower inspection, rise and return use 20%.
Allowed upper speeds are 1-50 percent; default is 10. Inspection descent and
the rise from inspection remain at 10 percent. Speed returns to 10 after a
successful cycle; a failed cycle can leave the last configured limit active.
This option does not affect full pick/place or gripper speed.

In PREVIEW only, an endpoint feedback timeout offers WAIT to repeat the
read-only pose check for another 10 seconds. No motion command is resent and
no tolerance is widened. Any other input stops at the current pose. Hardware
faults stop immediately instead of offering WAIT; motion command errors are
not retried. Full pick/place still stops on an unconfirmed endpoint.


### Live manual inspection alignment

Add `--live-align` to `--enable-motion --preview-pick --preview-clearance-mm 30`
(with the usual area/reference options). This is operator-guided XY alignment,
not automatic visual servoing. The selected die may be hidden by the fingers;
there is no calibrated image target for the jaw centre yet. No additional descent
or gripper action is enabled.

At the lower inspection height, a live camera window replaces the terminal pause.
Tap **A/D** for robot **X -/+**, **S/W** for robot **Y -/+**, 5 mm per tap by default; keys **1/2/3** select **1/2/5 mm**.
Robot axes do not correspond to camera left/right. Check the real fingers and
release the key between taps. Nudges use 20% speed and preserve commanded Z/RPY. There is no cumulative XY
displacement limit; each movement still requires a separate key tap and valid
camera, pose and hardware checks.
Enter saves `experiments/vision/dice_alignment_measurement.json`, rises vertically
at the corrected XY, and returns to overview. This measurement does not update
pickup mapping, apply a correction on the next run, or authorize gripping. Q/Esc, closing the window, camera failure,
or a hardware/motion error stops in place without an automatic recovery move.
The normal pose feedback limits still apply.

Preview upper approach now defaults to 50%; descent, manual alignment, vertical
rise and return to overview use 20%. After a completed preview, the arm speed
setting is reset to 10%. If manual alignment pose feedback times out, terminal
command WAIT rechecks feedback only; it never resends the movement. Other input
stops in place, as do hardware or trajectory errors.


### Learn automatic preview compensation

This is a learned correction of repeatable position/yaw error, not closed-loop
camera tracking, and does not enable descent to grip height or gripping. Existing
single `dice_alignment_measurement.json` files are not imported as training data.

Start from the camera overview with the empty gripper and a fixed physical setup:

```powershell
.\.venv\Scripts\python.exe .\experiments\vision\dice_object_detection-rectified.py --enable-motion --preview-pick --refresh-area-view --board-area --preview-clearance-mm 30 --auto-yaw --live-align --teach-alignment --preview-speed 50
```

Confirm BOARD FIXED, select a stable non-glare die, P, PREVIEW. At inspection use
A/D (robot X), S/W (robot Y), keys 1/2/3 (1/2/5 mm), J/L (yaw -/+ 2 degrees).
Keep the die stationary during an example. Enter explicitly confirms that the
jaw midpoint is centred above the die and the jaw closing axis is parallel to a
die edge; it stores the example and rises vertically with the corrected yaw before
returning. Q/Esc stops without recording or returning. No gripper commands.
If pose feedback times out, WAIT in the terminal only rechecks the pose.

Collect at least eight examples spread around the working area and with at least
30 degrees of rotation spread. Use outside positions to enclose the area to be
used later. Examples closer than 10 mm with rotation difference below 10 degrees
are rejected as near-duplicates. Saved examples accumulate in
`experiments/vision/dice_alignment_samples.json`, bound to the camera intrinsics,
original mapping, board reference, tool and jaw reference. Do not move the board,
base, tool/camera mounting or change cube height. A physical change requires new
teaching; software binding cannot detect an unchanged-looking physical change.

The model fits world XY offset plus a yaw-dependent rotating XY offset and a
constant yaw correction. Leave-one-out prediction must be within 2 mm and 2 deg
on every example, with adequate position and angle coverage. These limits test
consistency of operator labels, not measured physical accuracy. Outliers are
rejected rather than silently discarded. Poor fit may require correcting the
original mapping or camera/tool calibration instead of adding samples.

After ALIGNMENT READY, quit and replace --teach-alignment with --use-alignment.
On new cube positions/angles, the correction is applied automatically to the
confirmed preview route. Keep --live-align to inspect it and manually stop or
adjust. It cannot extrapolate beyond the sampled convex position hull or angle
range. The corrected command must stay on the board, within 20 mm XY and 15 deg
of its original prediction. Inspection height remains 30 mm above taught grip Z.
Validate on new positions before considering a later gripping implementation.
Changing the jaw reference with Y invalidates the learned compensation.


Fine yaw measurement is also available in ordinary --live-align preview without
--teach-alignment. J/L turn yaw -/+; keys 1/2/3 select yaw steps 0.5/1/2 degrees
and XY steps 1/2/5 mm. Z, roll and pitch stay fixed. Maximum yaw correction remains
15 degrees. Enter saves delta_yaw_deg as well as delta_xy_mm in the measurement,
then rises with the corrected orientation before returning. Without teaching,
this measurement does not append to or change the learned model or jaw reference.


Every confirmed live inspection now appends to `dice_alignment_history.json`
before updating `dice_alignment_measurement.json`. The previously saved latest
measurement is imported on the first history write. Older overwritten latest-only
measurements cannot be recovered by this change. History collection does not
train or enable a correction model: only --teach-alignment appends bound examples
to dice_alignment_samples.json. Existing inconsistent training samples are kept
and remain blocked by validation; no limits are relaxed or samples discarded.


In --repair-pick, physically re-recorded heights may be staged one target at a
time when switching from inspection height to grip height. Mixed heights set
height_repair_pending and block both general calibration validation and the
fixed-target motion shortcut. All five measured heights must agree within 2 mm
before CHECK; orientation checks still apply on every RECORD. No XYZ values are
filled in automatically, and the old validated calibration is not replaced until
CHECK passes. The repair workflow backs up the original draft on entry.


Repair RECORD now waits up to 10 seconds for two consecutive stationary pose
readings before saving. It retains the measured six values and the original
1.15-degree orientation / 2-mm position limits. A persistent mismatch, movement,
Learning Mode, or hardware error rejects the record and preserves the old one.
Confirm success by both `Record N saved` and the updated XYZ shown in the menu;
typing RECORD alone is not a successful save. This wait sends no motion commands.
