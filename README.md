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
V PowerShelli spusti skript bez prepínača --calibrate:

.\.venv\Scripts\python.exe .\experiments\vision\dice_object_detection-rectified.py

Zobrazí sa obraz kamery a okná s detekovanými kockami. Ak chýba kalibračný súbor, detekcia môže bežať, ale pohyb robota zostane zablokovaný.

7. Ovládanie
Klikaj do hlavného okna Dice Detection:

Klik na detegovanú kocku — vyberie ju.
M — požiada o priblíženie k vybranej kocke. Skontroluj naplánované polohy a celú dráhu ramena. V termináli napíš MOVE iba vtedy, ak je pohyb bezpečný; čokoľvek iné pohyb zruší.
B — z detailného pohľadu požiada o návrat: najprv nahor a potom do overview polohy. Aj tento pohyb treba potvrdiť napísaním MOVE v termináli.
Q — ukončí program, ale robota nevráti. Ak je robot v detailnej polohe, vráť ho bezpečne ručne.
R — resetuje referenčný obrázok na zarovnanie.
S — uloží binárne obrázky detegovaných kociek do priečinka dice_dataset v aktuálnom pracovnom priečinku.
Skript nastavuje rýchlosť ramena na 10 % a po skončení ju neobnovuje. Priblíženie je určené na kontrolu obrazu, nie na uchopenie ani triedenie kociek. Tento postup nepredstavuje detekciu kolízií — bezpečnosť celej dráhy musíš vždy overiť ty.