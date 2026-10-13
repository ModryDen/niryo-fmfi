# Triedenie kociek

Robotické rameno Niryo NED sa pripája k notebooku cez ethernetový kábel. Jeho IP adresa je `169.254.200.200`.

## Napájanie a zapojenie

V FT Labe máme originálnu, najstaršiu verziu robota Niryo NED, ktorá je riadená doskou Raspberry Pi 3.

**Pozor:** Veľa manuálov na webe už predpokladá Niryo NED 2 alebo Niryo NED 3, takže pokyny z nich nemusia fungovať.

Dokumentácia: [Niryo NED](https://docs.niryo.com/robots/ned/).

Na napájanie používame nastaviteľný zdroj, ktorý treba nastaviť na **11,1 V**.

Niryo má na ramene namontovanú kameru. Je zakrytá gumenou čiapočkou, ktorú treba pred použitím zložiť a po použití opäť nasadiť. Kamera sa dá zaostriť ručne otáčaním.

Po zapnutí alebo reštarte sa rozsvieti červená LED, po chvíli zelená LED a potom modrá LED. Vtedy je Niryo pripravené na činnosť.

## Verzie softvéru a pripojenie

Výrobca poskytuje nástroj Niryo Studio. Potrebujete **NiryoStudio 1, verziu 4.1.2**, ktorú nájdete v [centre sťahovania](https://niryo.com/resources/download-center/).

V pravom hornom rohu vyberte spôsob pripojenia **Ethernet**.

## Kalibrácia

Po každom zapnutí robota je potrebné vykonať kalibráciu. V Niryo Studio nájdete tlačidlo **Request calibration**.

Na ovládanie robota v Pythone vo Windows nainštalujte knižnicu:

```bash
pip install pyniryo==1.1.2
```

**Pozor na správnu verziu knižnice.**

Robota skalibrujte nasledujúcim kódom:

```python
from pyniryo import *

r = NiryoRobot("169.254.200.200")
r.calibrate_auto()
r.close_connection()
```

## Uloženie snímky z kamery

Uložte jednu snímku z kamery:

```python
from pyniryo import NiryoRobot
from pyniryo.vision import uncompress_image, show_img_and_wait_close
import cv2

ROBOT_IP = "169.254.200.200"  # IP adresa robota
robot = NiryoRobot(ROBOT_IP)

# Získanie jednej snímky
img_comp = robot.get_img_compressed()
img = uncompress_image(img_comp)
cv2.imwrite("ned_frame.png", img[:, :, ::-1])  # Prípadný prevod na BGR pre OpenCV
show_img_and_wait_close("Ned camera", img)
```

## Úlohy

1. Pomocou kamery robota vytvorte niekoľko obrázkov kociek.
2. Porozmýšľajte, ako by sa dala z obrazu vybrať oblasť, v ktorej sa nachádzajú kocky.
3. Navrhnite, ako rozlíšiť jednotlivé kocky.
4. Navrhnite, ako určiť ich čísla.

## Nápoveda

1. Predpokladajúc, že kameru držíte tak, aby sa dívala zhora, zdetekujete kocku (to už viete), dostanete jej hranicu ako sekvenciu bodov.
2. Z nich viete nájsť ohraničujúci rotovaný obdĺžnik (na to je veľmi efektívny algoritmus `cv2.minAreaRect`).
3. Potom viete tento zrotovaný obdĺžnik vyrovnať, viď https://www.agentspace.org/opencv/ – kapitola 6, časť „Image alignment“.
4. A prípadne rovno projektovať do nejakého Vami zvoleného rozlíšenia, napr. 256 × 256.
5. Toto treba binarizovať (bude asi stačiť Otsu thresholding, ktorý v OpenCV je; keby nie, mám aj lepšie nápovedy, len potrebujem obrázok, na ktorom to vyskúšať).
6. A potom už len overiť voči binárnej matici (a takej istej otočenej o 90 stupňov).

Na všetky tieto úkony stačí knižnica OpenCV.

## Resources

- [Nyrio examples](https://agentspace.org/download/nyrio-examples.zip)
