RadarFusionV3 (separate project)

Arduino:
- Open arduino/radar_fusion_follow.ino
- Upload to your Arduino
- Serial: 9600 baud
- Output stays: angle,distance

Python:
- cd python
- pip install -r requirements.txt
- python fusion_follow.py --port COM4 --baud 9600

Controls:
- F : toggle FOLLOW (YOLO + fused TARGET:<angle>)
- S : back to SWEEP
- SPACE : pause/resume (STOP/GO)
- ESC : quit

Camera debug window:
- add --show-camera (press 'q' inside that window to close it)
