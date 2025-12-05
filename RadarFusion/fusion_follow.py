import argparse
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import pygame
import serial
from ultralytics import YOLO


@dataclass
class RadarReading:
    angle_deg: float = 0.0
    dist_cm: float = -1.0
    t: float = 0.0


@dataclass
class VisionReading:
    has_person: bool = False
    conf: float = 0.0
    offset_deg: float = 0.0   # camera offset (deg, right = + unless inverted)
    t: float = 0.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))  # clamp to [lo, hi]


def now_s() -> float:
    return time.time()  # unix seconds


class SerialWorker(threading.Thread):
    """Reads ONLY lines formatted exactly as: angle,distance"""
    def __init__(self, ser: serial.Serial):
        super().__init__(daemon=True)
        self.ser = ser
        self.lock = threading.Lock()  # guard latest
        self.latest = RadarReading()
        self.running = True

    def run(self):
        while self.running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()  # read serial line
                if not line or "," not in line:
                    continue

                a_str, d_str = line.split(",", 1)  # angle,dist
                a_str = a_str.strip()
                d_str = d_str.strip()
                if not a_str or not d_str:
                    continue

                a = float(a_str)
                d = float(d_str)

                with self.lock:
                    self.latest = RadarReading(angle_deg=a, dist_cm=d, t=now_s())  # publish latest
            except Exception:
                continue

    def get_latest(self) -> RadarReading:
        with self.lock:
            return self.latest  # thread-safe snapshot

    def stop(self):
        self.running = False


class YoloWorker(threading.Thread):
    """
    Tracks the best 'person' and converts x-offset to degrees via camera horizontal FOV.
    offset_deg is positive if the person center is to the RIGHT of the image center
    (unless invert is toggled in main).
    """
    def __init__(
        self,
        cam_index: int,
        model_path: str,
        fov_deg: float,
        conf_thresh: float,
        show_window: bool,
        draw_box: bool,
        img_w_fix: int = 0,
        img_h_fix: int = 0,
    ):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.model_path = model_path
        self.fov_deg = float(fov_deg)
        self.conf_thresh = float(conf_thresh)
        self.show_window = bool(show_window)
        self.draw_box = bool(draw_box)
        self.img_w_fix = int(img_w_fix)
        self.img_h_fix = int(img_h_fix)

        self.lock = threading.Lock()  # guard latest
        self.latest = VisionReading()
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            print(f"[YOLO] Could not open camera index {self.cam_index}")
            self.running = False
            return

        if self.img_w_fix > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_w_fix)  # optional force width
        if self.img_h_fix > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_h_fix)  # optional force height

        model = YOLO(self.model_path)

        names = getattr(model, "names", {})  # class names map
        person_id = None
        try:
            for k, v in names.items():
                if str(v).lower() == "person":
                    person_id = int(k)  # "person" class id
                    break
        except Exception:
            person_id = None

        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            cx = w / 2.0  # image center x

            res = model.predict(frame, verbose=False, conf=self.conf_thresh)
            best = None  # (score, conf, x1,y1,x2,y2)

            try:
                if res and len(res) > 0 and hasattr(res[0], "boxes") and res[0].boxes is not None:
                    for b in res[0].boxes:
                        cls = int(b.cls[0].item())
                        conf = float(b.conf[0].item())
                        if conf < self.conf_thresh:
                            continue
                        if person_id is not None and cls != person_id:
                            continue
                        x1, y1, x2, y2 = b.xyxy[0].tolist()

                        area = max(1.0, (x2 - x1) * (y2 - y1))  # bbox area
                        score = conf + 0.000001 * area  # conf first, area tie-break
                        if best is None or score > best[0]:
                            best = (score, conf, x1, y1, x2, y2)
            except Exception:
                best = None

            vr = VisionReading(has_person=False, conf=0.0, offset_deg=0.0, t=now_s())

            if best is not None:
                _, conf, x1, y1, x2, y2 = best
                bx = (x1 + x2) / 2.0  # bbox center x

                norm = (bx - cx) / cx          # -1..+1
                offset = norm * (self.fov_deg / 2.0)  # pixel offset -> degrees

                vr = VisionReading(has_person=True, conf=conf, offset_deg=float(offset), t=now_s())

                if self.show_window:
                    if self.draw_box:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"PERSON conf={conf:.2f} off={offset:+.1f}deg",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

            if self.show_window:
                cv2.line(frame, (int(cx), 0), (int(cx), h), (255, 255, 255), 1)  # center line
                cv2.imshow("YOLO Camera", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    self.running = False

            with self.lock:
                self.latest = vr  # publish latest

        cap.release()
        if self.show_window:
            cv2.destroyAllWindows()

    def get_latest(self) -> VisionReading:
        with self.lock:
            return self.latest  # thread-safe snapshot

    def stop(self):
        self.running = False


class RadarViz:
    def __init__(self, w=900, h=900, max_range_cm=300):
        pygame.init()
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Radar + YOLO Fusion (Follow Mode)")
        self.clock = pygame.time.Clock()

        self.center = (w // 2, h // 2)
        self.max_range_cm = max(1, int(max_range_cm))  # clamp max range
        self.radius_px = min(w, h) // 2 - 55  # usable radius

        self.font = pygame.font.SysFont(None, 22)
        self.big = pygame.font.SysFont(None, 40)

        self.dots = []  # (x,y,t)
        self.fade_s = 2.0  # dot fade time

    def polar_to_xy(self, angle_deg: float, dist_cm: float) -> Tuple[int, int]:
        rad = math.radians(angle_deg)  # deg -> rad
        r = (dist_cm / self.max_range_cm) * self.radius_px  # cm -> px
        x = self.center[0] + r * math.cos(rad)
        y = self.center[1] - r * math.sin(rad)
        return int(x), int(y)

    def draw_grid(self):
        self.screen.fill((0, 0, 0))
        green = (0, 255, 0)

        for frac in [1/6, 2/6, 3/6, 4/6, 5/6, 1.0]:
            r = int(self.radius_px * frac)
            pygame.draw.circle(self.screen, green, self.center, r, 1)

        for a in range(0, 360, 30):
            rad = math.radians(a)
            x = self.center[0] + self.radius_px * math.cos(rad)
            y = self.center[1] - self.radius_px * math.sin(rad)
            pygame.draw.line(self.screen, green, self.center, (x, y), 1)

    def add_dot(self, angle: float, dist: float):
        if dist <= 0 or dist >= self.max_range_cm:
            return
        x, y = self.polar_to_xy(angle, dist)
        self.dots.append((x, y, now_s()))  # store dot w/ timestamp

    def draw_dots(self):
        tnow = now_s()
        keep = []
        for (x, y, t0) in self.dots:
            age = tnow - t0
            if age < self.fade_s:
                alpha = int(255 * (1 - age / self.fade_s))  # fade alpha
                s = pygame.Surface((10, 10), pygame.SRCALPHA)
                pygame.draw.circle(s, (255, 0, 0, alpha), (5, 5), 5)
                self.screen.blit(s, (x - 5, y - 5))
                keep.append((x, y, t0))
        self.dots = keep  # drop expired dots

    def draw_ray(self, angle_deg: float, color, thickness=3):
        x, y = self.polar_to_xy(angle_deg, self.max_range_cm)
        pygame.draw.line(self.screen, color, self.center, (x, y), thickness)

    def text(self, lines):
        y = 10
        for ln in lines:
            surf = self.font.render(ln, True, (255, 255, 255))
            self.screen.blit(surf, (10, y))
            y += 22

    def banner(self, msg: str, color=(255, 0, 0)):
        surf = self.big.render(msg, True, color)
        self.screen.blit(surf, (self.center[0] - 220, 15))

    def tick(self, fps=60):
        self.clock.tick(fps)  # frame cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--fov", type=float, default=60.0, help="camera horizontal FOV in degrees (approx)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--max-range", type=float, default=300.0)
    ap.add_argument("--show-camera", action="store_true", help="show OpenCV camera window")
    ap.add_argument("--draw-box", action="store_true", help="draw YOLO bounding box")
    ap.add_argument("--max-angle", type=float, default=180.0, help="cap servo targets to this max angle (e.g., 150)")
    ap.add_argument("--min-angle", type=float, default=0.0, help="cap servo targets to this min angle (e.g., 20)")
    ap.add_argument("--send-hz", type=float, default=6.0, help="max TARGET send rate (Hz)")
    ap.add_argument("--lock-on-first", action="store_true", help="once a person is found, keep follow enabled")
    ap.add_argument("--img-w", type=int, default=0, help="force camera width (optional)")
    ap.add_argument("--img-h", type=int, default=0, help="force camera height (optional)")
    args = ap.parse_args()

    min_ang = float(args.min_angle)
    max_ang = float(args.max_angle)

    ser = serial.Serial(args.port, args.baud, timeout=0.15)
    time.sleep(1.5)  # allow Arduino reset

    def send_cmd(cmd: str):
        try:
            ser.write((cmd.strip() + "\n").encode())  # send one line
        except Exception:
            pass

    send_cmd("SWEEP")  # start scanning

    sw = SerialWorker(ser)
    sw.start()

    yw = YoloWorker(
        cam_index=args.camera,
        model_path=args.model,
        fov_deg=args.fov,
        conf_thresh=args.conf,
        show_window=args.show_camera,
        draw_box=args.draw_box,
        img_w_fix=args.img_w,
        img_h_fix=args.img_h,
    )
    yw.start()

    viz = RadarViz(900, 900, int(args.max_range))

    follow_enabled = False
    paused = False

    invert = False           # flip left/right mapping
    bias_deg = 0.0           # constant correction offset
    kp = 0.65                # proportional gain
    alpha = 0.10             # smoothing factor
    deadband_deg = 8.0       # ignore tiny offsets
    min_delta_send = 6.0     # ignore tiny target changes

    smoothed_target: Optional[float] = None

    last_sent: Optional[int] = None
    last_send_t = 0.0

    lock_state = False
    last_person_seen_t = 0.0

    send_period = 1.0 / max(0.5, float(args.send_hz))  # rate limit period

    running = True
    while running:
        viz.tick(60)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False

                elif e.key == pygame.K_SPACE:
                    paused = not paused
                    send_cmd("STOP" if paused else "GO")  # pause/resume

                elif e.key == pygame.K_s:
                    follow_enabled = False
                    lock_state = False
                    smoothed_target = None
                    send_cmd("SWEEP")  # back to sweep

                elif e.key == pygame.K_f:
                    follow_enabled = not follow_enabled
                    if follow_enabled:
                        send_cmd("FOLLOW")  # follow mode
                    else:
                        lock_state = False
                        smoothed_target = None
                        send_cmd("SWEEP")  # sweep mode

                elif e.key == pygame.K_i:
                    invert = not invert  # toggle invert

                elif e.key == pygame.K_LEFTBRACKET:
                    bias_deg -= 1.0  # bias --

                elif e.key == pygame.K_RIGHTBRACKET:
                    bias_deg += 1.0  # bias ++

                elif e.key == pygame.K_MINUS:
                    kp = max(0.10, kp - 0.05)  # kp --

                elif e.key == pygame.K_EQUALS:
                    kp = min(2.00, kp + 0.05)  # kp ++

                elif e.key == pygame.K_COMMA:
                    deadband_deg = max(0.0, deadband_deg - 0.5)  # deadband --

                elif e.key == pygame.K_PERIOD:
                    deadband_deg = min(30.0, deadband_deg + 0.5)  # deadband ++

                elif e.key == pygame.K_b:
                    vr = yw.get_latest()
                    if vr.has_person and (now_s() - vr.t) < 0.5:
                        off = vr.offset_deg
                        if invert:
                            off = -off
                        bias_deg += -off  # cancel current offset

        radar = sw.get_latest()
        vision = yw.get_latest()

        if vision.has_person and (now_s() - vision.t) < 0.5:
            last_person_seen_t = now_s()  # update seen time

        if 0 < radar.dist_cm < args.max_range:
            viz.add_dot(radar.angle_deg, radar.dist_cm)  # add radar dot

        fused_target: Optional[float] = None
        tnow = now_s()

        person_fresh = vision.has_person and (tnow - vision.t) < 0.35  # recent vision
        radar_fresh = (tnow - radar.t) < 0.60  # recent radar

        if args.lock_on_first:
            if person_fresh:
                lock_state = True  # latch once seen
            if lock_state and not follow_enabled and not paused:
                follow_enabled = True
                send_cmd("FOLLOW")  # auto-enter follow

        if follow_enabled and not paused and person_fresh and radar_fresh:
            cam_off = vision.offset_deg
            if invert:
                cam_off = -cam_off  # flip direction

            desired = radar.angle_deg + (kp * cam_off) + bias_deg  # fused target
            desired = clamp(desired, min_ang, max_ang)  # enforce servo limits

            if smoothed_target is None:
                smoothed_target = desired  # init filter
            else:
                smoothed_target = (1 - alpha) * smoothed_target + alpha * desired  # smooth

            fused_target = smoothed_target

            if (tnow - last_send_t) >= send_period:
                ang_int = int(round(fused_target))
                if last_sent is None:
                    do_send = True
                else:
                    do_send = abs(ang_int - last_sent) >= int(round(min_delta_send))  # min delta

                if abs(cam_off) <= deadband_deg and last_sent is not None:
                    do_send = abs(ang_int - last_sent) >= 1  # allow small settle

                if do_send:
                    send_cmd(f"TARGET:{ang_int}")  # send new target
                    last_sent = ang_int
                    last_send_t = tnow

        viz.draw_grid()
        viz.draw_dots()

        viz.draw_ray(radar.angle_deg, (0, 255, 0), 3)

        if fused_target is not None:
            viz.draw_ray(fused_target, (80, 160, 255), 3)

        seen_ms = int((tnow - last_person_seen_t) * 1000.0)
        locked = follow_enabled and person_fresh and not paused

        mode_str = "FOLLOW" if follow_enabled else "SWEEP"
        viz.text([
            f"PORT {args.port} @ {args.baud} (expects angle,distance)   invert={invert} (press I)  bias={bias_deg:+.1f}°",
            f"STATE: {mode_str}   FOLLOW_ENABLED: {follow_enabled}   PAUSED: {paused}",
            f"LOCKED: {locked} seen={seen_ms} ms",
            f"RADAR: angle={radar.angle_deg:.1f}° dist={radar.dist_cm:.0f}cm",
            f"YOLO: person={vision.has_person} conf={vision.conf:.2f} off={vision.offset_deg:+.1f}°",
            f"CTRL: deadband={deadband_deg:.1f}°  alpha={alpha:.2f}  kp={kp:.2f}  send={1.0/send_period:.1f}Hz  minΔ={min_delta_send:.1f}°",
            "Keys: [F]=auto-follow  [S]=sweep  [SPACE]=pause  [I]=invert  [B]=calibrate bias  [[/]]=bias  [-/=]=kp  [,/.]=deadband  [ESC]=quit"
        ])

        if locked:
            viz.banner("FOLLOWING (LOCKED)", (0, 255, 0))
            viz.banner("PERSON DETECTED", (0, 255, 0))
        elif person_fresh:
            viz.banner("PERSON DETECTED", (0, 255, 0))
        elif follow_enabled and not paused:
            viz.banner("SEARCHING...", (255, 255, 0))
        elif paused:
            viz.banner("PAUSED", (255, 0, 0))

        pygame.display.update()

    yw.stop()
    sw.stop()
    try:
        ser.close()
    except Exception:
        pass
    pygame.quit()


if __name__ == "__main__":
    main()
