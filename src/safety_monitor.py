"""
=============================================================================
AI-Based Industrial Safety Monitoring System Using Computer Vision
=============================================================================
Author      : Final Year B.Tech Project
Framework   : YOLOv8 + OpenCV + PyTorch
Description : Detects workers entering danger zones near industrial machines
              and triggers alerts in real time.
=============================================================================
"""

import cv2
import numpy as np
import time
import os
import sys
import datetime
import threading
import pandas as pd
from pathlib import Path

# ─── Try importing ultralytics (YOLOv8) ────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed. Running in DEMO mode.")
    print("          Install with: pip install ultralytics")

# ─── Optional: sound alert ──────────────────────────────────────────────────
try:
    import winsound  # Windows only
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False


# =============================================================================
#  CONFIGURATION — Edit these values to match your setup
# =============================================================================
class Config:
    # ── Model
    MODEL_PATH = "yolov8n.pt"          # YOLOv8 nano (auto-downloaded)
    CONFIDENCE_THRESHOLD = 0.40        # Minimum detection confidence
    PERSON_CLASS_ID = 0                # COCO class ID for 'person'

    # ── Danger Zone (x1, y1, x2, y2) in pixels — adjust per camera view
    # These coordinates define a rectangle around the dangerous machine area.
    DANGER_ZONE = (150, 200, 550, 480)

    # ── Alert
    ALERT_COOLDOWN_SECONDS = 3         # Seconds between repeated alerts
    LOG_FILE = "outputs/alert_log.csv" # CSV log for all events

    # ── Display
    FRAME_WIDTH = 720
    FRAME_HEIGHT = 480
    SHOW_FPS = True

    # ── Colors (BGR format for OpenCV)
    COLOR_DANGER_ZONE_SAFE    = (0, 200, 0)    # Green  – no intrusion
    COLOR_DANGER_ZONE_ALERT   = (0, 0, 255)    # Red    – intrusion detected
    COLOR_BBOX_PERSON         = (255, 165, 0)  # Orange – person bounding box
    COLOR_BBOX_DANGER_PERSON  = (0, 0, 255)    # Red    – person in danger zone
    COLOR_TEXT_WARNING        = (0, 0, 255)    # Red warning text
    COLOR_TEXT_INFO           = (255, 255, 255)# White info text


# =============================================================================
#  ALERT MANAGER
# =============================================================================
class AlertManager:
    """Handles all alert actions when a worker enters the danger zone."""

    def __init__(self, log_file: str):
        self.last_alert_time = 0
        self.alert_count = 0
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Initialize log CSV
        if not os.path.exists(log_file):
            df = pd.DataFrame(columns=["timestamp", "event", "persons_in_zone"])
            df.to_csv(log_file, index=False)

    def trigger_alert(self, persons_in_zone: int):
        """Fire all alert actions."""
        now = time.time()
        if now - self.last_alert_time < Config.ALERT_COOLDOWN_SECONDS:
            return  # Cooldown active, skip

        self.last_alert_time = now
        self.alert_count += 1
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── 1. Console alert to CCTV Operator
        print("\n" + "="*60)
        print(f"  ⚠  ALERT #{self.alert_count} — {timestamp}")
        print(f"  🚨 DANGER ZONE ENTERED!")
        print(f"  👷 Persons detected in zone: {persons_in_zone}")
        print(f"  📡 Alert sent to CCTV Monitoring Operator")
        print(f"  🔔 Alarm triggered!")
        print(f"  🛑 Machine Stop Signal ACTIVATED (Simulated)")
        print("="*60 + "\n")

        # ── 2. Sound alarm (Windows)
        self._sound_alarm()

        # ── 3. Log to CSV
        self._log_event(timestamp, "DANGER_ZONE_ENTERED", persons_in_zone)

        # ── 4. Simulate machine stop command
        self._simulate_machine_stop()

    def _sound_alarm(self):
        """Beep alarm on Windows; print on Linux/Mac."""
        if SOUND_AVAILABLE:
            threading.Thread(
                target=lambda: winsound.Beep(1000, 600), daemon=True
            ).start()
        else:
            print("\a")  # Terminal bell

    def _log_event(self, timestamp: str, event: str, persons: int):
        """Append event to the CSV log."""
        try:
            row = pd.DataFrame([[timestamp, event, persons]],
                               columns=["timestamp", "event", "persons_in_zone"])
            row.to_csv(self.log_file, mode="a", header=False, index=False)
        except Exception as e:
            print(f"[LOG ERROR] {e}")

    def _simulate_machine_stop(self):
        """Simulate sending a stop command to the machine controller."""
        # In a real system, this would send a signal via GPIO, MQTT, or serial port.
        print(f"  [MACHINE CTRL] >>> STOP command sent to PLC/Controller <<<")


# =============================================================================
#  DANGER ZONE CHECKER
# =============================================================================
def is_in_danger_zone(bbox: tuple, zone: tuple) -> bool:
    """
    Check if a detected person's bounding box overlaps with the danger zone.

    Args:
        bbox : (x1, y1, x2, y2) — person bounding box
        zone : (zx1, zy1, zx2, zy2) — danger zone rectangle
    Returns:
        True if there is any overlap.
    """
    px1, py1, px2, py2 = bbox
    zx1, zy1, zx2, zy2 = zone

    # Person's center point
    cx = (px1 + px2) // 2
    cy = (py1 + py2) // 2

    # Check if center is inside the zone
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


# =============================================================================
#  FRAME ANNOTATOR
# =============================================================================
def annotate_frame(frame: np.ndarray,
                   detections: list,
                   danger_zone: tuple,
                   alert_active: bool,
                   alert_manager: AlertManager,
                   fps: float) -> np.ndarray:
    """
    Draw bounding boxes, danger zone, and alert overlays on a frame.

    Args:
        frame       : Current video frame (BGR)
        detections  : List of (x1,y1,x2,y2,confidence) for each detected person
        danger_zone : (x1,y1,x2,y2) rectangle
        alert_active: Whether an alert is currently active
        alert_manager: For displaying alert count
        fps         : Current frames per second
    Returns:
        Annotated frame
    """
    zx1, zy1, zx2, zy2 = danger_zone
    overlay = frame.copy()

    # ── Draw Danger Zone rectangle
    zone_color = Config.COLOR_DANGER_ZONE_ALERT if alert_active else Config.COLOR_DANGER_ZONE_SAFE
    zone_thickness = 3 if alert_active else 2

    # Semi-transparent fill
    alpha = 0.25 if alert_active else 0.10
    cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), zone_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Border
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), zone_color, zone_thickness)

    # Label
    cv2.putText(frame, "DANGER ZONE", (zx1 + 5, zy1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, zone_color, 2)

    # ── Draw person bounding boxes
    persons_in_zone = 0
    for (x1, y1, x2, y2, conf) in detections:
        in_zone = is_in_danger_zone((x1, y1, x2, y2), danger_zone)
        color = Config.COLOR_BBOX_DANGER_PERSON if in_zone else Config.COLOR_BBOX_PERSON
        if in_zone:
            persons_in_zone += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"Person {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── HUD: top-left status panel
    _draw_hud(frame, len(detections), persons_in_zone, alert_active,
              alert_manager.alert_count, fps)

    # ── Big WARNING banner when alert is active
    if alert_active:
        _draw_warning_banner(frame)

    return frame


def _draw_hud(frame, total_persons, persons_in_zone, alert_active, alert_count, fps):
    """Draw status HUD in top-left corner."""
    h, w = frame.shape[:2]
    panel_h = 120
    panel_w = 320
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    lines = [
        f"System: AI Industrial Safety Monitor",
        f"Persons Detected : {total_persons}",
        f"In Danger Zone   : {persons_in_zone}",
        f"Alert Status     : {'⚠ ACTIVE' if alert_active else 'SAFE'}",
        f"Total Alerts     : {alert_count}",
    ]
    if Config.SHOW_FPS:
        lines.append(f"FPS              : {fps:.1f}")

    for i, line in enumerate(lines):
        color = (0, 0, 255) if (i == 3 and alert_active) else (200, 255, 200)
        cv2.putText(frame, line, (10, 18 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def _draw_warning_banner(frame):
    """Draw flashing red warning banner at the bottom."""
    h, w = frame.shape[:2]
    banner_h = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    warnings = [
        "⚠  DANGER ZONE ENTERED  ⚠",
        "Alert Sent  |  Machine Stop Signal Activated",
    ]
    for i, text in enumerate(warnings):
        cv2.putText(frame, text, (20, h - banner_h + 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65 if i == 0 else 0.5,
                    (255, 255, 255), 2 if i == 0 else 1)


# =============================================================================
#  DEMO MODE (when YOLO is not installed)
# =============================================================================
def demo_mode():
    """
    Demo visualization without a real camera or YOLO model.
    Simulates a person walking into the danger zone.
    """
    print("\n[DEMO MODE] Simulating worker detection and danger zone alert...")
    print("Press 'q' to quit.\n")

    alert_manager = AlertManager(Config.LOG_FILE)
    frame_count = 0
    fps = 20.0
    prev_time = time.time()

    # Simulate a person walking across the frame
    person_x = 50

    while True:
        frame = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)

        # Simple industrial background
        cv2.rectangle(frame, (0, 0), (Config.FRAME_WIDTH, Config.FRAME_HEIGHT),
                      (40, 40, 50), -1)
        # Machine silhouette
        cv2.rectangle(frame, (200, 150), (500, 420), (80, 80, 90), -1)
        cv2.rectangle(frame, (200, 150), (500, 420), (120, 120, 130), 2)
        cv2.putText(frame, "INDUSTRIAL MACHINE", (220, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 170), 1)
        # Conveyor belt lines
        for y in range(300, 420, 20):
            cv2.line(frame, (210, y), (490, y), (100, 100, 110), 1)

        # Moving person (simulated bounding box)
        px1, py1 = person_x, 280
        px2, py2 = person_x + 60, 430
        detection = [(px1, py1, px2, py2, 0.92)]

        alert_active = is_in_danger_zone((px1, py1, px2, py2), Config.DANGER_ZONE)
        if alert_active:
            alert_manager.trigger_alert(persons_in_zone=1)

        # FPS
        cur_time = time.time()
        fps = 0.9 * fps + 0.1 / max(cur_time - prev_time, 1e-6)
        prev_time = cur_time

        annotated = annotate_frame(frame.copy(), detection, Config.DANGER_ZONE,
                                   alert_active, alert_manager, fps)

        cv2.imshow("AI Industrial Safety Monitor [DEMO]", annotated)

        person_x = (person_x + 3) % (Config.FRAME_WIDTH - 80)

        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break

        frame_count += 1

    cv2.destroyAllWindows()
    print(f"\n[DEMO] Finished. {alert_manager.alert_count} alerts generated.")
    print(f"[LOG] Alert log saved to: {Config.LOG_FILE}")


# =============================================================================
#  MAIN MONITORING LOOP
# =============================================================================
def run_monitor(source=0):
    """
    Main real-time safety monitoring loop.

    Args:
        source : Camera index (0 = default webcam) or path to video file
                 or YouTube stream URL
    """
    if not YOLO_AVAILABLE:
        print("[INFO] YOLO not available. Starting demo mode instead.")
        demo_mode()
        return

    print(f"\n[INFO] Loading YOLOv8 model: {Config.MODEL_PATH}")
    model = YOLO(Config.MODEL_PATH)
    print(f"[INFO] Model loaded. Opening video source: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    # Optionally resize
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

    alert_manager = AlertManager(Config.LOG_FILE)
    fps = 0.0
    prev_time = time.time()
    frame_num = 0

    print(f"[INFO] Monitoring started. Press 'q' to quit, 'z' to reconfigure zone.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break

        frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
        frame_num += 1

        # ── Run YOLO inference
        results = model(frame, conf=Config.CONFIDENCE_THRESHOLD,
                        classes=[Config.PERSON_CLASS_ID], verbose=False)

        # ── Parse detections
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))

        # ── Check danger zone intrusions
        persons_in_zone = sum(
            1 for (x1, y1, x2, y2, _) in detections
            if is_in_danger_zone((x1, y1, x2, y2), Config.DANGER_ZONE)
        )
        alert_active = persons_in_zone > 0
        if alert_active:
            alert_manager.trigger_alert(persons_in_zone)

        # ── FPS calculation
        cur_time = time.time()
        fps = 0.9 * fps + 0.1 / max(cur_time - prev_time, 1e-6)
        prev_time = cur_time

        # ── Annotate and display
        annotated = annotate_frame(frame, detections, Config.DANGER_ZONE,
                                   alert_active, alert_manager, fps)
        cv2.imshow("AI Industrial Safety Monitor", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Session complete.")
    print(f"  Total frames processed : {frame_num}")
    print(f"  Total alerts triggered : {alert_manager.alert_count}")
    print(f"  Alert log saved to     : {Config.LOG_FILE}")


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Industrial Safety Monitoring System"
    )
    parser.add_argument(
        "--source", default="0",
        help="Video source: 0=webcam, path to video file, or stream URL"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run in demo mode (no camera or YOLO required)"
    )
    args = parser.parse_args()

    if args.demo:
        demo_mode()
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        run_monitor(source)
