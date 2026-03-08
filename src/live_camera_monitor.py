"""
=============================================================================
  AI-Based Industrial Safety Monitoring System
  TWO MODES:
    Mode 1 → LIVE CAMERA  (point webcam at another screen)
    Mode 2 → VIDEO FILE   (upload/provide any video file)
=============================================================================
  CHANGES MADE (Bug Fixes):
  ─────────────────────────────────────────────
  FIX 1 → Black border auto-crop
           Video had black borders around it. System now automatically
           detects the actual content area and crops to it before
           running detection. This ensures the danger zone covers
           the real machine area, not the black empty border.

  FIX 2 → Sensitivity lowered from 0.25 → 0.15
           Workers standing close to the press machine are partially
           hidden (occluded) by the machine body. Lower threshold
           allows detection of partially visible persons.

  FIX 3 → Detection every 2nd frame instead of every 3rd frame
           With finer stride (4,4) instead of (8,8) for better
           accuracy on persons who are close to machines.
  ─────────────────────────────────────────────

  HOW TO RUN:
      python src/live_camera_monitor.py

  KEYBOARD CONTROLS:
  ─────────────────────────────────────────────
  Z     → Draw danger zone (click & drag on window)
  S     → Save screenshot
  SPACE → Pause / Resume
  R     → Reset danger zone to default
  Q     → Quit
=============================================================================
"""

import cv2
import numpy as np
import datetime
import os
import time
import argparse

# ─────────────────────────────────────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0       # 0 = built-in webcam. Change to 1 or 2 if wrong camera
WINDOW_W       = 960
WINDOW_H       = 540
ALERT_COOLDOWN = 25      # frames between repeated alerts

# FIX 2 → Lowered from 0.25 to 0.15 to detect partially occluded persons
SENSITIVITY    = 0.15

# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT FOLDERS
# ─────────────────────────────────────────────────────────────────────────────
OUT_ROOT    = "live_outputs"
SCREENSHOTS = f"{OUT_ROOT}/screenshots"
ALERT_SNAPS = f"{OUT_ROOT}/alert_snapshots"
LOG_FILE    = f"{OUT_ROOT}/alert_log.txt"
for d in [SCREENSHOTS, ALERT_SNAPS]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  COLORS (BGR)
# ─────────────────────────────────────────────────────────────────────────────
GREEN  = (0, 220, 0)
RED    = (0, 0, 255)
ORANGE = (0, 140, 255)
YELLOW = (0, 215, 255)
WHITE  = (255, 255, 255)
DARKBG = (18, 18, 28)

# ─────────────────────────────────────────────────────────────────────────────
#  MOUSE / ZONE STATE
# ─────────────────────────────────────────────────────────────────────────────
state = {
    "drawing"     : False,
    "define_mode" : False,
    "zone"        : None,
    "drag_start"  : None,
    "drag_current": None,
}


def mouse_handler(event, x, y, flags, param):
    if not state["define_mode"]:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        state["drawing"]      = True
        state["drag_start"]   = (x, y)
        state["drag_current"] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
        state["drag_current"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        state["drawing"]      = False
        state["drag_current"] = (x, y)
        x1 = min(state["drag_start"][0], x)
        y1 = min(state["drag_start"][1], y)
        x2 = max(state["drag_start"][0], x)
        y2 = max(state["drag_start"][1], y)
        if (x2 - x1) > 20 and (y2 - y1) > 20:
            state["zone"]        = (x1, y1, x2, y2)
            state["define_mode"] = False
            print(f"\n  ✅ Danger Zone Set: ({x1},{y1}) → ({x2},{y2})")
            print("     Zone is ACTIVE. Press Z to redefine anytime.\n")
        else:
            print("  ⚠  Zone too small — drag a bigger area.")


def get_zone(w, h):
    if state["zone"]:
        return state["zone"]
    # Default zone: left-center area (press machine side)
    return (int(w*0.05), int(h*0.05), int(w*0.60), int(h*0.95))


def in_zone(bbox, zone):
    x1, y1, x2, y2  = bbox
    zx1,zy1,zx2,zy2 = zone
    cx = (x1+x2)//2
    cy = (y1+y2)//2
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


# ─────────────────────────────────────────────────────────────────────────────
#  FIX 1 → AUTO CROP BLACK BORDERS
#  Detects the actual content area inside black borders and crops to it
# ─────────────────────────────────────────────────────────────────────────────
def crop_black_borders(frame):
    """
    Automatically detects and removes black borders from video frames.
    This fixes the issue where danger zones were being set on black
    empty border areas instead of the actual video content.
    Returns the cropped frame. If no borders found, returns original.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        return frame  # no content found, return original

    x, y, w, h = cv2.boundingRect(coords)

    # Only crop if borders are significant (more than 5% of frame)
    fh, fw = frame.shape[:2]
    if w > fw * 0.90 and h > fh * 0.90:
        return frame  # borders are tiny, not worth cropping

    cropped = frame[y:y+h, x:x+w]
    return cropped


# ─────────────────────────────────────────────────────────────────────────────
#  HOG PERSON DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect(frame):
    """
    Detect persons in frame using HOG descriptor.
    FIX 3 → winStride changed from (8,8) to (4,4) for finer detection
             of partially occluded workers near machines.
    """
    h, w  = frame.shape[:2]
    sc    = 640 / max(w, h)
    small = cv2.resize(frame, (int(w*sc), int(h*sc)))

    # FIX 3 → finer stride (4,4) instead of (8,8)
    boxes, weights = hog.detectMultiScale(
        small,
        winStride=(4, 4),   # FIX 3: was (8,8) — finer = catches more persons
        padding=(8, 8),
        scale=1.03           # FIX 3: was 1.05 — smaller scale step = more thorough
    )

    results = []
    if len(boxes):
        for (x, y, bw, bh), wt in zip(boxes, weights):
            c = float(wt[0]) if hasattr(wt, '__len__') else float(wt)
            if c < SENSITIVITY:   # FIX 2: threshold is now 0.15 (was 0.25)
                continue
            results.append((
                int(x/sc), int(y/sc),
                int((x+bw)/sc), int((y+bh)/sc),
                round(min(c, 1.0), 2)
            ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  DRAW ALL OVERLAYS
# ─────────────────────────────────────────────────────────────────────────────
def draw(frame, detections, zone, alert_active,
         alert_count, frame_num, fps, mode_label, total_frames=None):

    h, w = frame.shape[:2]
    zx1, zy1, zx2, zy2 = zone

    # ── Danger zone fill + border
    zone_col = RED if alert_active else GREEN
    ov = frame.copy()
    cv2.rectangle(ov, (zx1,zy1), (zx2,zy2), zone_col, -1)
    alpha = 0.28 if alert_active else 0.10
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    cv2.rectangle(frame, (zx1,zy1), (zx2,zy2), zone_col, 3)
    cv2.putText(frame, "DANGER ZONE", (zx1+8, zy1+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, zone_col, 2)

    # ── Person bounding boxes
    persons_in = 0
    for (x1, y1, x2, y2, conf) in detections:
        inside = in_zone((x1,y1,x2,y2), zone)
        col    = RED if inside else ORANGE
        persons_in += inside
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
        lbl = f"Person  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+8, y1), col, -1)
        cv2.putText(frame, lbl, (x1+4, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, WHITE, 1)

    # ── HUD panel top-left
    ph, pw = 160, 400
    p = frame.copy()
    cv2.rectangle(p, (0,0), (pw,ph), DARKBG, -1)
    cv2.addWeighted(p, 0.72, frame, 0.28, 0, frame)

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    progress_txt = ""
    if total_frames and total_frames > 0:
        pct = int((frame_num / total_frames) * 100)
        progress_txt = f"  {pct}% ({frame_num}/{total_frames})"

    hud = [
        (" AI Industrial Safety Monitor",            YELLOW),
        (f" Mode              :  {mode_label}",       (150,220,255)),
        (f" Persons Detected  :  {len(detections)}",  WHITE),
        (f" In Danger Zone    :  {persons_in}",       RED if persons_in else WHITE),
        (f" Status            :  {'⚠ DANGER!' if alert_active else '✔  SAFE'}",
                                                       RED if alert_active else GREEN),
        (f" Total Alerts      :  {alert_count}",      WHITE),
        (f" FPS:{fps:>4.1f}  Time:{ts}{progress_txt}", (110,110,110)),
    ]
    for i, (txt, col) in enumerate(hud):
        cv2.putText(frame, txt, (6, 22+i*21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)

    # ── Progress bar for video mode
    if total_frames and total_frames > 0:
        bx1, by1 = 6, ph-14
        bx2, by2 = pw-6, ph-5
        cv2.rectangle(frame, (bx1,by1), (bx2,by2), (55,55,55), -1)
        fill = int(bx1 + (bx2-bx1) * (frame_num/total_frames))
        cv2.rectangle(frame, (bx1,by1), (fill,by2),
                      RED if alert_active else GREEN, -1)

    # ── Hint bar bottom
    hint = " [Q]Quit  [Z]Set Zone  [S]Screenshot  [SPACE]Pause  [R]Reset Zone"
    cv2.putText(frame, hint, (6, h-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100,100,100), 1)

    # ── ALERT BANNER bottom
    if alert_active:
        bh = 68
        ab = frame.copy()
        cv2.rectangle(ab, (0,h-bh), (w,h), (0,0,175), -1)
        cv2.addWeighted(ab, 0.83, frame, 0.17, 0, frame)
        cv2.putText(frame, "  ⚠   DANGER ZONE ENTERED   ⚠",
                    (int(w*0.12), h-bh+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, WHITE, 2)
        cv2.putText(frame,
                    "  Alert Sent to CCTV Operator  |  Machine Stop Signal ACTIVATED",
                    (int(w*0.05), h-bh+56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210,210,255), 1)

    # ── Zone define mode overlay
    if state["define_mode"]:
        cv2.rectangle(frame, (0,0), (w,42), (0,80,0), -1)
        cv2.putText(frame,
                    "  ZONE DEFINE MODE — Click & Drag to draw Danger Zone",
                    (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, YELLOW, 2)
        if state["drawing"] and state["drag_start"] and state["drag_current"]:
            cv2.rectangle(frame,
                          state["drag_start"], state["drag_current"], YELLOW, 2)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  ALERT ACTION
# ─────────────────────────────────────────────────────────────────────────────
def fire_alert(frame, alert_count, frame_num, persons):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  ⚠  ALERT #{alert_count}  |  {ts}")
    print(f"  🚨 DANGER ZONE ENTERED!")
    print(f"  👷 Persons in zone     : {persons}")
    print(f"  📡 Alert sent to CCTV Operator")
    print(f"  🔔 Alarm triggered!")
    print(f"  🛑 Machine Stop Signal ACTIVATED  (Simulated)")
    print(f"{'='*60}\n")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] ALERT #{alert_count} | Frame {frame_num} | "
                f"{persons} person(s) | DANGER_ZONE_ENTERED | STOP SIGNAL SENT\n")
    snap = f"{ALERT_SNAPS}/alert_{alert_count:03d}_frame{frame_num}.jpg"
    cv2.imwrite(snap, frame)


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED PROCESSING LOOP (used by both modes)
# ─────────────────────────────────────────────────────────────────────────────
def run_monitor(cap, mode_label, total_frames=None):

    WIN = f"AI Industrial Safety Monitor  —  {mode_label}"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, WINDOW_W, WINDOW_H)
    cv2.setMouseCallback(WIN, mouse_handler)

    frame_num    = 0
    alert_count  = 0
    cooldown     = 0
    paused       = False
    screenshot_n = 0
    detections   = []
    last_frame   = None
    fps          = 0.0
    fps_timer    = time.time()
    fps_count    = 0

    print(f"\n  ✅ Running — {mode_label}")
    print("  ► Press  Z  to draw the danger zone around the machine area")
    print("  ► Press  Q  to quit\n")

    while True:

        # ── Read frame ───────────────────────────────────────────────────────
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if mode_label == "VIDEO FILE":
                    print("\n  [INFO] Video ended.")
                else:
                    print("\n  [ERROR] Camera read failed.")
                break
            last_frame = frame.copy()
        else:
            frame = last_frame.copy() if last_frame is not None \
                    else np.zeros((WINDOW_H, WINDOW_W, 3), np.uint8)

        frame_num += 1
        cooldown  = max(0, cooldown - 1)

        # FIX 1 → Auto crop black borders before doing anything
        frame = crop_black_borders(frame)

        # Resize for display
        frame = cv2.resize(frame, (WINDOW_W, WINDOW_H))

        # ── FPS counter ──────────────────────────────────────────────────────
        fps_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps       = fps_count / (now - fps_timer)
            fps_timer = now
            fps_count = 0

        # FIX 3 → Detect every 2nd frame instead of every 3rd frame
        if frame_num % 2 == 0 and not paused:
            detections = detect(frame)

        # ── Zone intrusion check ─────────────────────────────────────────────
        zone = get_zone(WINDOW_W, WINDOW_H)
        persons_in_zone = sum(
            1 for (x1,y1,x2,y2,_) in detections
            if in_zone((x1,y1,x2,y2), zone)
        )
        alert_active = persons_in_zone > 0

        if alert_active and cooldown == 0:
            alert_count += 1
            cooldown     = ALERT_COOLDOWN
            fire_alert(frame, alert_count, frame_num, persons_in_zone)

        # ── Draw overlays ────────────────────────────────────────────────────
        display = draw(frame, detections, zone,
                       alert_active, alert_count, frame_num,
                       fps, mode_label, total_frames)

        if paused:
            cv2.putText(display, "  PAUSED  —  Press SPACE to resume",
                        (int(WINDOW_W*0.18), int(WINDOW_H*0.50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, YELLOW, 3)

        cv2.imshow(WIN, display)

        # ── Keyboard controls ─────────────────────────────────────────────────
        wait_ms = 30 if mode_label == "VIDEO FILE" else 1
        key     = cv2.waitKey(wait_ms) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            print("\n  [QUIT] Stopping...")
            break
        elif key in (ord('z'), ord('Z')):
            state["define_mode"] = not state["define_mode"]
            if state["define_mode"]:
                print("\n  [ZONE MODE] Click and drag on screen to draw danger zone\n")
            else:
                print("  [ZONE MODE] Cancelled\n")
        elif key in (ord('r'), ord('R')):
            state["zone"] = None
            print("  [RESET] Zone reset to default\n")
        elif key in (ord('s'), ord('S')):
            screenshot_n += 1
            path = f"{SCREENSHOTS}/screenshot_{screenshot_n:03d}.jpg"
            cv2.imwrite(path, display)
            print(f"  [SCREENSHOT] Saved → {path}")
        elif key == ord(' '):
            paused = not paused
            print(f"  [{'PAUSED' if paused else 'RESUMED'}]")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"  Session Summary  |  {mode_label}")
    print(f"  Frames processed : {frame_num}")
    print(f"  Alerts triggered : {alert_count}")
    print(f"  Alert log        : {LOG_FILE}")
    print(f"  Alert snapshots  : {ALERT_SNAPS}/")
    print(f"  Screenshots      : {SCREENSHOTS}/")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 1 — LIVE CAMERA
# ─────────────────────────────────────────────────────────────────────────────
def run_camera():
    print(f"\n  Opening camera {CAMERA_INDEX} ...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"\n  [ERROR] Cannot open camera {CAMERA_INDEX}")
        print("  ► Change  CAMERA_INDEX = 1  at top of file and try again")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("  ✅ Camera opened!")
    print("  ► Point camera at the screen playing the industrial video")
    run_monitor(cap, "LIVE CAMERA")


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 2 — VIDEO FILE
# ─────────────────────────────────────────────────────────────────────────────
def run_video(video_path):
    video_path = video_path.strip().strip('"').strip("'")

    if not os.path.exists(video_path):
        print(f"\n  [ERROR] File not found:\n  {video_path}")
        print("  ► Check the path and try again")
        return

    print(f"\n  Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n  [ERROR] Cannot open this video file.")
        return

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_v  = cap.get(cv2.CAP_PROP_FPS)
    vw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur    = total / fps_v if fps_v > 0 else 0

    print(f"  ✅ Video loaded!")
    print(f"     Resolution : {vw} x {vh}")
    print(f"     FPS        : {fps_v:.1f}")
    print(f"     Duration   : {dur:.1f} seconds  ({total} frames)")

    run_monitor(cap, "VIDEO FILE", total)


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE MENU
# ─────────────────────────────────────────────────────────────────────────────
def show_menu():
    print("\n" + "="*60)
    print("   AI Industrial Safety Monitoring System")
    print("="*60)
    print()
    print("  Choose Mode:")
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  1  →  LIVE CAMERA                              │")
    print("  │         Point your webcam at another screen     │")
    print("  │         playing the industrial video            │")
    print("  ├─────────────────────────────────────────────────┤")
    print("  │  2  →  VIDEO FILE                               │")
    print("  │         Provide path to any .mp4 / .avi video   │")
    print("  │         and it will detect persons in it        │")
    print("  └─────────────────────────────────────────────────┘")
    print()

    choice = input("  Enter 1 or 2 : ").strip()

    if choice == "1":
        run_camera()
    elif choice == "2":
        print()
        print("  Enter the full path to your video file.")
        print(r"  Example: D:\project\ML-Accident Prevention\dataset\video.mp4")
        print("  (You can drag & drop the file into this window)")
        print()
        path = input("  Video path : ").strip().strip('"').strip("'")
        run_video(path)
    else:
        print("\n  Invalid choice. Please enter 1 or 2.\n")
        show_menu()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Industrial Safety Monitor")
    parser.add_argument("--mode", choices=["camera", "video"],
                        help="camera = live webcam | video = video file")
    parser.add_argument("--file", type=str,
                        help="Path to video file (use with --mode video)")
    args = parser.parse_args()

    if args.mode == "camera":
        run_camera()
    elif args.mode == "video":
        if not args.file:
            print("\n  [ERROR] Provide video path:  --file path/to/video.mp4")
        else:
            run_video(args.file)
    else:
        show_menu()