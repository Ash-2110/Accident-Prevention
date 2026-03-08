"""
=============================================================================
Danger Zone Configurator
=============================================================================
An interactive tool to visually define the danger zone rectangle on
a camera frame or video still. Click and drag to draw the zone, then
press 'S' to save the coordinates to config.

Usage:
    python src/configure_zone.py --source 0          # webcam
    python src/configure_zone.py --source video.mp4  # video file
=============================================================================
"""

import cv2
import numpy as np
import argparse
import json
import os

# State
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
zone_defined = False


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, fx, fy, zone_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y
        zone_defined = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        zone_defined = True
        print(f"\n[ZONE] Defined: ({ix}, {iy}) → ({fx}, {fy})")
        print(f"       Width: {abs(fx-ix)}px  Height: {abs(fy-iy)}px")
        print("       Press 'S' to save, 'R' to reset, 'Q' to quit.")


def configure_zone(source=0):
    global zone_defined

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read frame.")
        return

    cap.release()
    frame = cv2.resize(frame, (720, 480))
    base_frame = frame.copy()

    cv2.namedWindow("Danger Zone Configurator")
    cv2.setMouseCallback("Danger Zone Configurator", mouse_callback)

    print("\n=== Danger Zone Configurator ===")
    print("  Click and drag to draw the danger zone rectangle.")
    print("  Press S to save   | R to reset   | Q to quit")

    while True:
        display = base_frame.copy()

        # Draw instructions
        cv2.putText(display, "Draw danger zone: Click & Drag",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Draw current rectangle
        if ix >= 0 and iy >= 0 and fx >= 0 and fy >= 0:
            color = (0, 200, 0) if zone_defined else (0, 165, 255)
            cv2.rectangle(display,
                          (min(ix,fx), min(iy,fy)),
                          (max(ix,fx), max(iy,fy)),
                          color, 2)
            if zone_defined:
                cv2.putText(display,
                            f"Zone: ({min(ix,fx)},{min(iy,fy)}) -> ({max(ix,fx)},{max(iy,fy)})",
                            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imshow("Danger Zone Configurator", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('s') or key == ord('S'):
            if zone_defined:
                zone = (min(ix,fx), min(iy,fy), max(ix,fx), max(iy,fy))
                save_zone(zone)
            else:
                print("[WARNING] No zone defined yet. Draw a rectangle first.")

        elif key == ord('r') or key == ord('R'):
            ix, iy, fx, fy = -1, -1, -1, -1
            zone_defined = False
            print("[RESET] Zone cleared.")

        elif key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()


def save_zone(zone: tuple):
    """Save zone coordinates to JSON config."""
    os.makedirs("outputs", exist_ok=True)
    config = {
        "danger_zone": {
            "x1": zone[0], "y1": zone[1],
            "x2": zone[2], "y2": zone[3]
        },
        "description": "Danger zone coordinates for safety_monitor.py"
    }
    path = "outputs/danger_zone_config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[SAVED] Danger zone config → {path}")
    print(f"        DANGER_ZONE = ({zone[0]}, {zone[1]}, {zone[2]}, {zone[3]})")
    print("\n  Update Config.DANGER_ZONE in safety_monitor.py with these values.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danger Zone Configurator")
    parser.add_argument("--source", default="0",
                        help="Video source: 0=webcam or path to video")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    configure_zone(source)
