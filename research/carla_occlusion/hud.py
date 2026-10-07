"""HUD overlay for recorded CARLA video (layout after CARLA's manual_control.py HUD)."""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, x, y, scale=0.5, color=(235, 235, 235), bold=1):
    cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), bold + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, bold, cv2.LINE_AA)


def _bar(img, x, y, w, value, lo=0.0, hi=1.0, color=(90, 200, 255), centered=False):
    cv2.rectangle(img, (x, y), (x + w, y + 10), (70, 70, 70), 1)
    f = (value - lo) / (hi - lo)
    if centered:
        mid = x + w // 2
        end = int(x + np.clip(f, 0, 1) * w)
        cv2.rectangle(img, (min(mid, end), y + 1), (max(mid, end), y + 9), color, -1)
    else:
        cv2.rectangle(img, (x + 1, y + 1), (x + int(np.clip(f, 0, 1) * (w - 1)), y + 9), color, -1)


def draw(img, info: dict, x0: int = 12, y0: int = 12, width: int = 360):
    """info: title, lines [(label, value)], bars [(label, value, lo, hi, centered)], alerts [str]."""
    lines, bars, alerts = info.get("lines", []), info.get("bars", []), info.get("alerts", [])
    h = 34 + 20 * len(lines) + 24 * len(bars) + 22 * len(alerts) + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)
    y = y0 + 24
    _text(img, info.get("title", ""), x0 + 10, y, 0.55, (255, 255, 255), 1)
    y += 8
    for label, value in lines:
        y += 20
        _text(img, f"{label}", x0 + 10, y, 0.45, (170, 175, 180))
        _text(img, f"{value}", x0 + 140, y, 0.45)
    for label, value, lo, hi, centered in bars:
        y += 24
        _text(img, label, x0 + 10, y + 9, 0.45, (170, 175, 180))
        _bar(img, x0 + 140, y, width - 160, value, lo, hi, centered=centered)
    for a in alerts:
        y += 22
        _text(img, a, x0 + 10, y + 4, 0.5, (80, 80, 255), 1)
    return img
