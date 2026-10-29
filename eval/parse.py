"""Parse RoboBrain 2.0 text outputs into coordinates.

RoboBrain 2.0 (Qwen2.5-VL) emits absolute pixel coordinates in the *resized* model
input frame (smart_resize). `scale` = (resized_w, resized_h) converts them to [0,1].
"""
import re

NUM = r"-?\d+(?:\.\d+)?"
PAIR = re.compile(rf"[\[(]\s*({NUM})\s*,\s*({NUM})\s*[\])]")
BOX = re.compile(rf"\[\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\]")


def _norm(v, size):
    return float(v) / size


def parse_points(text, scale):
    """If every value is <= 1 the answer is already normalised (e.g. Where2Place's 0-1 prompt)."""
    pairs = [(float(x), float(y)) for x, y in PAIR.findall(text)]
    if pairs and all(x <= 1 and y <= 1 for x, y in pairs) and any("." in v for p in PAIR.findall(text) for v in p):
        return pairs
    w, h = scale
    return [(_norm(x, w), _norm(y, h)) for x, y in pairs]


def parse_box(text, scale):
    """First box in the answer, normalised and with corners ordered."""
    m = BOX.search(text)
    if not m:
        return None
    w, h = scale
    x1, y1, x2, y2 = (float(v) for v in m.groups())
    x1, x2 = sorted((x1 / w, x2 / w))
    y1, y2 = sorted((y1 / h, y2 / h))
    return (x1, y1, x2, y2)


def valid_coords(pts):
    """Accept small overshoot from resize rounding; reject clearly off-frame outputs."""
    return bool(pts) and all(-0.05 <= x <= 1.05 and -0.05 <= y <= 1.05 for x, y in pts)
