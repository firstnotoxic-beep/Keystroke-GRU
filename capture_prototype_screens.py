"""Capture prototype screenshots (collector + real-time login) and compose the paper banner.

States are set from real model outputs on real test samples (no live typing, no lock):
  (a) data-collection window, (b) owner accepted, (c) impostor rejected (fail streak 2/3).

Usage:
    python capture_prototype_screens.py
Outputs: figures/screens/*.png and figures/figure0_prototype_banner.png (15.5 x 4.9 cm @ 300 dpi)
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
import sys
import time
import tkinter as tk

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageGrab  # noqa: E402

from config import FEAT_COLS, PROJECT_ROOT  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except (AttributeError, OSError):
    pass

SCREENS_DIR = PROJECT_ROOT / "figures" / "screens"
BANNER_PATH = PROJECT_ROOT / "figures" / "figure0_prototype_banner.png"


def window_rect(root: tk.Tk) -> tuple[int, int, int, int]:
    """Screen rectangle of the whole window (incl. title bar, excl. invisible DWM border)."""
    hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
    rect = ctypes.wintypes.RECT()
    ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect))
    return rect.left, rect.top, rect.right, rect.bottom


def grab(root: tk.Tk, name: str) -> Image.Image:
    """Bring window to front, let it paint, then screenshot it."""
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()
    for _ in range(20):
        root.update()
        time.sleep(0.05)
    img = ImageGrab.grab(bbox=window_rect(root), all_screens=True)
    SCREENS_DIR.mkdir(parents=True, exist_ok=True)
    img.save(SCREENS_DIR / f"{name}.png")
    print(f"[capture] {name}: {img.size}")
    return img


def pick_samples():
    """One real owner and one real impostor test sample with their model scores."""
    from tensorflow.keras.models import load_model

    from config import REALTIME_MODEL_PATH, REALTIME_THRESHOLD
    from experiment.common import LABEL_COL, load_real_data, reshape_sequences, split_real

    _, _, test = split_real(load_real_data())
    model = load_model(REALTIME_MODEL_PATH)
    X = test[FEAT_COLS].to_numpy(dtype=np.float64)
    p = model.predict(reshape_sequences(X), verbose=0).ravel()
    y = test[LABEL_COL].to_numpy()
    owner = int(np.where(y == 1)[0][np.argsort(-p[y == 1])][len(p[y == 1]) // 4])
    imp_idx = np.where((y == 0) & (p < REALTIME_THRESHOLD))[0]
    impostor = int(imp_idx[np.argsort(p[imp_idx])][len(imp_idx) // 2])
    return model, X[owner], float(p[owner]), X[impostor], float(p[impostor])


def capture_collector() -> Image.Image:
    """Data-collection window for the owner, mid-session."""
    import collect_data as cd

    root = tk.Tk()
    app = cd.KeystrokeCollectorApp(root)
    import pandas as pd

    app.password_var.set(".tie5")
    feats = pd.read_csv(cd.subject_raw_path("owner")).iloc[-1][FEAT_COLS].to_numpy(dtype=float)
    app.feedback_var.set(f"H (เฉลี่ย): {feats[:11].mean():.4f} s | "
                         f"DD (เฉลี่ย): {feats[11:21].mean():.4f} s")
    app._set_status(cd.SUCCESS_MESSAGE, "#1e8449")
    img = grab(root, "a_collect")
    root.destroy()
    return img


def capture_login(model, p_owner: float, p_imp: float) -> tuple[Image.Image, Image.Image]:
    """Login window: owner accepted, then impostor rejected with fail streak 2/3."""
    import login as lg

    root = tk.Tk()
    app = lg.KeystrokeLoginApp(root, model)
    for _ in range(40):
        root.update()
        time.sleep(0.05)
        if app.ready:
            break
    app.fail_count = 0
    app._update_fail_label()
    app.password_var.set("")
    app._set_status(f"ผ่าน — ระบุเป็นเจ้าของเครื่อง (P(owner)={p_owner:.3f})", "#1e8449")
    img_b = grab(root, "b_login_accept")

    app.fail_count = 2
    app._update_fail_label()
    app._set_status(f"ไม่ผ่าน — P(owner)={p_imp:.3f} < {lg.LOGIN_THRESHOLD} "
                    f"(fail 2/{lg.MAX_CONSECUTIVE_FAILS})", "#c0392b")
    img_c = grab(root, "c_login_reject")
    app._on_close()
    return img_b, img_c


def compose_banner(shots: list[Image.Image], captions: list[str]) -> None:
    """Place screenshots side by side on a 15.5 x 4.9 cm canvas (300 dpi) with captions."""
    W, H = round(15.5 / 2.54 * 300), round(4.9 / 2.54 * 300)
    margin, gap, cap_h = 22, 30, 58
    avail_h = H - 2 * margin - cap_h
    aspect_sum = sum(s.width / s.height for s in shots)
    target_h = min(avail_h, (W - 2 * margin - gap * (len(shots) - 1)) / aspect_sum)
    shots = [s.resize((round(s.width * target_h / s.height), round(target_h)), Image.LANCZOS)
             for s in shots]
    canvas = Image.new("RGB", (W, H), "white")
    total_w = sum(s.width for s in shots) + gap * (len(shots) - 1)
    x = (W - total_w) // 2
    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for shot, cap in zip(shots, captions):
        y = margin + (avail_h - shot.height) // 2
        shadow = Image.new("RGBA", (shot.width + 24, shot.height + 24), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rectangle([12, 12, shot.width + 12, shot.height + 12],
                                         fill=(0, 0, 0, 70))
        shadow = shadow.filter(ImageFilter.GaussianBlur(7))
        canvas.paste(shadow, (x - 8, y - 6), shadow)
        canvas.paste(shot, (x, y))
        draw.rectangle([x, y, x + shot.width - 1, y + shot.height - 1], outline="#b8c2cc")
        tw = draw.textlength(cap, font=font)
        draw.text((x + (shot.width - tw) / 2, H - margin - cap_h + 18), cap,
                  fill="#1a1a1a", font=font)
        x += shot.width + gap
    canvas.save(BANNER_PATH, dpi=(300, 300))
    print(f"[banner] {BANNER_PATH} {canvas.size}")


def main() -> None:
    """Capture all three states and build the banner."""
    model, _, p_owner, _, p_imp = pick_samples()
    print(f"[samples] owner P={p_owner:.3f}, impostor P={p_imp:.3f}")
    shots = [capture_collector(), *capture_login(model, p_owner, p_imp)]
    compose_banner(shots, ["(a) Data collection", "(b) Owner accepted",
                           "(c) Impostor rejected (2/3)"])


if __name__ == "__main__":
    main()
