"""
Login interface: capture keystroke of .tie5Roanl + Enter, classify Owner vs Impostor.
Impostor → lock the Windows workstation.
"""

from __future__ import annotations

import ctypes
import sys
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from tensorflow.keras.models import load_model

from collect_data import (
    CANONICAL_ENTER,
    INVALID_MESSAGE,
    MODIFIER_KEYS,
    KeyEvent,
    KeystrokeRound,
    extract_features,
    has_negative_timing,
    normalize_keysym,
)
from config import MODEL_PATH, NUM_FEATURES, NUM_KEYS, OWNER_THRESHOLD, PASSWORD
from preprocess import normalize_feature_vector
from split_and_save import reshape_to_3d

LOCK_DELAY_MS = 400


def lock_workstation() -> bool:
    """Lock the interactive Windows session. Returns False on non-Windows."""
    if sys.platform != "win32":
        return False
    ctypes.windll.user32.LockWorkStation()
    return True


class KeystrokeLoginApp:
    def __init__(self, root: tk.Tk, model) -> None:
        self.root = root
        self.model = model
        self.round = KeystrokeRound()
        self.password_var = tk.StringVar()
        self.status_var = tk.StringVar(value="พิมพ์รหัสผ่านแล้วกด Enter")
        self.busy = False

        self.root.title("Keystroke Dynamics Login")
        self.root.geometry("520x280")
        self.root.resizable(False, False)

        self._build_ui()
        self._bind_keys()
        self.password_entry.focus_set()

    def _build_ui(self) -> None:
        frm = ttk.Frame(self.root, padding=20)
        frm.pack(fill="both", expand=True)

        ttk.Label(
            frm,
            text="ยืนยันตัวตนด้วยจังหวะการพิมพ์",
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            frm,
            text=f"พิมพ์รหัสผ่านที่กำหนด ({NUM_KEYS} คีย์ รวม Enter) ให้ครบหนึ่งรอบ",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(4, 16))

        ttk.Label(frm, text="Password").pack(anchor="w")
        self.password_entry = ttk.Entry(
            frm,
            textvariable=self.password_var,
            show="*",
            font=("Consolas", 12),
        )
        self.password_entry.pack(fill="x", pady=(4, 12))

        self.status_label = ttk.Label(
            frm,
            textvariable=self.status_var,
            wraplength=460,
        )
        self.status_label.pack(anchor="w", pady=(8, 0))

    def _bind_keys(self) -> None:
        self.password_entry.bind("<KeyPress>", self._on_key_press)
        self.password_entry.bind("<KeyRelease>", self._on_key_release)

    def _set_status(self, message: str, color: str = "#2c3e50") -> None:
        self.status_var.set(message)
        self.status_label.configure(foreground=color)

    def _expected_char(self, index: int) -> str | None:
        if index < len(PASSWORD):
            return PASSWORD[index]
        if index == len(PASSWORD):
            return None
        return ""

    def _invalidate_round(self, message: str = INVALID_MESSAGE) -> None:
        self.round.reset()
        self.round.invalidated = True
        self.password_var.set("")
        self._set_status(message, "#c0392b")
        self.password_entry.focus_set()

    def _on_key_press(self, event: tk.Event) -> str | None:
        if self.busy:
            return "break"

        keysym = normalize_keysym(event.keysym)
        keycode = int(getattr(event, "keycode", 0) or 0)

        if keysym in MODIFIER_KEYS:
            return None

        if self.round.invalidated:
            if keysym not in ("BackSpace", CANONICAL_ENTER):
                self.round.invalidated = False
            else:
                return "break"

        if keysym == "BackSpace":
            self._invalidate_round()
            return "break"

        if keycode:
            if keycode in self.round.keys_down:
                return "break"
        elif keysym in {ev.keysym for ev in self.round.events if ev.release_time is None}:
            return "break"

        key_idx = self.round.key_index()
        if key_idx >= NUM_KEYS:
            self._invalidate_round("พิมพ์เกินจำนวนที่กำหนด — เริ่มใหม่")
            return "break"

        expected = self._expected_char(key_idx)
        if expected is None:
            if keysym != CANONICAL_ENTER:
                self._invalidate_round()
                return "break"
        else:
            char = event.char
            if not char or char != expected:
                self._invalidate_round()
                return "break"

        if keycode:
            self.round.keys_down.add(keycode)
        self.round.events.append(
            KeyEvent(keysym=keysym, keycode=keycode, press_time=time.perf_counter())
        )
        self._set_status(f"กำลังบันทึกจังหวะพิมพ์... ({key_idx + 1}/{NUM_KEYS})")
        return "break" if keysym == CANONICAL_ENTER else None

    def _on_key_release(self, event: tk.Event) -> str | None:
        if self.busy:
            return "break"

        keysym = normalize_keysym(event.keysym)
        keycode = int(getattr(event, "keycode", 0) or 0)

        if keysym in MODIFIER_KEYS:
            return None

        if self.round.invalidated:
            if keycode:
                self.round.keys_down.discard(keycode)
            return "break"

        open_by_code = bool(keycode) and self.round.has_open_press(keycode)
        open_by_sym = self.round.has_open_press(0, keysym)
        tracked = bool(keycode) and keycode in self.round.keys_down

        if not tracked and not open_by_code and not open_by_sym:
            return "break"

        if keycode:
            self.round.keys_down.discard(keycode)
        self.round.record_release(keycode, time.perf_counter(), keysym=keysym)

        if self.round.is_complete():
            self._classify_round()

        return "break" if keysym == CANONICAL_ENTER else None

    def _classify_round(self) -> None:
        if not self.round.is_complete():
            self._invalidate_round("ข้อมูล timing ไม่ครบ — เริ่มพิมพ์ใหม่")
            return

        try:
            features = extract_features(self.round.press_times(), self.round.release_times())
        except ValueError as exc:
            self._invalidate_round(str(exc))
            return

        if len(features) != NUM_FEATURES or has_negative_timing(features):
            self._invalidate_round("ข้อมูล timing ไม่ถูกต้อง — เริ่มพิมพ์ใหม่")
            return

        normalized = normalize_feature_vector(features)
        x = reshape_to_3d(normalized.reshape(1, -1))
        prob_owner = float(self.model.predict(x, verbose=0).ravel()[0])
        is_owner = prob_owner >= OWNER_THRESHOLD

        self.round.reset()
        self.password_var.set("")

        if is_owner:
            self._set_status(
                f"ผ่าน — ระบุเป็นเจ้าของเครื่อง (P(owner)={prob_owner:.3f})",
                "#1e8449",
            )
            self.password_entry.focus_set()
            return

        self.busy = True
        self._set_status(
            f"ไม่ผ่าน — ระบุเป็นผู้ไม่ได้รับอนุญาต (P(owner)={prob_owner:.3f}) กำลังล็อกหน้าจอ",
            "#c0392b",
        )
        self.root.after(LOCK_DELAY_MS, self._lock_and_reset)

    def _lock_and_reset(self) -> None:
        locked = lock_workstation()
        self.busy = False
        if not locked:
            self._set_status(
                "ระบุเป็นผู้ไม่ได้รับอนุญาต แต่ล็อกหน้าจอได้เฉพาะบน Windows",
                "#c0392b",
            )
        self.password_entry.focus_set()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not MODEL_PATH.is_file():
        print(f"ERROR: ไม่พบโมเดล {MODEL_PATH}")
        print("กรุณารัน train.py ก่อน")
        sys.exit(1)

    model = load_model(MODEL_PATH)
    root = tk.Tk()
    app = KeystrokeLoginApp(root, model)
    app.run()


if __name__ == "__main__":
    main()
