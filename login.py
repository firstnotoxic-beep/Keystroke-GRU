"""
Login interface: capture keystroke of .tie5Roanl + Enter, classify Owner vs Impostor.
Real-time continuous authentication GUI (pynput Producer-Consumer).
Impostor streak → optionally lock the Windows workstation (unless DRY_RUN).
"""

from __future__ import annotations

import csv
import ctypes
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk
from typing import Any

import numpy as np
from pynput import keyboard
from tensorflow.keras.models import load_model

from collect_data import (
    CANONICAL_ENTER,
    INVALID_MESSAGE,
    KeyEvent,
    KeystrokeRound,
    extract_features,
    has_negative_timing,
)
from config import (
    DRY_RUN,
    MAX_CONSECUTIVE_FAILS,
    NUM_FEATURES,
    NUM_KEYS,
    PASSWORD,
    PROJECT_ROOT,
    REALTIME_MODEL_PATH as MODEL_PATH,
    REALTIME_NORMALIZE,
    REALTIME_THRESHOLD as LOGIN_THRESHOLD,
)
from experiment.common import intra_sequence_ratio_normalization, reshape_sequences

LOCK_DELAY_MS = 400
HEADER_BG = "#1f3a5f"
BODY_BG = "#ffffff"
STATUS_TINTS = {
    "#1e8449": "#e8f6ee",
    "#c0392b": "#fdecea",
    "#7f8c8d": "#f2f4f5",
    "#2c3e50": "#eef3f8",
}
LOGIN_LOG_PATH = PROJECT_ROOT / "data" / "login_log.csv"
VALIDATION_FAIL_MESSAGE = "ข้อมูล timing ไม่ถูกต้อง — กรุณาพิมพ์ใหม่"
PASTE_BLOCK_MESSAGE = "ตรวจพบการวางข้อความ (Ctrl+V) — กรุณาพิมพ์ใหม่"
DELETE_RESET_MESSAGE = "ลบข้อความแล้ว — กรุณาพิมพ์ใหม่"

# Sentinel to stop the consumer thread.
_STOP = object()


def lock_workstation() -> bool:
    """Lock the interactive Windows session. Returns False on non-Windows."""
    if sys.platform != "win32":
        return False
    ctypes.windll.user32.LockWorkStation()
    return True


def _append_login_log(p_owner: float, action: str) -> None:
    """Append one prediction row to data/login_log.csv (demo evidence)."""
    LOGIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOGIN_LOG_PATH.is_file()
    with LOGIN_LOG_PATH.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["timestamp", "p_owner", "action"])
        writer.writerow(
            [datetime.now().isoformat(timespec="seconds"), f"{p_owner:.6f}", action]
        )


def _pynput_vk(key: Any) -> int:
    return int(getattr(key, "vk", 0) or 0)


def _is_ctrl_key(key: Any) -> bool:
    return key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)


def _is_enter_key(key: Any) -> bool:
    return key in (keyboard.Key.enter,)


def _is_backspace_key(key: Any) -> bool:
    return key == keyboard.Key.backspace


def _is_delete_key(key: Any) -> bool:
    return key == keyboard.Key.delete


def _is_paste_v(key: Any) -> bool:
    """True for V / Ctrl+V control char under modifiers (Windows vk 0x56)."""
    if isinstance(key, keyboard.KeyCode):
        ch = key.char
        if ch in ("v", "V", "\x16"):
            return True
        if _pynput_vk(key) == 0x56:
            return True
    return False


def _key_to_symbol(key: Any) -> tuple[str | None, int]:
    """Map a pynput key to (keysym-like token, vk/keycode) for KeystrokeRound."""
    vk = _pynput_vk(key)
    if _is_enter_key(key):
        return CANONICAL_ENTER, vk or 13
    if _is_backspace_key(key):
        return "BackSpace", vk or 8
    if _is_delete_key(key):
        return "Delete", vk or 46
    if isinstance(key, keyboard.KeyCode) and key.char:
        # Ignore pure control chars except we already handled paste separately.
        if ord(key.char) < 32:
            return None, vk
        return key.char, vk
    return None, vk


class KeystrokeLoginApp:
    def __init__(self, root: tk.Tk, model) -> None:
        self.root = root
        self.model = model
        self.round = KeystrokeRound()
        self.password_var = tk.StringVar()
        self.status_var = tk.StringVar(value="พิมพ์รหัสผ่านแล้วกด Enter")
        self.fail_count_var = tk.StringVar(value="Fail streak: 0")
        self.busy = False
        self.ready = False
        self.fail_count = 0
        self.ctrl_down = False
        self._event_queue: queue.Queue[Any] = queue.Queue()
        self._stop_event = threading.Event()
        self._listener: keyboard.Listener | None = None
        self._consumer: threading.Thread | None = None

        self.root.title("Keystroke Dynamics Login")
        self.root.geometry("540x310")
        self.root.resizable(False, False)

        self._build_ui()
        # Block Entry from handling keys itself; timing comes from pynput.
        self.password_entry.bind("<Key>", lambda _e: "break")
        self.password_entry.focus_set()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        dry_note = " [DRY_RUN]" if DRY_RUN else ""
        self._set_status(
            f"กำลังอุ่นโมเดล...{dry_note}",
            "#7f8c8d",
        )
        self.root.after(50, self._finish_startup)

    def _build_ui(self) -> None:
        self.root.configure(bg=BODY_BG)
        header = tk.Frame(self.root, bg=HEADER_BG, padx=20, pady=12)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Keystroke Dynamics Authentication",
            font=("Segoe UI", 14, "bold"),
            fg="white",
            bg=HEADER_BG,
        ).pack(anchor="w")
        tk.Label(
            header,
            text="ยืนยันตัวตนต่อเนื่องด้วยจังหวะการพิมพ์ (GRU)",
            font=("Segoe UI", 9),
            fg="#c9d6e8",
            bg=HEADER_BG,
        ).pack(anchor="w", pady=(2, 0))

        frm = tk.Frame(self.root, bg=BODY_BG, padx=20, pady=14)
        frm.pack(fill="both", expand=True)

        tk.Label(
            frm,
            text=f"Password  ·  พิมพ์ให้ครบ {NUM_KEYS} คีย์ รวม Enter",
            font=("Segoe UI", 9, "bold"),
            fg="#34495e",
            bg=BODY_BG,
        ).pack(anchor="w")
        self.password_entry = ttk.Entry(
            frm,
            textvariable=self.password_var,
            show="●",
            font=("Consolas", 14),
        )
        self.password_entry.pack(fill="x", pady=(6, 12), ipady=3)

        self.status_label = tk.Label(
            frm,
            textvariable=self.status_var,
            wraplength=470,
            justify="left",
            anchor="w",
            font=("Segoe UI", 10),
            padx=12,
            pady=8,
        )
        self.status_label.pack(fill="x")

        bottom = tk.Frame(frm, bg=BODY_BG)
        bottom.pack(fill="x", pady=(12, 0))
        self.fail_dots = tk.Canvas(
            bottom,
            width=18 * MAX_CONSECUTIVE_FAILS,
            height=14,
            bg=BODY_BG,
            highlightthickness=0,
        )
        self.fail_dots.pack(side="left")
        self.fail_label = tk.Label(
            bottom,
            textvariable=self.fail_count_var,
            font=("Segoe UI", 9),
            fg="#7f8c8d",
            bg=BODY_BG,
        )
        self.fail_label.pack(side="left", padx=(6, 0))
        tk.Label(
            bottom,
            text=f"threshold τ = {LOGIN_THRESHOLD}",
            font=("Segoe UI", 9),
            fg="#95a5a6",
            bg=BODY_BG,
        ).pack(side="right")
        self._draw_fail_dots()

    def _draw_fail_dots(self) -> None:
        self.fail_dots.delete("all")
        for i in range(MAX_CONSECUTIVE_FAILS):
            x = 2 + i * 18
            fill = "#c0392b" if i < self.fail_count else "#dfe4e8"
            self.fail_dots.create_oval(x, 1, x + 12, 13, fill=fill, outline="")

    def _set_status(self, message: str, color: str = "#2c3e50") -> None:
        self.status_var.set(message)
        self.status_label.configure(
            foreground=color, background=STATUS_TINTS.get(color, "#eef3f8")
        )

    def _ui(self, fn) -> None:
        """Schedule a UI update on the Tk main thread."""
        self.root.after(0, fn)

    def _update_fail_label(self) -> None:
        mode = "DRY_RUN" if DRY_RUN else "LIVE"
        self.fail_count_var.set(
            f"Fail streak: {self.fail_count}/{MAX_CONSECUTIVE_FAILS} ({mode})"
        )
        self._draw_fail_dots()

    def _expected_char(self, index: int) -> str | None:
        if index < len(PASSWORD):
            return PASSWORD[index]
        if index == len(PASSWORD):
            return None
        return ""

    def _window_focused(self) -> bool:
        try:
            return self.root.focus_displayof() is not None
        except tk.TclError:
            return False

    def _invalidate_round(self, message: str = INVALID_MESSAGE) -> None:
        self.round.reset()
        self.round.invalidated = True
        self.password_var.set("")
        self._set_status(message, "#c0392b")
        self.password_entry.focus_set()

    def _finish_startup(self) -> None:
        """Warm-up predict, then start Producer (listener) + Consumer threads."""
        try:
            # Must run before accepting real input (requirement).
            self.model.predict(np.zeros((1, 11, 3), dtype=np.float32), verbose=0)
        except Exception as exc:  # noqa: BLE001 — show warm-up failure in UI
            self._set_status(f"อุ่นโมเดลไม่สำเร็จ: {exc}", "#c0392b")
            return

        self.ready = True
        dry_note = " (DRY_RUN — จะไม่ล็อกหน้าจอ)" if DRY_RUN else ""
        self._set_status(f"พิมพ์รหัสผ่านแล้วกด Enter{dry_note}", "#2c3e50")
        self._update_fail_label()
        self._start_threads()

    def _start_threads(self) -> None:
        self._consumer = threading.Thread(
            target=self._consumer_loop,
            name="login-consumer",
            daemon=True,
        )
        self._consumer.start()

        self._listener = keyboard.Listener(
            on_press=self._on_press_producer,
            on_release=self._on_release_producer,
        )
        self._listener.daemon = True
        self._listener.start()

    def _on_press_producer(self, key: Any) -> None:
        # Listener thread: enqueue only — never predict here.
        self._event_queue.put(("press", key, time.perf_counter()))

    def _on_release_producer(self, key: Any) -> None:
        self._event_queue.put(("release", key, time.perf_counter()))

    def _consumer_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is _STOP:
                break
            kind, key, ts = item
            if kind == "press":
                self._handle_press(key, ts)
            else:
                self._handle_release(key, ts)

    def _handle_press(self, key: Any, ts: float) -> None:
        if _is_ctrl_key(key):
            self.ctrl_down = True
            return

        if not self.ready or self.busy or not self._window_focused():
            return

        # Ctrl+V — block paste like continuous-auth collection policy.
        if self.ctrl_down and _is_paste_v(key):
            self._ui(lambda: self._invalidate_round(PASTE_BLOCK_MESSAGE))
            return

        if _is_backspace_key(key) or _is_delete_key(key):
            msg = DELETE_RESET_MESSAGE if _is_delete_key(key) else INVALID_MESSAGE
            self._ui(lambda m=msg: self._invalidate_round(m))
            return

        symbol, keycode = _key_to_symbol(key)
        if symbol is None:
            return

        if self.round.invalidated:
            if symbol not in ("BackSpace", "Delete", CANONICAL_ENTER):
                self.round.invalidated = False
            else:
                return

        # Deduplicate auto-repeat while key is held.
        if keycode:
            if keycode in self.round.keys_down:
                return
        elif symbol in {ev.keysym for ev in self.round.events if ev.release_time is None}:
            return

        key_idx = self.round.key_index()
        if key_idx >= NUM_KEYS:
            self._ui(lambda: self._invalidate_round("พิมพ์เกินจำนวนที่กำหนด — กรุณาพิมพ์ใหม่"))
            return

        expected = self._expected_char(key_idx)
        if expected is None:
            # Final key must be Enter.
            if symbol != CANONICAL_ENTER:
                self._ui(lambda: self._invalidate_round(INVALID_MESSAGE))
                return
        else:
            if symbol != expected:
                # Key outside sequence .tie5Roanl → reset buffer.
                self._ui(lambda: self._invalidate_round(INVALID_MESSAGE))
                return

        if keycode:
            self.round.keys_down.add(keycode)
        self.round.events.append(
            KeyEvent(keysym=symbol, keycode=keycode, press_time=ts)
        )
        shown = "*" * len(self.round.events)
        idx = key_idx + 1
        self._ui(
            lambda s=shown, i=idx: (
                self.password_var.set(s),
                self._set_status(f"กำลังบันทึกจังหวะพิมพ์... ({i}/{NUM_KEYS})"),
            )
        )

    def _handle_release(self, key: Any, ts: float) -> None:
        if _is_ctrl_key(key):
            self.ctrl_down = False
            return

        if not self.ready or self.busy or not self._window_focused():
            return

        symbol, keycode = _key_to_symbol(key)
        if symbol is None and not keycode:
            return

        if self.round.invalidated:
            if keycode:
                self.round.keys_down.discard(keycode)
            return

        open_by_code = bool(keycode) and self.round.has_open_press(keycode)
        open_by_sym = bool(symbol) and self.round.has_open_press(0, symbol)
        tracked = bool(keycode) and keycode in self.round.keys_down

        if not tracked and not open_by_code and not open_by_sym:
            return

        if keycode:
            self.round.keys_down.discard(keycode)
        self.round.record_release(keycode, ts, keysym=symbol)

        # Predict only on KeyRelease of Enter when the round is complete.
        if symbol == CANONICAL_ENTER and self.round.is_complete():
            self._classify_round()

    def _classify_round(self) -> None:
        """Validate features then predict on the consumer thread (not listener)."""
        if not self.round.is_complete():
            self._ui(lambda: self._invalidate_round("ข้อมูล timing ไม่ครบ — กรุณาพิมพ์ใหม่"))
            return

        try:
            features = extract_features(self.round.press_times(), self.round.release_times())
        except ValueError as exc:
            self._ui(lambda m=str(exc): self._invalidate_round(m))
            return

        # Exactly 31 features; H >= 0 and DD >= 0 (UD may be negative).
        if len(features) != NUM_FEATURES or has_negative_timing(features):
            self._ui(lambda: self._invalidate_round(VALIDATION_FAIL_MESSAGE))
            return

        vec = np.asarray(features, dtype=np.float64)
        if REALTIME_NORMALIZE:
            vec = intra_sequence_ratio_normalization(vec)
        x = reshape_sequences(vec.reshape(1, -1))
        p_owner = float(self.model.predict(x, verbose=0).ravel()[0])

        self.round.reset()
        self._ui(lambda: self.password_var.set(""))

        if p_owner < LOGIN_THRESHOLD:
            self.fail_count += 1
            action = "reject"
            should_lock = self.fail_count >= MAX_CONSECUTIVE_FAILS and not DRY_RUN
            if self.fail_count >= MAX_CONSECUTIVE_FAILS and DRY_RUN:
                action = "reject_dry_run_lock_skipped"
            elif should_lock:
                action = "reject_lock"

            _append_login_log(p_owner, action)
            fail_n = self.fail_count
            self._ui(lambda: self._update_fail_label())

            if should_lock:
                self.busy = True
                self._ui(
                    lambda p=p_owner, n=fail_n: self._set_status(
                        f"ไม่ผ่าน — P(owner)={p:.3f} (fail {n}/{MAX_CONSECUTIVE_FAILS}) "
                        f"กำลังล็อกหน้าจอ",
                        "#c0392b",
                    )
                )
                self._ui(lambda: self.root.after(LOCK_DELAY_MS, self._lock_and_reset))
                return

            dry = " [DRY_RUN — ไม่ล็อก]" if DRY_RUN else ""
            self._ui(
                lambda p=p_owner, n=fail_n, d=dry: self._set_status(
                    f"ไม่ผ่าน — P(owner)={p:.3f} < {LOGIN_THRESHOLD} "
                    f"(fail {n}/{MAX_CONSECUTIVE_FAILS}){d}",
                    "#c0392b",
                )
            )
            self._ui(lambda: self.password_entry.focus_set())
            return

        self.fail_count = 0
        _append_login_log(p_owner, "accept")
        self._ui(lambda: self._update_fail_label())
        self._ui(
            lambda p=p_owner: self._set_status(
                f"ผ่าน — ระบุเป็นเจ้าของเครื่อง (P(owner)={p:.3f})",
                "#1e8449",
            )
        )
        self._ui(lambda: self.password_entry.focus_set())

    def _lock_and_reset(self) -> None:
        locked = lock_workstation()
        self.busy = False
        if not locked:
            self._set_status(
                "ระบุเป็นผู้ไม่ได้รับอนุญาต แต่ล็อกหน้าจอได้เฉพาะบน Windows",
                "#c0392b",
            )
        self.password_entry.focus_set()

    def _on_close(self) -> None:
        self._stop_event.set()
        self._event_queue.put(_STOP)
        if self._listener is not None:
            self._listener.stop()
        self.root.destroy()

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
