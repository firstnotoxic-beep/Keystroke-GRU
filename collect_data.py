"""
สคริปต์เก็บข้อมูล Keystroke Dynamics ผ่าน GUI (tkinter)
รหัสผ่านเป้าหมาย: .tie5Roanl + Enter (11 คีย์ → 31 ฟีเจอร์)
"""

from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv
import sys
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk

from config import (
    NUM_DD,
    NUM_FEATURES,
    NUM_HOLD,
    NUM_KEYS,
    NUM_UD,
    OUTPUT_COLS,
    PASSWORD,
    RAW_DATA_DIR,
    normalize_subject_name,
    subject_raw_path,
)

ANOMALY_THRESHOLD_SEC = 1.5

MODIFIER_KEYS = frozenset({
    "Shift_L", "Shift_R", "Control_L", "Control_R",
    "Alt_L", "Alt_R", "Meta_L", "Meta_R", "Caps_Lock",
})

INVALID_MESSAGE = "พบการพิมพ์ผิด: กรุณาเริ่มพิมพ์ใหม่ตั้งแต่ต้น"
FOCUS_WARNING_MESSAGE = (
    "กรุณามีสมาธิและงดการพูดคุยขณะพิมพ์ เพื่อความแม่นยำของข้อมูลที่ใช้ฝึก AI"
)
FEEDBACK_PLACEHOLDER = "H (เฉลี่ย): — s | DD (เฉลี่ย): — s"
SUCCESS_MESSAGE = "รอบล่าสุด: สำเร็จ (พิมพ์ปกติ)"
ANOMALY_MESSAGE = (
    "ตรวจพบจังหวะพิมพ์ชะงักผิดปกติ (อาจเกิดจากการใจลอยหรือพูดคุย) "
    "ระบบจะไม่บันทึกรอบนี้ลง CSV กรุณาพิมพ์ใหม่อีกครั้ง"
)
NEGATIVE_TIMING_MESSAGE = (
    "ตรวจพบค่า timing ติดลบ (ข้อมูลไม่ถูกต้อง) — กรุณาพิมพ์ใหม่อีกครั้ง"
)
LABEL_MISMATCH_MESSAGE = (
    "Subject Name นี้เคยใช้กับ Label อื่นแล้ว — "
    "กรุณาใช้ Label เดิมหรือเปลี่ยนชื่อ Subject"
)
UNDO_SUCCESS_MESSAGE = "ลบข้อมูลรอบล่าสุดสำเร็จแล้ว"


def build_csv_header() -> list[str]:
    """สร้างรายชื่อคอลัมน์ CSV ตามสเปก [Subject_Name, Label, H*, DD*, UD*]."""
    return list(OUTPUT_COLS)


def extract_features(press_times: list[float], release_times: list[float]) -> list[float]:
    """
    สกัด 31 ฟีเจอร์จาก timestamp ของการกด/ปล่อยปุ่ม 11 คีย์
    - Hold (H): release - press
    - Down-Down (DD): press[N+1] - press[N]
    - Up-Down (UD): press[N+1] - release[N]
    """
    if len(press_times) != NUM_KEYS or len(release_times) != NUM_KEYS:
        raise ValueError(
            f"ต้องการ timestamp {NUM_KEYS} คีย์ แต่ได้ press={len(press_times)}, "
            f"release={len(release_times)}"
        )

    hold = [release_times[i] - press_times[i] for i in range(NUM_KEYS)]
    dd = [press_times[i + 1] - press_times[i] for i in range(NUM_DD)]
    ud = [press_times[i + 1] - release_times[i] for i in range(NUM_UD)]
    return hold + dd + ud


def split_feature_groups(features: list[float]) -> tuple[list[float], list[float], list[float]]:
    """แยกฟีเจอร์เป็นกลุ่ม hold, down-down, up-down."""
    hold = features[:NUM_HOLD]
    dd = features[NUM_HOLD:NUM_HOLD + NUM_DD]
    ud = features[NUM_HOLD + NUM_DD:]
    return hold, dd, ud


def average_hold_and_dd(features: list[float]) -> tuple[float, float]:
    """คืนค่าเฉลี่ย Hold Time และ Down-Down Time ของรอบนั้น."""
    hold, dd, _ = split_feature_groups(features)
    return sum(hold) / len(hold), sum(dd) / len(dd)


def has_anomalous_delay(
    features: list[float],
    threshold: float = ANOMALY_THRESHOLD_SEC,
) -> bool:
    """True ถ้ามีค่า DD หรือ UD ใดตัวหนึ่งสูงเกิน threshold."""
    _, dd, ud = split_feature_groups(features)
    return max(dd) > threshold or max(ud) > threshold


def has_negative_timing(features: list[float]) -> bool:
    """True ถ้ามีค่า timing ใดตัวหนึ่งติดลบ."""
    return any(value < 0 for value in features)


def count_data_rows(csv_path: Path) -> int:
    """นับจำนวน data rows ในไฟล์ CSV per-subject."""
    if not csv_path.exists():
        return 0

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            return sum(1 for _ in csv.DictReader(fh))
    except (OSError, csv.Error):
        return 0


def load_subject_label_map(csv_path: Path) -> dict[str, int]:
    """โหลด mapping Subject_Name → Label จากไฟล์ CSV per-subject."""
    mapping: dict[str, int] = {}
    if not csv_path.exists():
        return mapping

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                name = normalize_subject_name(row.get("Subject_Name") or "")
                if not name:
                    continue
                try:
                    mapping[name] = int(row["Label"])
                except (KeyError, TypeError, ValueError):
                    continue
    except (OSError, csv.Error):
        pass
    return mapping


def load_all_subject_label_maps(raw_dir: Path) -> dict[str, int]:
    """สแกน raw_data_*.csv ทั้งหมดใน RAW_DATA_DIR."""
    mapping: dict[str, int] = {}
    if not raw_dir.is_dir():
        return mapping

    for path in sorted(raw_dir.glob("raw_data_*.csv")):
        mapping.update(load_subject_label_map(path))
    return mapping


class KeystrokeDataWriter:
    """จัดการการอ่าน/เขียน/ลบข้อมูลในไฟล์ CSV per-subject."""

    def __init__(self, raw_data_dir: Path) -> None:
        self.raw_data_dir = raw_data_dir
        self.header = build_csv_header()

    def read_data_rows(self, csv_path: Path) -> list[dict[str, str]]:
        """อ่าน data rows ทั้งหมดจาก CSV (ไม่รวม header)."""
        if not csv_path.exists():
            return []

        try:
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                return list(csv.DictReader(fh))
        except (OSError, csv.Error):
            return []

    def can_undo_for_subject(self, subject_name: str) -> bool:
        """True เมื่อไฟล์ของ subject มี data row อย่างน้อย 1 แถว."""
        subject = normalize_subject_name(subject_name)
        if not subject:
            return False
        csv_path = subject_raw_path(subject)
        return len(self.read_data_rows(csv_path)) > 0

    def undo_last_row_for_subject(self, subject_name: str) -> tuple[bool, str]:
        """ลบแถวสุดท้ายจากไฟล์ raw ของ subject นั้น แล้วเขียนไฟล์กลับ."""
        subject = normalize_subject_name(subject_name)
        if not subject:
            return False, "กรุณากรอก Subject Name"

        csv_path = subject_raw_path(subject)
        rows = self.read_data_rows(csv_path)
        if not rows:
            return False, "ไม่มีข้อมูลให้ลบ"

        if normalize_subject_name(rows[-1].get("Subject_Name") or "") != subject:
            return False, "แถวสุดท้ายไม่ใช่ข้อมูลของ Subject นี้"

        rows.pop()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(self.header)
                for row in rows:
                    writer.writerow([row.get(col, "") for col in self.header])
        except OSError as exc:
            return False, f"ไม่สามารถเขียนไฟล์ได้: {exc}"

        return True, UNDO_SUCCESS_MESSAGE

    def append_row(self, subject_name: str, label: int, features: list[float]) -> None:
        """บันทึกแถวใหม่ — สร้างไฟล์พร้อม header หากยังไม่มี."""
        subject = normalize_subject_name(subject_name)
        csv_path = subject_raw_path(subject)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            if not file_exists:
                writer.writerow(self.header)
            row = [subject, label] + [f"{v:.6f}" for v in features]
            writer.writerow(row)


@dataclass
class KeyEvent:
    """บันทึก press/release ของคีย์หนึ่งครั้งในลำดับการพิมพ์."""

    keysym: str
    press_time: float
    release_time: float | None = None


@dataclass
class KeystrokeRound:
    """สถานะการเก็บข้อมูลของรอบพิมพ์ปัจจุบัน."""

    events: list[KeyEvent] = field(default_factory=list)
    keys_down: set[str] = field(default_factory=set)
    invalidated: bool = False

    def reset(self) -> None:
        self.events.clear()
        self.keys_down.clear()
        self.invalidated = False

    def key_index(self) -> int:
        """ลำดับคีย์ที่กำลังพิมพ์ (0-based) — อิงจำนวน press ที่บันทึกแล้ว."""
        return len(self.events)

    def press_times(self) -> list[float]:
        return [ev.press_time for ev in self.events]

    def release_times(self) -> list[float]:
        """คืน release timestamp เรียงตามลำดับคีย์ (index ตรงกับ press_times)."""
        assert all(ev.release_time is not None for ev in self.events)
        return [ev.release_time for ev in self.events]

    def is_complete(self) -> bool:
        return (
            not self.invalidated
            and len(self.events) == NUM_KEYS
            and all(ev.release_time is not None for ev in self.events)
        )

    def record_release(self, keysym: str, timestamp: float) -> None:
        """จับคู่ release กับ press ล่าสุดของ keysym เดียวกัน (รองรับ overlap typing)."""
        for ev in reversed(self.events):
            if ev.keysym == keysym and ev.release_time is None:
                ev.release_time = timestamp
                return


class KeystrokeCollectorApp:
    """หน้าต่าง GUI หลักสำหรับเก็บข้อมูล Keystroke Dynamics."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Keystroke Dynamics — Data Collection")
        self.root.geometry("560x520")
        self.root.resizable(False, False)

        self.writer = KeystrokeDataWriter(RAW_DATA_DIR)
        self.round = KeystrokeRound()
        self.subject_label_map = load_all_subject_label_maps(RAW_DATA_DIR)

        self._build_ui()
        self._bind_events()
        self._refresh_counter()
        self._update_undo_button_state()

    def _build_ui(self) -> None:
        pad = {"padx": 12, "pady": 6}

        frm = ttk.Frame(self.root, padding=16)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Subject Name:").grid(row=0, column=0, sticky="w", **pad)
        self.subject_var = tk.StringVar(value="Owner")
        self.subject_entry = ttk.Entry(frm, textvariable=self.subject_var, width=30)
        self.subject_entry.grid(row=0, column=1, sticky="ew", **pad)
        self.subject_var.trace_add("write", lambda *_: self._on_subject_changed())

        ttk.Label(frm, text="Label:").grid(row=1, column=0, sticky="w", **pad)
        label_frame = ttk.Frame(frm)
        label_frame.grid(row=1, column=1, sticky="w", **pad)
        self.label_var = tk.IntVar(value=1)
        ttk.Radiobutton(
            label_frame, text="Owner (1)", variable=self.label_var, value=1
        ).pack(side="left", padx=(0, 16))
        ttk.Radiobutton(
            label_frame, text="Impostor (0)", variable=self.label_var, value=0
        ).pack(side="left")

        ttk.Label(frm, text="Target Password:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Label(
            frm,
            text=PASSWORD,
            font=("Consolas", 14, "bold"),
            foreground="#1a5276",
        ).grid(row=2, column=1, sticky="w", **pad)

        tk.Label(
            frm,
            text=FOCUS_WARNING_MESSAGE,
            font=("Segoe UI", 11, "bold"),
            foreground="#922b21",
            wraplength=500,
            justify="left",
        ).grid(row=3, column=0, columnspan=2, sticky="w", **pad)

        ttk.Label(frm, text="Type Password:").grid(row=4, column=0, sticky="w", **pad)
        self.password_var = tk.StringVar()
        self.password_entry = ttk.Entry(
            frm, textvariable=self.password_var, width=30, font=("Consolas", 12)
        )
        self.password_entry.grid(row=4, column=1, sticky="ew", **pad)

        ttk.Label(frm, text="Successful Rounds:").grid(row=5, column=0, sticky="w", **pad)
        counter_frame = ttk.Frame(frm)
        counter_frame.grid(row=5, column=1, sticky="ew", **pad)
        self.counter_var = tk.StringVar(value="0")
        ttk.Label(
            counter_frame,
            textvariable=self.counter_var,
            font=("Segoe UI", 12, "bold"),
        ).pack(side="left")
        self.undo_button = ttk.Button(
            counter_frame,
            text="Undo Last Sample",
            command=self._on_undo_click,
            state="disabled",
        )
        self.undo_button.pack(side="left", padx=(16, 0))

        ttk.Label(frm, text="Last Round Feedback:").grid(row=6, column=0, sticky="w", **pad)
        self.feedback_var = tk.StringVar(value=FEEDBACK_PLACEHOLDER)
        ttk.Label(
            frm,
            textvariable=self.feedback_var,
            font=("Consolas", 11),
        ).grid(row=6, column=1, sticky="w", **pad)

        self.status_var = tk.StringVar(value="พิมพ์รหัสผ่านให้ถูกต้องแล้วกด Enter")
        self.status_label = tk.Label(
            frm,
            textvariable=self.status_var,
            foreground="#555555",
            wraplength=500,
            justify="left",
        )
        self.status_label.grid(row=7, column=0, columnspan=2, sticky="w", **pad)

        frm.columnconfigure(1, weight=1)
        self.password_entry.focus_set()

    def _bind_events(self) -> None:
        self.password_entry.bind("<KeyPress>", self._on_key_press)
        self.password_entry.bind("<KeyRelease>", self._on_key_release)
        self.password_entry.bind("<Control-v>", lambda e: "break")
        self.password_entry.bind("<Control-V>", lambda e: "break")
        self.password_entry.bind("<Shift-Insert>", lambda e: "break")
        self.password_entry.bind("<Button-2>", lambda e: "break")
        self.password_entry.bind("<Button-3>", lambda e: "break")

    def _on_subject_changed(self) -> None:
        self._refresh_counter()
        self._update_undo_button_state()

    def _refresh_counter(self) -> None:
        subject = normalize_subject_name(self.subject_var.get())
        count = count_data_rows(subject_raw_path(subject)) if subject else 0
        self.counter_var.set(str(count))

    def _update_undo_button_state(self) -> None:
        subject = normalize_subject_name(self.subject_var.get())
        state = "normal" if self.writer.can_undo_for_subject(subject) else "disabled"
        self.undo_button.config(state=state)

    def _set_status(self, text: str, color: str = "#555555") -> None:
        self.status_var.set(text)
        self.status_label.config(foreground=color)

    def _update_feedback(self, avg_h: float | None, avg_dd: float | None) -> None:
        if avg_h is None or avg_dd is None:
            self.feedback_var.set(FEEDBACK_PLACEHOLDER)
            return
        self.feedback_var.set(
            f"H (เฉลี่ย): {avg_h:.4f} s | DD (เฉลี่ย): {avg_dd:.4f} s"
        )

    def _on_undo_click(self) -> None:
        subject = normalize_subject_name(self.subject_var.get())
        if not subject:
            messagebox.showwarning("ข้อมูลไม่ครบ", "กรุณากรอก Subject Name")
            return

        success, message = self.writer.undo_last_row_for_subject(subject)
        if success:
            self.subject_label_map = load_all_subject_label_maps(RAW_DATA_DIR)
            self._refresh_counter()
            self._set_status(message, "#1e8449")
        else:
            self._set_status(message, "#c0392b")
            if message == "ไม่มีข้อมูลให้ลบ":
                messagebox.showwarning("ไม่สามารถ Undo ได้", message)

        self._update_undo_button_state()

    def _expected_char(self, index: int) -> str | None:
        """คืนค่าตัวอักษรที่คาดหวัง ณ ตำแหน่ง index; None หมายถึง Enter (คีย์ที่ 11)."""
        if index < len(PASSWORD):
            return PASSWORD[index]
        if index == len(PASSWORD):
            return None
        return ""

    def _invalidate_round(self, message: str = INVALID_MESSAGE) -> None:
        """ยกเลิกรอบปัจจุบัน — ล้าง timestamp, ล้างช่องพิมพ์, แสดงคำเตือน."""
        self.round.reset()
        self.round.invalidated = True
        self.password_var.set("")
        self._set_status(message)
        self.password_entry.focus_set()

    def _discard_invalid_sample(self, message: str) -> None:
        """ทิ้ง sample ที่ไม่ผ่าน QC แล้วให้ผู้ใช้พิมพ์ใหม่."""
        self.round.reset()
        self.password_var.set("")
        self._update_feedback(None, None)
        self._set_status(message, "#c0392b")
        self.password_entry.focus_set()

    def _on_key_press(self, event: tk.Event) -> str | None:
        keysym = event.keysym

        if keysym in MODIFIER_KEYS:
            return None

        if self.round.invalidated:
            if keysym not in ("BackSpace", "Return"):
                self.round.invalidated = False
            else:
                return "break"

        if keysym == "BackSpace":
            self._invalidate_round()
            return "break"

        if keysym in self.round.keys_down:
            return "break"

        key_idx = self.round.key_index()
        if key_idx >= NUM_KEYS:
            self._invalidate_round("พิมพ์เกินจำนวนที่กำหนด — เริ่มใหม่")
            return "break"

        expected = self._expected_char(key_idx)
        if expected is None:
            if keysym != "Return":
                self._invalidate_round()
                return "break"
        else:
            char = event.char
            if not char or char != expected:
                self._invalidate_round()
                return "break"

        self.round.keys_down.add(keysym)
        self.round.events.append(KeyEvent(keysym=keysym, press_time=time.perf_counter()))
        self._set_status(f"กำลังบันทึก... ({key_idx + 1}/{NUM_KEYS} คีย์)")
        return None

    def _on_key_release(self, event: tk.Event) -> str | None:
        keysym = event.keysym

        if keysym in MODIFIER_KEYS:
            return None

        if self.round.invalidated:
            self.round.keys_down.discard(keysym)
            return "break"

        if keysym not in self.round.keys_down:
            return "break"

        self.round.keys_down.discard(keysym)
        self.round.record_release(keysym, time.perf_counter())

        if keysym == "Return" and self.round.is_complete():
            self._finalize_round()

        return None

    def _finalize_round(self) -> None:
        """ตรวจสอบความถูกต้อง คำนวณฟีเจอร์ และบันทึกลง CSV."""
        subject = normalize_subject_name(self.subject_var.get())
        if not subject:
            messagebox.showwarning("ข้อมูลไม่ครบ", "กรุณากรอก Subject Name")
            self._invalidate_round("กรุณากรอก Subject Name ก่อนพิมพ์")
            return

        label = int(self.label_var.get())
        existing_label = self.subject_label_map.get(subject)
        if existing_label is not None and existing_label != label:
            messagebox.showwarning("Label ไม่สอดคล้อง", LABEL_MISMATCH_MESSAGE)
            self._invalidate_round(LABEL_MISMATCH_MESSAGE)
            return

        typed = self.password_var.get()
        if typed != PASSWORD:
            self._invalidate_round("รหัสผ่านไม่ตรง — เริ่มพิมพ์ใหม่")
            return

        if not self.round.is_complete():
            self._invalidate_round("ข้อมูล timing ไม่ครบ — เริ่มพิมพ์ใหม่")
            return

        try:
            features = extract_features(self.round.press_times(), self.round.release_times())
        except ValueError as exc:
            self._invalidate_round(str(exc))
            return

        if len(features) != NUM_FEATURES:
            self._invalidate_round("จำนวนฟีเจอร์ไม่ถูกต้อง — เริ่มพิมพ์ใหม่")
            return

        if has_negative_timing(features):
            self._discard_invalid_sample(NEGATIVE_TIMING_MESSAGE)
            return

        if has_anomalous_delay(features):
            self._discard_invalid_sample(ANOMALY_MESSAGE)
            return

        self.writer.append_row(subject, label, features)
        self.subject_label_map[subject] = label

        avg_h, avg_dd = average_hold_and_dd(features)
        self._update_feedback(avg_h, avg_dd)
        self._set_status(SUCCESS_MESSAGE, "#1e8449")

        self.round.reset()
        self.password_var.set("")
        self._refresh_counter()
        self._update_undo_button_state()
        self.password_entry.focus_set()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    root = tk.Tk()
    app = KeystrokeCollectorApp(root)
    app.run()


if __name__ == "__main__":
    main()
