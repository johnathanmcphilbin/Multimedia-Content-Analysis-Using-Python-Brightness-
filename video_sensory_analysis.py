
import os
import sys
import threading
import queue
import time
import math
import tkinter as tk
from tkinter import filedialog, messagebox, StringVar, IntVar, DoubleVar
try:
    from TkinterDnD2 import DND_FILES, TkinterDnD
    DND_OK = True
except Exception:
    DND_FILES = None
    TkinterDnD = None
    DND_OK = False
try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import *
except Exception:
    tb = None
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class App:
    def __init__(self):
        if tb:
            self.root = tb.Window(themename="darkly")
        else:
            if DND_OK:
                self.root = TkinterDnD.Tk()
            else:
                self.root = tk.Tk()
        self.root.title("Video Analysis: Brightness, Blue Light, Sharpness")
        self.video_path = StringVar(value="")
        self.frame_skip = IntVar(value=3)
        self.start_time = DoubleVar(value=0.0)
        self.end_time = DoubleVar(value=0.0)
        self.status = StringVar(value="Ready")
        self.playback_speed = DoubleVar(value=1.0)
        self.is_playing = False
        self.stop_playback = threading.Event()
        self.analysis_thread = None
        self.play_thread = None
        self.metrics_df = None
        self.cap = None
        self.fps = 0.0
        self.duration = 0.0
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        style = tb.Style() if tb else None
        frm = (tb.Frame if tb else tk.Frame)(self.root, padding=10)
        frm.pack(fill="both", expand=True)
        top = (tb.Labelframe if tb else tk.LabelFrame)(frm, text="Video", padding=10)
        top.pack(fill="x")
        ent = (tb.Entry if tb else tk.Entry)(top, textvariable=self.video_path)
        ent.pack(side="left", fill="x", expand=True, padx=(0,8))
        (tb.Button if tb else tk.Button)(top, text="Browse", command=self.browse).pack(side="left")
        if DND_OK:
            ent.drop_target_register(DND_FILES)
            ent.dnd_bind("<<Drop>>", self.on_drop)
        opts = (tb.Labelframe if tb else tk.LabelFrame)(frm, text="Options", padding=10)
        opts.pack(fill="x", pady=10)
        make = (tb.LabeledScale if tb else None)
        row = (tb.Frame if tb else tk.Frame)(opts)
        row.pack(fill="x")
        (tb.Label if tb else tk.Label)(row, text="Frame skip (process every n-th frame)").pack(side="left")
        fs = (tb.Spinbox if tb else tk.Spinbox)(row, from_=1, to=120, textvariable=self.frame_skip, width=6)
        fs.pack(side="left", padx=8)
        row2 = (tb.Frame if tb else tk.Frame)(opts)
        row2.pack(fill="x", pady=4)
        (tb.Label if tb else tk.Label)(row2, text="Start time (s)").pack(side="left")
        (tb.Entry if tb else tk.Entry)(row2, textvariable=self.start_time, width=10).pack(side="left", padx=8)
        (tb.Label if tb else tk.Label)(row2, text="End time (s, 0 for end)").pack(side="left")
        (tb.Entry if tb else tk.Entry)(row2, textvariable=self.end_time, width=10).pack(side="left", padx=8)
        row3 = (tb.Frame if tb else tk.Frame)(opts)
        row3.pack(fill="x", pady=4)
        (tb.Label if tb else tk.Label)(row3, text="Playback speed").pack(side="left")
        (tb.Spinbox if tb else tk.Spinbox)(row3, from_=0.25, to=4.0, increment=0.25, textvariable=self.playback_speed, width=6).pack(side="left", padx=8)
        actions = (tb.Frame if tb else tk.Frame)(frm)
        actions.pack(fill="x", pady=6)
        (tb.Button if tb else tk.Button)(actions, text="Analyze", command=self.start_analysis).pack(side="left")
        (tb.Button if tb else tk.Button)(actions, text="Play Overlay", command=self.start_playback).pack(side="left", padx=8)
        (tb.Button if tb else tk.Button)(actions, text="Stop", command=self.stop_all).pack(side="left")
        (tb.Button if tb else tk.Button)(actions, text="Export CSV", command=self.export_csv).pack(side="left", padx=8)
        self.status_label = (tb.Label if tb else tk.Label)(frm, textvariable=self.status, anchor="w")
        self.status_label.pack(fill="x", pady=(0,6))
        plots = (tb.Notebook if tb else tk.Frame)(frm)
        plots.pack(fill="both", expand=True)
        if tb:
            self.tab_lines = tb.Frame(plots)
            self.tab_hist = tb.Frame(plots)
            plots.add(self.tab_lines, text="Time Series")
            plots.add(self.tab_hist, text="Brightness Histogram")
        else:
            self.tab_lines = (tb.Frame if tb else tk.Frame)(plots)
            self.tab_hist = (tb.Frame if tb else tk.Frame)(plots)
            self.tab_lines.pack(side="left", fill="both", expand=True)
            self.tab_hist.pack(side="left", fill="both", expand=True)
        self.fig_lines = plt.Figure(figsize=(8,4), dpi=100)
        self.fig_hist = plt.Figure(figsize=(8,4), dpi=100)
        self.ax_b = self.fig_lines.add_subplot(311)
        self.ax_bl = self.fig_lines.add_subplot(312)
        self.ax_s = self.fig_lines.add_subplot(313)
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.canvas_lines = FigureCanvasTkAgg(self.fig_lines, master=self.tab_lines)
        self.canvas_lines.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=self.tab_hist)
        self.canvas_hist.get_tk_widget().pack(fill="both", expand=True)

    def on_drop(self, event):
        p = event.data
        if self.root.tk.splitlist(p):
            path = self.root.tk.splitlist(p)[0]
            self.video_path.set(path)
            self.load_video_meta()

    def browse(self):
        path = filedialog.askopenfilename(title="Select video", filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov"), ("All files","*.*")])
        if path:
            self.video_path.set(path)
            self.load_video_meta()

    def load_video_meta(self):
        vp = self.video_path.get()
        if not os.path.isfile(vp):
            return
        self.cap = cv2.VideoCapture(vp)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video.")
            return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        self.duration = frames / self.fps if self.fps > 0 else 0.0
        self.status.set(f"Loaded: {os.path.basename(vp)} | {self.fps:.2f} FPS | {self.duration:.2f}s")

    def start_analysis(self):
        if self.analysis_thread and self.analysis_thread.is_alive():
            return
        vp = self.video_path.get().strip('"')
        if not os.path.isfile(vp):
            messagebox.showwarning("Missing", "Choose a valid video file.")
            return
        self.stop_all()
        self.analysis_thread = threading.Thread(target=self._analyze, daemon=True)
        self.analysis_thread.start()

    def _analyze(self):
        try:
            self.status.set("Analyzing...")
            cap = cv2.VideoCapture(self.video_path.get())
            if not cap.isOpened():
                self.status.set("Failed to open video.")
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            dur = total / fps if fps>0 else 0.0
            st = max(0.0, float(self.start_time.get()))
            en = float(self.end_time.get())
            if en <= 0 or en > dur:
                en = dur
            start_frame = int(st * fps) if fps>0 else 0
            end_frame = int(en * fps) if fps>0 else int(total)
            fs = max(1, int(self.frame_skip.get()))
            data_t = []
            data_b = []
            data_bl = []
            data_s = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            fidx = start_frame
            last_update = time.time()
            while fidx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                if ((fidx - start_frame) % fs) == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    b_mean = float(np.mean(gray))
                    blue_mean = float(np.mean(frame[:,:,0]))
                    lap = cv2.Laplacian(gray, cv2.CV_64F)
                    sharp = float(lap.var())
                    t = fidx / fps if fps>0 else 0.0
                    data_t.append(t)
                    data_b.append(b_mean)
                    data_bl.append(blue_mean)
                    data_s.append(sharp)
                    now = time.time()
                    if now - last_update > 0.2:
                        self.status.set(f"Analyzing... t={t:.2f}s, frames {fidx}/{end_frame}")
                        last_update = now
                fidx += 1
            cap.release()
            df = pd.DataFrame({"time_s": data_t, "brightness": data_b, "blue": data_bl, "sharpness": data_s})
            self.metrics_df = df
            self._draw_plots()
            self.status.set(f"Done. Samples: {len(df)} from {st:.2f}s to {en:.2f}s")
        except Exception as e:
            self.status.set(f"Error: {e}")

    def _draw_plots(self):
        if self.metrics_df is None or self.metrics_df.empty:
            return
        df = self.metrics_df
        self.ax_b.clear()
        self.ax_bl.clear()
        self.ax_s.clear()
        self.ax_hist.clear()
        self.ax_b.plot(df["time_s"].values, df["brightness"].values)
        self.ax_b.set_ylabel("Brightness")
        self.ax_bl.plot(df["time_s"].values, df["blue"].values)
        self.ax_bl.set_ylabel("Blue")
        self.ax_s.plot(df["time_s"].values, df["sharpness"].values)
        self.ax_s.set_ylabel("Sharpness")
        self.ax_s.set_xlabel("Time (s)")
        b = df["brightness"].values
        t = df["time_s"].values
        if len(b) > 3:
            peaks, _ = find_peaks(b, distance=max(1,int(len(b)*0.01)))
            if peaks.size>0:
                self.ax_b.scatter(t[peaks], b[peaks])
        self.ax_hist.hist(b, bins=32)
        self.ax_hist.set_xlabel("Brightness")
        self.ax_hist.set_ylabel("Count")
        self.fig_lines.tight_layout()
        self.fig_hist.tight_layout()
        self.canvas_lines.draw()
        self.canvas_hist.draw()

    def start_playback(self):
        if self.play_thread and self.play_thread.is_alive():
            return
        vp = self.video_path.get().strip('"')
        if not os.path.isfile(vp):
            messagebox.showwarning("Missing", "Choose a valid video file.")
            return
        self.stop_playback.clear()
        self.is_playing = True
        self.play_thread = threading.Thread(target=self._play_overlay, daemon=True)
        self.play_thread.start()

    def _play_overlay(self):
        try:
            cap = cv2.VideoCapture(self.video_path.get())
            if not cap.isOpened():
                self.status.set("Failed to open video for playback.")
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            dur_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            st = max(0.0, float(self.start_time.get()))
            en = float(self.end_time.get())
            if en <= 0:
                en = dur_frames / fps if fps>0 else 0.0
            start_frame = int(st * fps) if fps>0 else 0
            end_frame = int(en * fps) if fps>0 else dur_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            delay = int(1000 / fps) if fps>0 else 33
            while not self.stop_playback.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                fpos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if fpos >= end_frame:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                b_mean = float(np.mean(gray))
                blue_mean = float(np.mean(frame[:,:,0]))
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                sharp = float(lap.var())
                t = fpos / fps if fps>0 else 0.0
                text = f"t={t:.2f}s  Brightness={b_mean:.1f}  Blue={blue_mean:.1f}  Sharpness={sharp:.1f}"
                cv2.rectangle(frame, (10,10), (10+int(7+len(text)*10), 50), (0,0,0), -1)
                cv2.putText(frame, text, (16,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow("Playback with Overlay - Press Q to stop", frame)
                key = cv2.waitKey(int(delay / self.playback_speed.get()))
                if key & 0xFF in (ord('q'), 27):
                    break
            cap.release()
            cv2.destroyAllWindows()
            self.is_playing = False
            self.status.set("Playback ended.")
        except Exception as e:
            self.status.set(f"Playback error: {e}")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            self.is_playing = False

    def stop_all(self):
        self.stop_playback.set()
        self.is_playing = False

    def export_csv(self):
        if self.metrics_df is None or self.metrics_df.empty:
            messagebox.showinfo("No data", "Run analysis first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not path:
            return
        try:
            self.metrics_df.to_csv(path, index=False)
            self.status.set(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_close(self):
        self.stop_all()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.root.after(200, self.root.destroy)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    App().run()
