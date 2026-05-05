# -*- coding: utf-8 -*-
"""
Document Scanner — Perspective Correction
pip install opencv-python-headless numpy Pillow
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os, sys, threading

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


# ── Perspective transform ─────────────────────────────────────────────────────

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp the quadrilateral region to a flat top-down rectangle."""
    # Removed `order_points`. The `pts` array is already explicitly defined 
    # as [TL, TR, BR, BL] by our UI handles. Trusting the user's layout 
    # avoids degenerate point-duplication bugs on heavily skewed quads.
    rect = pts.reshape(4, 2).astype("float32")
    tl, tr, br, bl = rect
    
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    
    if w < 2 or h < 2:
        raise ValueError("Degenerate quad — corners are too close together.")
        
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
    M   = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h))


def enhance_scan(warped: np.ndarray) -> np.ndarray:
    """Adaptive-threshold to simulate a B&W scan."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    bw   = cv2.adaptiveThreshold(gray, 255,
               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

# ── ML Model ──────────────────────────────────────────────────────────────────

class CornerDetector:
    def __init__(self, model_path="doc_corner_net_float32.tflite"):
        self.interpreter = None
        if not HAS_TF:
            print("TensorFlow not installed. AI Detection disabled.")
            return
        
        try:
            # Try to load GPU delegate first if available
            delegate = tf.lite.experimental.load_delegate('tflite_gpu_delegate.dll')
            self.interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
            print("Loaded TFLite model with GPU delegate.")
        except Exception:
            try:
                # Fallback to CPU
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                print("Loaded TFLite model on CPU.")
            except Exception as e:
                print(f"Failed to load TFLite model: {e}")
                pass
        
        if self.interpreter:
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
    def predict(self, img_bgr):
        if not self.interpreter:
            return None
            
        h_orig, w_orig = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256))
        
        # Normalize according to dataset.py (ImageNet mean/std)
        img_normalized = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_normalized - mean) / std
        
        input_data = np.expand_dims(img_normalized, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        heatmaps = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # (64, 64, 4)
        
        corners = []
        for i in range(4):
            hm = heatmaps[:, :, i]
            # Find the max response in the 64x64 heatmap
            _, _, _, max_loc = cv2.minMaxLoc(hm)
            x, y = max_loc
            # Scale coordinates back to original image size
            x_orig = (x / 64.0) * w_orig
            y_orig = (y / 64.0) * h_orig
            corners.append([x_orig, y_orig])
            
        return np.array(corners, dtype=np.float32)


# ── GUI theme constants ───────────────────────────────────────────────────────

HANDLE_R   = 10
BG         = "#12111A"
PANEL_BG   = "#1D1B2A"
CARD_BG    = "#252338"
TEXT       = "#E8E6FF"
MUTED      = "#7B79A0"
ACCENT     = "#7C6AF7"
ACCENT_DIM = "#5847c4"
SUCCESS    = "#4ECDC4"


# ── Progress overlay ──────────────────────────────────────────────────────────

class ProgressOverlay(tk.Toplevel):
    def __init__(self, parent, title="Working…"):
        super().__init__(parent)
        self.overrideredirect(True)
        self.configure(bg=PANEL_BG)
        self.attributes("-topmost", True)
        parent.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        w, h   = 340, 110
        self.geometry(f"{w}x{h}+{px+(pw-w)//2}+{py+(ph-h)//2}")

        tk.Label(self, text=title, bg=PANEL_BG, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack(pady=(16, 4))
        self.step_var = tk.StringVar(value="Please wait…")
        tk.Label(self, textvariable=self.step_var, bg=PANEL_BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack()
        frm = tk.Frame(self, bg=PANEL_BG)
        frm.pack(fill="x", padx=24, pady=10)
        self.bar = ttk.Progressbar(frm, mode="indeterminate", length=290)
        self.bar.pack(fill="x")
        self.bar.start(12)
        self.update_idletasks()

    def set_step(self, text: str):
        self.step_var.set(text)
        self.update_idletasks()

    def close(self):
        self.bar.stop()
        self.destroy()


# ── Main application ──────────────────────────────────────────────────────────

class DocumentScannerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Document Scanner — Perspective Correction")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(960, 660)

        self.orig_image:   np.ndarray | None = None
        self.display_scale: float = 1.0
        self.corners:      np.ndarray | None = None
        self.dragging_idx: int | None = None
        self.warped_image: np.ndarray | None = None
        self._img_origin   = (0, 0)
        self._src_tk       = None
        self._res_tk       = None

        self.scan_mode = tk.BooleanVar(value=False)
        self.detector = CornerDetector()
        
        self._build_ui()
        self._bind_events()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        bar = tk.Frame(self, bg=BG, pady=10)
        bar.pack(fill="x", padx=16)

        self._btn(bar, "📂  Open Image",  self._open_image).pack(side="left", padx=(0,8))
        self._btn(bar, "✂️  Warp / Scan", self._start_warp,  style="accent").pack(side="left", padx=(0,8))
        self._btn(bar, "💾  Save Result", self._save_result, style="success").pack(side="left")

        tk.Checkbutton(bar, text="B&W Scan", variable=self.scan_mode,
                       bg=BG, fg=TEXT, selectcolor=CARD_BG,
                       activebackground=BG, activeforeground=ACCENT,
                       font=("Segoe UI", 10), pady=4, padx=8,
                       cursor="hand2").pack(side="left", padx=14)

        self.status_var = tk.StringVar(value="Open an image to begin.")
        tk.Label(bar, textvariable=self.status_var,
                 bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(side="right", padx=8)

        paned = tk.PanedWindow(self, orient="horizontal", bg=BG, bd=0,
                               sashwidth=6, sashrelief="flat", sashpad=0)
        paned.pack(fill="both", expand=True, padx=16, pady=(0,8))

        lf = tk.Frame(paned, bg=PANEL_BG)
        paned.add(lf, stretch="always", minsize=400)
        tk.Label(lf, text="Source  (drag corners to adjust)",
                 bg=PANEL_BG, fg=MUTED,
                 font=("Segoe UI", 9, "bold"), pady=8).pack(fill="x", padx=12)
        lw = tk.Frame(lf, bg=CARD_BG)
        lw.pack(fill="both", expand=True, padx=8, pady=(0,8))
        self.src_canvas = tk.Canvas(lw, bg=CARD_BG, cursor="crosshair", highlightthickness=0)
        self.src_canvas.pack(fill="both", expand=True)

        rf = tk.Frame(paned, bg=PANEL_BG)
        paned.add(rf, stretch="always", minsize=400)
        tk.Label(rf, text="Corrected Document",
                 bg=PANEL_BG, fg=MUTED,
                 font=("Segoe UI", 9, "bold"), pady=8).pack(fill="x", padx=12)
        rw = tk.Frame(rf, bg=CARD_BG)
        rw.pack(fill="both", expand=True, padx=8, pady=(0,8))
        self.res_canvas = tk.Canvas(rw, bg=CARD_BG, highlightthickness=0)
        self.res_canvas.pack(fill="both", expand=True)

        tip = tk.Frame(self, bg=PANEL_BG, pady=5)
        tip.pack(fill="x", padx=16, pady=(0,10))
        tk.Label(tip, text="💡  Open image → drag the 4 corner handles to the paper edges → Warp → Save",
                 bg=PANEL_BG, fg=MUTED, font=("Segoe UI", 9)).pack()

    def _btn(self, parent, text, cmd, style="normal"):
        palettes = {
            "normal":    (CARD_BG,  TEXT,      "#2E2B44"),
            "secondary": (PANEL_BG, MUTED,     CARD_BG),
            "accent":    (ACCENT,   "#FFFFFF",  ACCENT_DIM),
            "success":   (SUCCESS,  "#0A1210",  "#37a9a0"),
        }
        bg, fg, abg = palettes[style]
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg=fg, activebackground=abg, activeforeground=fg,
                         font=("Segoe UI", 10, "bold"), bd=0,
                         padx=14, pady=7, cursor="hand2", relief="flat")

    def _bind_events(self):
        self.src_canvas.bind("<ButtonPress-1>",   self._on_press)
        self.src_canvas.bind("<B1-Motion>",       self._on_drag)
        self.src_canvas.bind("<ButtonRelease-1>", self._on_release)
        self.src_canvas.bind("<Configure>",       lambda _e: self._draw_source())
        self.res_canvas.bind("<Configure>",       lambda _e: self._safe_draw_result())

    # ── Load ─────────────────────────────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Select document photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Could not read:\n{path}")
            return
        self.orig_image   = img
        self.warped_image = None
        self.res_canvas.delete("all")
        
        h, w = img.shape[:2]
        pred_corners = self.detector.predict(img) if hasattr(self, 'detector') else None
        
        if pred_corners is not None:
            self.corners = pred_corners
            self._set_status(
                f"Loaded: {os.path.basename(path)}  ({w}×{h}) — AI Document Corners Detected."
            )
        else:
            pad  = 0.10
            self.corners = np.float32([
                [w*pad,     h*pad],
                [w*(1-pad), h*pad],
                [w*(1-pad), h*(1-pad)],
                [w*pad,     h*(1-pad)],
            ])
            self._set_status(
                f"Loaded: {os.path.basename(path)}  ({w}×{h}) — drag the handles to the paper corners."
            )
            
        self._draw_source()

    # ── Canvas draw ───────────────────────────────────────────────────────────

    def _draw_source(self):
        if self.orig_image is None:
            return
        cw = self.src_canvas.winfo_width()
        ch = self.src_canvas.winfo_height()
        if cw < 10 or ch < 10:
            self.after(50, self._draw_source)
            return

        ih, iw    = self.orig_image.shape[:2]
        scale     = min(cw / iw, ch / ih, 1.0)
        self.display_scale = scale
        dw, dh    = max(int(iw*scale), 1), max(int(ih*scale), 1)

        rgb = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((dw, dh), Image.LANCZOS)
        self._src_tk = ImageTk.PhotoImage(pil)

        ox, oy            = (cw-dw)//2, (ch-dh)//2
        self._img_origin  = (ox, oy)
        self.src_canvas.delete("all")
        self.src_canvas.create_image(ox, oy, anchor="nw", image=self._src_tk)

        if self.corners is not None:
            self._draw_overlay()

    def _draw_overlay(self):
        s       = self.display_scale
        ox, oy  = self._img_origin
        pts_c   = [(int(x*s+ox), int(y*s+oy)) for x, y in self.corners]
        flat    = [c for pt in pts_c for c in pt]

        self.src_canvas.create_polygon(*flat, fill=ACCENT, stipple="gray25", outline="")
        self.src_canvas.create_polygon(*flat, fill="", outline=ACCENT, width=2)

        for i, (cx, cy) in enumerate(pts_c):
            col = [ACCENT, SUCCESS, SUCCESS, ACCENT][i]
            lbl = ["TL","TR","BR","BL"][i]
            self.src_canvas.create_oval(
                cx-HANDLE_R, cy-HANDLE_R, cx+HANDLE_R, cy+HANDLE_R,
                fill=col, outline="#FFFFFF", width=2)
            self.src_canvas.create_text(cx, cy, text=lbl,
                fill="#FFFFFF", font=("Segoe UI", 7, "bold"))

    def _safe_draw_result(self):
        if self.warped_image is not None:
            self._draw_result()

    def _draw_result(self):
        if self.warped_image is None:
            return
        cw = self.res_canvas.winfo_width()
        ch = self.res_canvas.winfo_height()
        if cw < 10 or ch < 10:
            self.after(80, self._draw_result)
            return

        wh, ww = self.warped_image.shape[:2]
        scale  = min(cw/ww, ch/wh, 1.0)
        dw, dh = max(int(ww*scale),1), max(int(wh*scale),1)

        rgb = cv2.cvtColor(self.warped_image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((dw, dh), Image.LANCZOS)
        self._res_tk = ImageTk.PhotoImage(pil)

        ox, oy = (cw-dw)//2, (ch-dh)//2
        self.res_canvas.delete("all")
        self.res_canvas.create_image(ox, oy, anchor="nw", image=self._res_tk)
        self.res_canvas.create_text(cw-6, ch-6, text=f"{ww}\u00d7{wh} px",
                                    anchor="se", fill=MUTED, font=("Segoe UI", 8))
        self.res_canvas.update_idletasks()

    # ── Drag ─────────────────────────────────────────────────────────────────

    def _hit_test(self, cx, cy):
        if self.corners is None:
            return None
        s = self.display_scale
        ox, oy = self._img_origin
        best_i, best_d = None, HANDLE_R * 2.5
        for i, (x, y) in enumerate(self.corners):
            d = ((cx-(x*s+ox))**2 + (cy-(y*s+oy))**2)**0.5
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _on_press(self, e):   self.dragging_idx = self._hit_test(e.x, e.y)
    def _on_release(self, _): self.dragging_idx = None

    def _on_drag(self, e):
        if self.dragging_idx is None or self.corners is None:
            return
        ih, iw = self.orig_image.shape[:2]
        ox, oy = self._img_origin
        s      = self.display_scale
        rx = max(0, min(iw-1, (e.x-ox)/s))
        ry = max(0, min(ih-1, (e.y-oy)/s))
        self.corners[self.dragging_idx] = [rx, ry]
        self._draw_source()

    # ── Warp (threaded) ───────────────────────────────────────────────────────

    def _start_warp(self):
        if self.orig_image is None:
            messagebox.showinfo("No Image", "Open an image first."); return
        if self.corners is None:
            messagebox.showinfo("No Corners", "Detect or place corners first."); return

        ov = ProgressOverlay(self, "Processing document…")

        # FIX: Capture all Tkinter state synchronously on the main thread
        # Reading Tkinter variables inside a background thread will cause silent crashes
        use_scan_mode = self.scan_mode.get()
        src_image = self.orig_image
        src_corners = self.corners.copy()

        # Helper to safely update the UI text from the background thread
        def safe_set_step(msg):
            self.after(0, lambda m=msg: ov.set_step(m))

        def worker():
            try:
                safe_set_step("Applying perspective transform…")
                warped = four_point_transform(src_image, src_corners)
                
                if use_scan_mode:
                    safe_set_step("Applying B&W scan enhancement…")
                    warped = enhance_scan(warped)
                    
                # Safe handoff back to main thread
                self.after(0, lambda w=warped: self._finish_warp(ov, w, None))
            except Exception as exc:
                self.after(0, lambda e=exc: self._finish_warp(ov, None, e))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_warp(self, ov: ProgressOverlay, warped: np.ndarray | None, error):
        ov.close()
        if error:
            messagebox.showerror("Warp Error", str(error))
            self._set_status("❌  Warp failed.")
            return
        self.warped_image = warped
        wh, ww = warped.shape[:2]
        self._set_status(f"✅  Done — {ww}\u00d7{wh} px  •  Save with 💾")
        self.after_idle(self._draw_result)

    # ── Save ─────────────────────────────────────────────────────────────────

    def _save_result(self):
        if self.warped_image is None:
            messagebox.showinfo("Nothing to Save", "Warp the document first."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("All","*.*")],
            title="Save corrected document")
        if not path: return
        if cv2.imwrite(path, self.warped_image):
            self._set_status(f"💾  Saved → {os.path.basename(path)}")
        else:
            messagebox.showerror("Save Error", f"Could not write:\n{path}")

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = DocumentScannerApp()
    try:
        ttk.Style(app).theme_use("clam")
    except Exception:
        pass
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            app.orig_image = img
            h, w = img.shape[:2]
            pred_corners = app.detector.predict(img) if hasattr(app, 'detector') else None
            
            if pred_corners is not None:
                app.corners = pred_corners
            else:
                pad  = 0.10
                app.corners = np.float32([
                    [w*pad,     h*pad],
                    [w*(1-pad), h*pad],
                    [w*(1-pad), h*(1-pad)],
                    [w*pad,     h*(1-pad)],
                ])
                
            app.after(200, app._draw_source)
    app.mainloop()