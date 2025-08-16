"""Minimal Tkinter GUI for Age & Emotion Voice Detector (Male Only).

Features:
- Pick an audio file (wav/flac/ogg/mp3)
- Run inference using loaded models
- Display results with guardrail-enforced behavior

Note: This is a minimal interface intended for local testing only.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import sys

# Ensure project root is on sys.path when running as a script from the gui/ folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_utils import run_inference
from gui.gui_helpers import load_default_models, parse_emotion_classes


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Voice Detector – Minimal GUI")
        self.geometry("520x300")

        # State
        self.selected_path_var = tk.StringVar(value="")
        self.emotions_var = tk.StringVar(value="neutral,calm,happy,sad,angry,fear,disgust,surprised")
        self.force_emotion_var = tk.BooleanVar(value=False)

        # Models
        self.models = load_default_models(project_root=Path(__file__).parents[1])

        # Layout
        self._build_widgets()

    def _build_widgets(self):
        padding = {"padx": 10, "pady": 6}

        tk.Label(self, text="Audio file:").grid(row=0, column=0, sticky="w", **padding)
        tk.Entry(self, textvariable=self.selected_path_var, width=48).grid(row=0, column=1, **padding)
        tk.Button(self, text="Browse", command=self.on_browse).grid(row=0, column=2, **padding)

        tk.Label(self, text="Valid emotions (csv):").grid(row=1, column=0, sticky="w", **padding)
        tk.Entry(self, textvariable=self.emotions_var, width=48).grid(row=1, column=1, **padding)
        tk.Checkbutton(self, text="Always predict emotion (for male)", variable=self.force_emotion_var).grid(row=1, column=2, sticky="w", **padding)

        tk.Button(self, text="Run Inference", command=self.on_infer).grid(row=2, column=1, **padding)

        self.output = tk.Text(self, height=10, width=62, state="disabled")
        self.output.grid(row=3, column=0, columnspan=3, **padding)

    def on_browse(self):
        path = filedialog.askopenfilename(
            title="Select audio",
            filetypes=[
                ("Audio Files", "*.wav *.flac *.ogg *.mp3"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.selected_path_var.set(path)

    def on_infer(self):
        audio_path = self.selected_path_var.get()
        if not audio_path:
            messagebox.showwarning("Missing input", "Please select an audio file first.")
            return

        if self.models.gender_bundle is None or self.models.age_model is None or self.models.emotion_bundle is None:
            messagebox.showwarning(
                "Models not loaded",
                "Please place trained models in the 'models/' folder: gender_model.pkl, age_model.(pkl|h5), emotion_model.(pkl|h5)",
            )
            return

        try:
            emotions = parse_emotion_classes(self.emotions_var.get())
            # If forcing emotion, set a very low threshold so any positive age triggers emotion
            age_threshold = -1.0 if self.force_emotion_var.get() else 60.0
            result = run_inference(
                audio_path,
                gender_bundle=self.models.gender_bundle,
                age_model=self.models.age_model,
                emotion_bundle=self.models.emotion_bundle,
                valid_emotions=emotions,
                age_threshold=age_threshold,
            )
            self._render_output(result)
        except Exception as error:  # noqa: BLE001
            messagebox.showerror("Inference error", str(error))

    def _render_output(self, result: dict):
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        lines = []
        gender = result.get("gender", "N/A")
        lines.append(f"Gender: {gender}")
        if gender == "Female":
            # Explicit UX per requirement: reject female voices and prompt for male upload
            reject_msg = "Female voice rejected. Upload male voice."
            lines.append(reject_msg)
            try:
                messagebox.showinfo("Female voice rejected", reject_msg)
            except Exception:
                pass
        elif gender == "Male":
            if "age" in result:
                lines.append(f"Age: {result['age']:.1f}")
            if "emotion" in result:
                lines.append(f"Emotion: {result['emotion']}")
        self.output.insert(tk.END, "\n".join(lines))
        self.output.configure(state="disabled")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()

