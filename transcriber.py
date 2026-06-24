#!/usr/bin/env python3
"""
Audio Transcriber — A minimal, fully-offline desktop transcription app.

Uses Faster-Whisper (large-v3) to transcribe audio files locally.
No internet required after the initial model download.
No API keys. No cloud services. No recurring costs.
"""

import os
import sys
import threading
from pathlib import Path
import time
import traceback as _traceback
import numpy as np

# ---------------------------------------------------------------------------
#  Startup log — always write to ~/Library/Logs/AudioTranscriber.log
#  so we can diagnose crashes when launched from Finder.
# ---------------------------------------------------------------------------
_LOG_PATH = Path.home() / "Library" / "Logs" / "AudioTranscriber.log"

def _log(msg: str):
    """Append a line to the app log file (create/truncate at startup)."""
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass  # logging must never crash the app

# Log the very first thing – if we get here, at least the process started.
_log("=== Audio Transcriber starting ===")
_log(f"sys.executable: {sys.executable}")
_log(f"sys.path: {sys.path[:6]}")
_log(f"cwd: {os.getcwd()}")
_log(f"HOME: {os.environ.get('HOME', '(not set)')}")
_log(f"DISPLAY: {os.environ.get('DISPLAY', '(not set)')}")

# ---------------------------------------------------------------------------
#  Pre-load Tcl/Tk dylibs (PyInstaller bundle fix for Finder launch)
#
#  When bundled as a macOS .app, _tkinter.so references libtcl8.6.dylib,
#  libtk8.6.dylib via @rpath — but the bootloader executable has no
#  LC_RPATH.  By pre-loading them with ctypes BEFORE importing tkinter,
#  dyld finds them in its loaded-image cache instead of trying to
#  resolve @rpath on disk.
#
#  libtcl8.6.dylib itself needs @rpath/libz.1.dylib, so we load that
#  first.
# ---------------------------------------------------------------------------
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    _log("PyInstaller bundle detected — pre-loading Tcl/Tk dylibs…")
    _res_dir = sys._MEIPASS  # Contents/Resources/ for a .app bundle
    for _lib in ("libz.1.dylib", "libtcl8.6.dylib", "libtk8.6.dylib"):
        _p = os.path.join(_res_dir, _lib)
        if os.path.isfile(_p):
            try:
                import ctypes
                ctypes.cdll.LoadLibrary(_p)
                _log(f"  ✓ {_lib}")
            except Exception as _e:
                _log(f"  ✗ {_lib}: {_e}")
        else:
            _log(f"  ✗ {_lib} — file not found at {_p}")
else:
    _log("Not a PyInstaller bundle (no sys._MEIPASS)")

_log("Importing tkinter…")
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ---------------------------------------------------------------------------
#  Faster-Whisper import (handled gracefully if missing)
# ---------------------------------------------------------------------------
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

# ---------------------------------------------------------------------------
#  Diagnostics: log model / cache paths at import time
# ---------------------------------------------------------------------------
_cache_home = os.path.expanduser("~/.cache/huggingface")
_hub_dir = os.path.join(_cache_home, "hub")
_model_id = "models--Systran--faster-whisper-large-v3"
_model_cache_dir = os.path.join(_hub_dir, _model_id)

def _log_model_diagnostics():
    """Log details about the Whisper model cache so we can diagnose load failures."""
    _log(f"cache home: {_cache_home}")
    _log(f"cache exists: {os.path.isdir(_cache_home)}")
    _log(f"cache size: {_dir_size(_cache_home)}")
    if os.path.isdir(_model_cache_dir):
        snap_dir = os.path.join(_model_cache_dir, "snapshots")
        if os.path.isdir(snap_dir):
            snaps = os.listdir(snap_dir)
            _log(f"model snapshots: {snaps}")
            if snaps:
                snap_path = os.path.join(snap_dir, snaps[0])
                for fn in ("model.bin", "tokenizer.json", "config.json", "vocabulary.json", "preprocessor_config.json"):
                    fp = os.path.join(snap_path, fn)
                    sz = _file_size(fp)
                    _log(f"  {fn}: {sz}")
    # Log faster-whisper package assets (silero_vad_v6.onnx etc.)
    try:
        import faster_whisper
        fw_dir = os.path.dirname(faster_whisper.__file__)
        assets_dir = os.path.join(fw_dir, "assets")
        if os.path.isdir(assets_dir):
            for afn in os.listdir(assets_dir):
                afp = os.path.join(assets_dir, afn)
                if os.path.isfile(afp):
                    _log(f"fw asset: {afn} ({os.path.getsize(afp)} bytes)")
    except Exception:
        pass

def _file_size(path: str) -> str:
    try:
        sz = os.path.getsize(path)
        if sz > 1_000_000_000:
            return f"{sz / 1_000_000_000:.2f} GB"
        if sz > 1_000_000:
            return f"{sz / 1_000_000:.1f} MB"
        return f"{sz:,} bytes"
    except Exception:
        return "NOT FOUND"

def _dir_size(path: str) -> str:
    try:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                try:
                    total += os.path.getsize(fp)
                except Exception:
                    pass
        if total > 1_000_000_000:
            return f"{total / 1_000_000_000:.2f} GB"
        return f"{total:,} bytes"
    except Exception:
        return "?"

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
APP_TITLE = "Audio Transcriber"
APP_SIZE = "820x660"
MODEL_SIZE = "large-v3"         # ~3 GB download on first run, cached locally

SUPPORTED_EXTENSIONS = (
    ".mp3", ".mp4", ".wav", ".m4a", ".aac",
    ".flac", ".ogg", ".webm"
)

SUPPORTED_FORMATS_STR = "Audio / Video files (*.mp3 *.mp4 *.wav *.m4a *.aac *.flac *.ogg *.webm)"

CONFIDENCE_THRESHOLD = 0.4  # segments below this avg_logprob get [unclear] marking

# ---------------------------------------------------------------------------
#  Transcription engine (runs in a worker thread)
# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()


def _load_model(status_callback=None):
    """Load the Faster-Whisper model (thread‑safe, downloaded once)."""
    global _model

    if status_callback:
        status_callback(f"Loading model {MODEL_SIZE} (first run downloads ~3 GB)…")

    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not installed.\n"
            "Run: pip install faster-whisper"
        )

    with _model_lock:
        if _model is None:
            _log("WhisperModel constructor called…")
            _log_model_diagnostics()
            try:
                _model = WhisperModel(
                    MODEL_SIZE,
                    device="cpu",
                    compute_type="int8",   # best speed/accuracy on Apple Silicon CPU
                    cpu_threads=4,
                    num_workers=2,
                )
                _log("WhisperModel constructor OK")
            except Exception as _me:
                _log(f"WhisperModel constructor FAILED: {_me}")
                _log(f"Full traceback:\n{_traceback.format_exc()}")
                raise
    if status_callback:
        status_callback("Model loaded.")


# ---------------------------------------------------------------------------
#  Audio loader — uses PyAV instead of FFmpeg subprocess
#  PyAV is bundled in the standalone app, eliminating the FFmpeg dependency.
# ---------------------------------------------------------------------------

def load_audio(file_path: str) -> np.ndarray:
    """
    Decode any audio/video file to 16 kHz mono float32 using PyAV.

    Returns a numpy array in [-1, 1] range, compatible with Faster-Whisper.
    """
    import av

    _log(f"PyAV opening: {file_path}")
    _log(f"PyAV version: {av.__version__}")

    try:
        container = av.open(file_path)
    except Exception as _e:
        _log(f"PyAV open() FAILED: {_e}")
        _log(f"Traceback:\n{_traceback.format_exc()}")
        raise ValueError(f"Cannot open audio file: {_e}") from _e

    # Find the audio stream
    audio_stream = None
    for stream in container.streams:
        if stream.type == "audio":
            audio_stream = stream
            _log(f"  Audio stream: index={stream.index}  codec={stream.codec_context.name}  "
                 f"sample_rate={stream.sample_rate}  channels={stream.channels}")
            break

    if audio_stream is None:
        container.close()
        raise ValueError(f"No audio stream found in '{file_path}'")

    # Use format='s16' (interleaved int16) with mono + 16 kHz
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=16000,
    )

    audio_stream.thread_type = "AUTO"

    all_frames = []
    try:
        for frame in container.decode(audio=0):
            if frame is None:
                continue
            frame.pts = None  # prevent PTS discontinuity warnings
            out = resampler.resample(frame)
            # In newer PyAV, resample() may return a list of AudioFrames
            frames_out = out if isinstance(out, (list, tuple)) else [out]
            for f_out in frames_out:
                if f_out is not None:
                    arr = f_out.to_ndarray()
                    if arr.size > 0:
                        all_frames.append(arr)
    except Exception as _de:
        _log(f"PyAV decode error: {_de}")
        _log(f"Traceback:\n{_traceback.format_exc()}")
        # Some files may have decode errors mid-stream; collect what we can
    finally:
        container.close()

    if not all_frames:
        raise ValueError(f"No audio data could be decoded from '{file_path}'")

    audio = np.concatenate(all_frames, axis=None)
    # Remove DC offset
    audio = audio - np.mean(audio)
    # Convert int16 → float32 in [-1, 1]
    audio = audio.astype(np.float32) / 32768.0

    duration = len(audio) / 16000.0
    _log(f"Audio loaded: {len(audio)} samples, {duration:.1f}s, "
         f"range=[{audio.min():.4f}, {audio.max():.4f}]")

    return audio


def transcribe_audio(
    file_path: str,
    progress_callback=None,
    status_callback=None,
) -> dict:
    """
    Run transcription on *file_path* inside a worker thread.

    Uses PyAV to decode audio (supports MP3, MP4, WAV, M4A, AAC, FLAC,
    OGG, WEBM — and any format PyAV can handle).

    Returns a dict with keys:
        text          – formatted transcription
        raw_segments  – list of segment dicts (for debugging / confidence)
        language      – detected language code
        language_prob – detection confidence
        duration      – audio duration in seconds
    """
    # --- Step 1: Load audio with PyAV ---
    _log(f"Loading audio from: {file_path}")
    ext = Path(file_path).suffix.lower()
    _log(f"File extension: {ext}")
    _log(f"File size: {_file_size(file_path)}")

    if status_callback:
        status_callback("Decoding audio…")

    try:
        audio = load_audio(file_path)
        _log(f"Audio loaded successfully: {len(audio)} samples")
    except Exception as _ae:
        _log(f"Audio loading FAILED: {_ae}")
        _log(f"Full traceback:\n{_traceback.format_exc()}")
        raise

    # --- Step 2: Ensure model is loaded ---
    _load_model(status_callback)

    if status_callback:
        status_callback("Transcribing…")

    # --- Step 3: Transcribe ---
    with _model_lock:
        _log("model.transcribe() called (numpy array input)…")
        try:
            segments, info = _model.transcribe(
                audio,                  # pass numpy array directly
                beam_size=5,
                best_of=5,
                temperature=0.0,          # deterministic, highest accuracy
                word_timestamps=False,    # we don't need per-word data
                vad_filter=True,          # skip silence
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    threshold=0.5,
                ),
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
            )
            _log("model.transcribe() returned OK")
        except Exception as _te:
            _log(f"model.transcribe() FAILED: {_te}")
            _log(f"Full traceback:\n{_traceback.format_exc()}")
            raise

    # Collect results
    collected_segments = []

    for seg in segments:
        seg_dict = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
            "confidence": seg.avg_logprob,  # alias for clarity
        }
        collected_segments.append(seg_dict)

        # Update progress roughly based on time consumed
        if progress_callback:
            dur = getattr(info, "duration", 1.0) or 1.0
            pct = min(95, int((seg.end / max(dur, 1)) * 100))
            progress_callback(pct)

    if status_callback:
        status_callback("Formatting transcript…")

    # Build the final text with paragraph formatting
    formatted = _format_transcript(collected_segments)

    duration = getattr(info, "duration", 0.0) or (len(audio) / 16000.0)
    language = getattr(info, "language", "unknown") or "unknown"
    language_prob = getattr(info, "language_probability", 0.0) or 0.0

    _log(f"Transcription complete: lang={language}, prob={language_prob:.2%}, "
         f"duration={duration:.1f}s, segments={len(collected_segments)}")

    return {
        "text": formatted,
        "raw_segments": collected_segments,
        "language": language,
        "language_prob": language_prob,
        "duration": duration,
    }


def _format_transcript(segments: list) -> str:
    """
    Join segments into paragraphs, respecting natural breaks.

    Low‑confidence segments are prefixed with ``[unclear]``.
    """
    lines = []
    buffer = []
    MAX_SENTENCES_PER_PARA = 5

    def flush_buffer():
        if buffer:
            lines.append(" ".join(buffer))
            buffer.clear()

    for seg in segments:
        text = seg["text"]

        # Mark low‑confidence segments
        if seg.get("avg_logprob", 0) is not None and seg["avg_logprob"] < CONFIDENCE_THRESHOLD:
            text = f"[unclear] {text}"

        # Heuristic: very short segments (< 1 s) are likely continuations
        duration = seg["end"] - seg["start"]
        is_new_paragraph = (
            seg.get("no_speech_prob", 0) is not None
            and seg["no_speech_prob"] > 0.8
        )

        buffer.append(text)

        if is_new_paragraph or len(buffer) >= MAX_SENTENCES_PER_PARA:
            flush_buffer()
            lines.append("")  # blank line between paragraphs

    flush_buffer()

    # Remove trailing blank lines
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
#  GUI
# ---------------------------------------------------------------------------
class TranscriberApp:

    _last_reported_pct: int = -1

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(APP_SIZE)
        self.root.minsize(640, 520)

        # Prevent the window from being closed while transcribing
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.file_path: str | None = None
        self.transcription_text: str = ""
        self.is_transcribing = False
        self.worker_thread: threading.Thread | None = None

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self):
        # Use a themed style for a more modern look
        style = ttk.Style()
        style.theme_use("clam")

        # --- Top frame: file selection ---
        top_frame = ttk.Frame(self.root, padding=12)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Audio Transcriber", font=("Helvetica", 18, "bold")).pack(
            anchor=tk.W, pady=(0, 2)
        )
        ttk.Label(
            top_frame,
            text="Local transcription using Faster‑Whisper — fully offline after setup.",
            foreground="#666",
            font=("Helvetica", 10),
        ).pack(anchor=tk.W, pady=(0, 10))

        # Row: select button + file label
        select_row = ttk.Frame(top_frame)
        select_row.pack(fill=tk.X)

        self.select_btn = ttk.Button(
            select_row, text="Select Audio", command=self._select_file
        )
        self.select_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.file_label = ttk.Label(
            select_row, text="No file selected", foreground="#888"
        )
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Status / progress ---
        progress_frame = ttk.Frame(self.root, padding=(12, 0, 12, 4))
        progress_frame.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(
            progress_frame, textvariable=self.status_var, font=("Helvetica", 9)
        )
        status_label.pack(anchor=tk.W)

        self.progress = ttk.Progressbar(
            progress_frame, mode="determinate", length=200
        )
        self.progress.pack(fill=tk.X, pady=(4, 0))

        # --- Language detection info ---
        self.lang_var = tk.StringVar(value="")
        lang_label = ttk.Label(
            progress_frame, textvariable=self.lang_var,
            font=("Helvetica", 9), foreground="#555"
        )
        lang_label.pack(anchor=tk.W, pady=(2, 0))

        # --- Action buttons row ---
        action_frame = ttk.Frame(self.root, padding=(12, 4, 12, 4))
        action_frame.pack(fill=tk.X)

        self.transcribe_btn = ttk.Button(
            action_frame,
            text="Transcribe",
            command=self._start_transcription,
            style="Accent.TButton",
        )
        self.transcribe_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.copy_btn = ttk.Button(
            action_frame,
            text="Copy Transcript",
            command=self._copy_text,
            state=tk.DISABLED,
        )
        self.copy_btn.pack(side=tk.LEFT, padx=6)

        self.save_btn = ttk.Button(
            action_frame,
            text="Save Transcript",
            command=self._save_text,
            state=tk.DISABLED,
        )
        self.save_btn.pack(side=tk.LEFT, padx=6)

        # --- Transcript text area ---
        text_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=("Helvetica", 11),
            relief=tk.FLAT,
            borderwidth=0,
            padx=8,
            pady=8,
            bg="#fafafa",
            fg="#222",
            state=tk.DISABLED,
        )
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.configure(yscrollcommand=scrollbar.set)

        # Give the Transcribe button a distinct accent colour
        style.configure(
            "Accent.TButton",
            font=("Helvetica", 10, "bold"),
            foreground="white",
            background="#2563eb",
            bordercolor="#1d4ed8",
            lightcolor="#3b82f6",
            darkcolor="#1e40af",
        )
        style.map(
            "Accent.TButton",
            background=[("disabled", "#94a3b8")],
            foreground=[("disabled", "#e2e8f0")],
        )

        # Keyboard shortcuts
        self.root.bind("<Command-o>", lambda _: self._select_file())
        self.root.bind("<Command-t>", lambda _: self._start_transcription())
        self.root.bind("<Command-c>", lambda _: self._copy_text())
        self.root.bind("<Command-s>", lambda _: self._save_text())

    # ── File selection ─────────────────────────────────────────────────

    def _select_file(self):
        if self.is_transcribing:
            return

        path = filedialog.askopenfilename(
            title="Select an audio file",
            filetypes=[
                (SUPPORTED_FORMATS_STR, " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS)),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        ext = Path(path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            messagebox.showerror(
                "Unsupported format",
                f"File type '{ext}' is not supported.\n\n"
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
            )
            return

        self.file_path = path
        self.file_label.config(text=Path(path).name, foreground="#222")
        self.status_var.set("Ready — press Transcribe to start")
        self.lang_var.set("")

    # ── Transcription (worker thread) ──────────────────────────────────

    def _start_transcription(self):
        if self.is_transcribing:
            return
        if not self.file_path:
            messagebox.showinfo("No file", "Please select an audio file first.")
            return

        self.is_transcribing = True
        self._set_ui_busy(True)

        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert("1.0", "Transcribing… please wait.\n(This may take a few minutes for long files.)")
        self.text_widget.config(state=tk.DISABLED)

        self.progress["value"] = 0
        self._last_reported_pct = -1
        self.worker_thread = threading.Thread(
            target=self._transcribe_worker,
            args=(self.file_path,),
            daemon=True,
        )
        self.worker_thread.start()

    def _transcribe_worker(self, file_path: str):
        try:
            result = transcribe_audio(
                file_path,
                progress_callback=self._on_progress,
                status_callback=self._on_status,
            )
            self.root.after(0, self._on_transcription_done, result)
        except Exception as exc:
            self.root.after(0, self._on_transcription_error, exc)

    def _on_progress(self, pct: int):
        """Called from worker thread — schedule UI update on main thread."""
        # Debounce: skip if the rounded value hasn't changed more than 5%
        rounded = (pct // 5) * 5
        if rounded <= self._last_reported_pct:
            return
        self._last_reported_pct = rounded
        self.root.after_idle(lambda: self._update_progress_ui(pct))

    def _update_progress_ui(self, pct: int):
        self.progress["value"] = pct
        self.status_var.set(f"Transcribing… {pct}%")

    def _on_status(self, msg: str):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _on_transcription_done(self, result: dict):
        self.progress.stop()
        self.is_transcribing = False
        self._set_ui_busy(False)

        # Update language info
        lang = result["language"]
        prob = result.get("language_prob", 0)
        dur = result.get("duration", 0)
        prob_pct = f"{prob * 100:.0f}%" if prob else "—"
        dur_str = f"{dur:.1f}s" if dur else "—"
        self.lang_var.set(f"Language: {lang} ({prob_pct})  |  Duration: {dur_str}")

        # Display text
        text = result["text"]
        self.transcription_text = text

        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert("1.0", text if text else "(empty transcript)")
        self.text_widget.config(state=tk.DISABLED)

        if text:
            self.copy_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            self.status_var.set("Transcription complete.")
        else:
            self.status_var.set("Transcription complete — no speech detected.")
            self.lang_var.set("")

    def _on_transcription_error(self, exc: Exception):
        self.progress.stop()
        self.is_transcribing = False
        self._set_ui_busy(False)

        err_msg = str(exc)
        self.status_var.set(f"Error: {err_msg}")

        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(
            "1.0",
            f"Transcription failed.\n\n{err_msg}",
        )
        self.text_widget.config(state=tk.DISABLED)

        messagebox.showerror("Transcription Error", err_msg)

    # ── Copy / Save ────────────────────────────────────────────────────

    def _copy_text(self):
        if not self.transcription_text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.transcription_text)
        self.status_var.set("Copied to clipboard.")

    def _save_text(self):
        if not self.transcription_text:
            return

        initial_name = ""
        if self.file_path:
            initial_name = Path(self.file_path).stem

        path = filedialog.asksaveasfilename(
            title="Save transcript as…",
            initialfile=f"{initial_name}_transcript.txt",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.transcription_text)
            self.status_var.set(f"Saved to {Path(path).name}")
        except OSError as exc:
            messagebox.showerror("Save error", str(exc))

    # ── UI helpers ─────────────────────────────────────────────────────

    def _set_ui_busy(self, busy: bool):
        state = tk.DISABLED if busy else tk.NORMAL
        self.select_btn.config(state=state)
        self.transcribe_btn.config(state=state)

        # Keep copy/save enabled only when we have text
        if not busy and not self.transcription_text:
            self.copy_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)

    def _on_close(self):
        if self.is_transcribing:
            if not messagebox.askyesno(
                "Quit?", "Transcription is in progress. Quit anyway?"
            ):
                return
        self.root.destroy()


# ---------------------------------------------------------------------------    # Startup log: write right after tkinter import so we know it worked
# ---------------------------------------------------------------------------
_log(f"tkinter imported OK: {tk.Tcl().eval('info patchlevel')}")

# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------
def main():
    import multiprocessing as _mp
    _mp.freeze_support()
    _log("main() called, freeze_support() done")

    if WhisperModel is None:
        msg = "faster-whisper is not installed. Run: pip install faster-whisper"
        _log(f"FATAL: {msg}")
        print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    # Log diagnostics before creating the UI
    _log_model_diagnostics()

    try:
        _log("Creating Tk root window…")
        root = tk.Tk()
        _log("Tk root window created")
        _log(f"Tk display: {root.tk.call('tk', 'windowingsystem')}")
        app = TranscriberApp(root)
        _log("TranscriberApp initialized, entering mainloop…")
        root.mainloop()
        _log("mainloop exited (app closed)")
    except Exception as exc:
        _log(f"FATAL startup error: {exc}")
        import traceback
        _log(traceback.format_exc())
        raise


if __name__ == "__main__":
    _log("__name__ == '__main__', calling main()")
    main()
