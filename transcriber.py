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
APP_SIZE = "860x700"
MODEL_SIZE = "large-v3"

# Recording constants
LIVE_SAMPLE_RATE = 16000

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
    cancel_check=None,
) -> dict:
    """
    Run transcription on *file_path* inside a worker thread.

    Uses PyAV to decode audio (supports MP3, MP4, WAV, M4A, AAC, FLAC,
    OGG, WEBM — and any format PyAV can handle).

    *cancel_check* is an optional callable that returns True if
    cancellation has been requested.

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

    # Check for cancellation after loading
    if cancel_check and cancel_check():
        _log("Transcription cancelled after audio loading")
        raise RuntimeError("Cancelled")

    # --- Step 2: Ensure model is loaded ---
    _load_model(status_callback)

    if status_callback:
        status_callback("Transcribing…")

    # Check for cancellation before transcribe
    if cancel_check and cancel_check():
        _log("Transcription cancelled before model.transcribe()")
        raise RuntimeError("Cancelled")

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
        # Check for cancellation between segments
        if cancel_check and cancel_check():
            _log("Transcription cancelled during segment collection")
            raise RuntimeError("Cancelled")

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

    # If Hindi was detected, transliterate to Hinglish for readability
    if language and language.startswith("hi"):
        _log("Hindi detected — transliterating to Hinglish")
        formatted = _to_hinglish(formatted)

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
#  Hinglish transliteration — converts Devanagari Hindi text to Roman script
# ---------------------------------------------------------------------------

# Mapping for Devanagari → Roman (ITRANS-like scheme)
_HINGLISH_MAP = {
    # Vowels (independent)
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee',
    'उ': 'u', 'ऊ': 'oo', 'ए': 'e', 'ऐ': 'ai',
    'ओ': 'o', 'औ': 'au', 'अं': 'an', 'अः': 'ah',
    # Consonants
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'ड़': 'd', 'ढ़': 'rh', 'क्ष': 'ksh', 'त्र': 'tr', 'ज्ञ': 'gy',
    # Matras (vowel signs) — applied to previous consonant
    'ा': 'aa', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    # Halant (removes inherent vowel)
    '्': '',
    # Other marks
    'ं': 'n', 'ः': 'h', '।': '.', '॥': '..',
}

# Also map uppercase/lowercase Devanagari numerals
_HINGLISH_NUM = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
}


def _to_hinglish(text: str) -> str:
    """
    Transliterate Devanagari Hindi text to Romanized Hinglish.

    If the text contains Devanagari characters, they are converted
    to a Roman script approximation.  Text that is already in Latin
    script (English) is returned as-is.

    Example: "नमस्ते दोस्तों" → "namaste doston"
    """
    import re as _re

    # Quick check: does this text contain any Devanagari?
    if not _re.search(r'[\u0900-\u097F]', text):
        return text  # No Devanagari, return as-is

    result = []
    i = 0
    while i < len(text):
        ch = text[i]

        # Try two-character sequences first (e.g., क्ष, त्र, अं)
        if i + 1 < len(text):
            pair = text[i:i+2]
            if pair in _HINGLISH_MAP:
                result.append(_HINGLISH_MAP[pair])
                i += 2
                continue

        # Single character
        if ch in _HINGLISH_MAP:
            result.append(_HINGLISH_MAP[ch])
        elif ch in _HINGLISH_NUM:
            result.append(_HINGLISH_NUM[ch])
        else:
            # Pass through spaces, punctuation, English letters unchanged
            result.append(ch)
        i += 1

    # Clean up: remove extra spaces
    romanized = "".join(result)
    romanized = _re.sub(r' +', ' ', romanized).strip()
    return romanized


# ---------------------------------------------------------------------------
#  GUI
# ---------------------------------------------------------------------------
class TranscriberApp:

    _last_reported_pct: int = -1

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(APP_SIZE)
        self.root.minsize(640, 560)

        # Prevent the window from being closed while transcribing
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.file_path: str | None = None
        self.transcription_text: str = ""
        self.is_transcribing = False
        self.worker_thread: threading.Thread | None = None
        self._cancel_requested = False
        self._cancel_gen = 0
        self._worker_gen = 0

        # Live microphone state
        self.is_listening = False
        self.live_thread: threading.Thread | None = None
        self.live_stop = threading.Event()
        self.live_start_time: float = 0.0
        self._timer_job: str | None = None

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self):
        # Use a themed style for a more modern look
        style = ttk.Style()
        style.theme_use("clam")

        # --- Top: title only (no subtitle) ---
        title_frame = ttk.Frame(self.root, padding=(12, 12, 12, 0))
        title_frame.pack(fill=tk.X)

        ttk.Label(
            title_frame, text="Audio Transcriber",
            font=("Helvetica", 18, "bold")
        ).pack(anchor=tk.W)

        # --- Action buttons row 1: Select, Transcribe File, Record ---
        action_row1 = ttk.Frame(self.root, padding=(12, 6, 12, 2))
        action_row1.pack(fill=tk.X)

        self.select_btn = ttk.Button(
            action_row1, text="Select File", command=self._select_file
        )
        self.select_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.file_label = ttk.Label(
            action_row1, text="No file selected", foreground="#888",
            font=("Helvetica", 10)
        )
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.transcribe_btn = ttk.Button(
            action_row1,
            text="Transcribe File",
            command=self._start_transcription,
            style="Accent.TButton",
            width=16,
        )
        self.transcribe_btn.pack(side=tk.LEFT, padx=(6, 6))

        self.record_btn = ttk.Button(
            action_row1,
            text="Record",
            command=self._toggle_record,
            style="Record.TButton",
            width=12,
        )
        self.record_btn.pack(side=tk.LEFT)

        # --- Status / progress ---
        # Row 1: status text (above the bar)
        status_frame = ttk.Frame(self.root, padding=(12, 2, 12, 0))
        status_frame.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            status_frame, textvariable=self.status_var, font=("Helvetica", 9)
        ).pack(anchor=tk.W)

        # Row 2: progress bar + cancel ✕ (no separate % label — percentage is in status text above)
        bar_frame = ttk.Frame(self.root, padding=(12, 1, 12, 2))
        bar_frame.pack(fill=tk.X)

        self.progress = ttk.Progressbar(
            bar_frame, mode="determinate"
        )
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Cancel "✕" — styled as a proper clickable area
        # Fixed width (32px), centered ✕, same height as progress bar, hover effect
        self.cancel_lbl = tk.Label(
            bar_frame,
            text="✕",
            fg="#dc2626",
            font=("Helvetica", 15, "bold"),
            cursor="hand2",
            width=3,
            anchor=tk.CENTER,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            padx=0,
            pady=0,
        )
        # No bg set — inherits from parent frame automatically (no visible box)
        self.cancel_lbl.pack(side=tk.LEFT, padx=(4, 0))
        self.cancel_lbl.bind("<Button-1>", lambda e: self._cancel_transcription())
        # Hover effect: light red background on enter, restore to default on leave
        self._cancel_bg_default = self.cancel_lbl.cget("background")
        self.cancel_lbl.bind("<Enter>", lambda e: self.cancel_lbl.config(bg="#fde8e8"))
        self.cancel_lbl.bind("<Leave>", lambda e: self.cancel_lbl.config(bg=self._cancel_bg_default))
        self.cancel_lbl.pack_forget()  # hidden until transcription starts

        # --- Info row: language + elapsed time ---
        info_frame = ttk.Frame(self.root, padding=(12, 0, 12, 2))
        info_frame.pack(fill=tk.X)

        self.lang_var = tk.StringVar(value="")
        ttk.Label(
            info_frame, textvariable=self.lang_var,
            font=("Helvetica", 9), foreground="#555"
        ).pack(side=tk.LEFT)

        self.timer_var = tk.StringVar(value="")
        ttk.Label(
            info_frame, textvariable=self.timer_var,
            font=("Helvetica", 9), foreground="#2563eb"
        ).pack(side=tk.RIGHT)

        # --- Action buttons row 2: Copy, Save ---
        action_row2 = ttk.Frame(self.root, padding=(12, 2, 12, 4))
        action_row2.pack(fill=tk.X)

        self.copy_btn = ttk.Button(
            action_row2,
            text="Copy Transcript",
            command=self._copy_text,
            state=tk.DISABLED,
            width=16,
        )
        self.copy_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.save_btn = ttk.Button(
            action_row2,
            text="Save Transcript",
            command=self._save_text,
            state=tk.DISABLED,
            width=16,
        )
        self.save_btn.pack(side=tk.LEFT)

        # Separator
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)

        # --- Transcript text area with embedded Clear button ---
        text_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        text_frame.pack(fill=tk.BOTH, expand=True)

        # Inner frame for text + Clear button overlay
        text_inner = tk.Frame(text_frame, bg="#fafafa", highlightthickness=0)
        text_inner.pack(fill=tk.BOTH, expand=True)

        self.text_widget = tk.Text(
            text_inner,
            wrap=tk.WORD,
            font=("Helvetica", 13),
            relief=tk.FLAT,
            borderwidth=0,
            padx=10,
            pady=10,
            bg="#fafafa",
            fg="#222",
            state=tk.DISABLED,
        )
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(text_inner, orient=tk.VERTICAL, command=self.text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.configure(yscrollcommand=scrollbar.set)

        # Clear button below the text widget — always visible (disabled when empty)
        clear_frame = ttk.Frame(text_frame, padding=(0, 4, 0, 0))
        clear_frame.pack(fill=tk.X)

        self.clear_btn = ttk.Button(
            clear_frame,
            text="Clear Transcript",
            command=self._clear_transcript,
            width=16,
            state=tk.DISABLED,
        )
        self.clear_btn.pack(side=tk.RIGHT)

        # --- Button styles (all uniform) ---
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
            background=[("active", "#1d4ed8"), ("!active", "#2563eb"), ("disabled", "#94a3b8")],
            foreground=[("disabled", "#e2e8f0")],
        )
        style.configure(
            "Record.TButton",
            font=("Helvetica", 10, "bold"),
            foreground="white",
            background="#059669",
            bordercolor="#047857",
            lightcolor="#10b981",
            darkcolor="#065f46",
        )
        style.map(
            "Record.TButton",
            background=[("active", "#047857"), ("!active", "#059669"), ("disabled", "#94a3b8")],
            foreground=[("disabled", "#e2e8f0")],
        )
        style.configure(
            "StopRecord.TButton",
            font=("Helvetica", 10, "bold"),
            foreground="white",
            background="#dc2626",
            bordercolor="#b91c1c",
            lightcolor="#ef4444",
            darkcolor="#991b1b",
        )
        style.map(
            "StopRecord.TButton",
            background=[("active", "#b91c1c"), ("!active", "#dc2626"), ("disabled", "#94a3b8")],
            foreground=[("disabled", "#e2e8f0")],
        )
        style.configure(
            "Clear.TButton",
            font=("Helvetica", 9),
        )


        # Keyboard shortcuts
        self.root.bind("<Command-o>", lambda _: self._select_file())
        self.root.bind("<Command-t>", lambda _: self._start_transcription())
        self.root.bind("<Command-c>", lambda _: self._copy_text())
        self.root.bind("<Command-s>", lambda _: self._save_text())
        self.root.bind("<Command-r>", lambda _: self._toggle_record())
        self.root.bind("<Command-Escape>", lambda _: self._toggle_record() if self.is_listening else None)

    # ── Recording (record → save WAV → transcribe via same pipeline) ──

    def _toggle_record(self):
        """Toggle between Record and Stop states."""
        if self.is_listening:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        """Start recording from microphone. Only records — no transcription yet."""
        if self.is_listening or self.is_transcribing:
            return

        self.is_listening = True
        self.live_stop.clear()
        self.record_btn.config(text="Stop", style="StopRecord.TButton")
        self.clear_btn.config(state=tk.DISABLED)
        self.select_btn.config(state=tk.DISABLED)
        self.transcribe_btn.config(state=tk.DISABLED)

        self.live_start_time = time.time()
        self.status_var.set("Recording...")
        self.lang_var.set("")
        self.timer_var.set("")
        _log("Recording started")
        self._start_timer()

        self.live_thread = threading.Thread(
            target=self._record_worker,
            daemon=True,
        )
        self.live_thread.start()

    def _stop_recording(self):
        """Stop recording, save to temp WAV, then transcribe via same pipeline."""
        if not self.is_listening:
            return
        self.is_listening = False
        self.live_stop.set()
        self._stop_timer()
        self.record_btn.config(text="Record", style="Record.TButton")
        self.status_var.set("Processing recording...")
        _log("Recording stopped — saving and transcribing")

    def _record_worker(self):
        """
        Background worker: record all audio to a buffer, then save as WAV
        and transcribe using the same transcribe_audio() pipeline as file uploads.
        """
        import sounddevice as sd
        import queue

        _log("Record worker started")

        try:
            # Check for microphone
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            _log(f"Default input device: {default_input}")
            if default_input is None or default_input < 0:
                raise RuntimeError("No input microphone found.")

            # Buffer all incoming audio
            audio_queue = queue.Queue()

            def _callback(indata, frames, time_info, status):
                if status:
                    _log(f"InputStream status: {status}")
                audio_queue.put(indata.copy())

            stream = sd.InputStream(
                samplerate=LIVE_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=_callback,
            )
            stream.start()
            _log("InputStream started — recording...")

            # Collect all audio until stopped
            all_chunks = []
            try:
                while not self.live_stop.is_set():
                    try:
                        data = audio_queue.get(timeout=0.1)
                        all_chunks.append(data)
                    except queue.Empty:
                        pass
            finally:
                stream.stop()
                stream.close()
                _log(f"InputStream closed — recorded {len(all_chunks)} chunks")

            if not all_chunks:
                _log("No audio recorded")
                self.root.after(0, self._on_recording_error, "No audio recorded.")
                return

            # Concatenate all audio into one array
            recording = np.concatenate(all_chunks, axis=0).flatten()
            duration = len(recording) / LIVE_SAMPLE_RATE
            _log(f"Recording complete: {len(recording)} samples, {duration:.1f}s")

            if duration < 0.5:
                _log("Recording too short, skipping")
                self.root.after(0, self._on_recording_error, "Recording too short (under 0.5s).")
                return

            # Save to temp WAV file
            import struct
            temp_dir = Path.home() / "Library" / "Caches" / "AudioTranscriber"
            temp_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            temp_path = temp_dir / f"recording_{timestamp}.wav"

            # Write WAV file (16-bit PCM, mono, 16kHz)
            recording_int16 = (recording * 32767).astype(np.int16)
            with open(temp_path, "wb") as f:
                # WAV header
                data_size = len(recording_int16) * 2  # 16-bit = 2 bytes per sample
                f.write(b"RIFF")
                f.write(struct.pack("<I", 36 + data_size))
                f.write(b"WAVE")
                f.write(b"fmt ")
                f.write(struct.pack("<I", 16))  # chunk size
                f.write(struct.pack("<H", 1))   # PCM format
                f.write(struct.pack("<H", 1))   # mono
                f.write(struct.pack("<I", LIVE_SAMPLE_RATE))  # sample rate
                f.write(struct.pack("<I", LIVE_SAMPLE_RATE * 2))  # byte rate
                f.write(struct.pack("<H", 2))   # block align
                f.write(struct.pack("<H", 16))  # bits per sample
                f.write(b"data")
                f.write(struct.pack("<I", data_size))
                f.write(recording_int16.tobytes())

            _log(f"WAV saved: {temp_path} ({temp_path.stat().st_size} bytes)")

            # Now transcribe using the SAME pipeline as file uploads
            self.root.after(0, self._on_recording_saved, str(temp_path))

        except Exception as _e:
            _log(f"Record worker error: {_e}")
            _log(f"Traceback:\n{_traceback.format_exc()}")
            err_msg = str(_e)
            if "CoreAudio" in err_msg or "input" in err_msg.lower():
                err_msg = (
                    "Microphone access denied or no microphone found.\n\n"
                    "Please grant microphone permission in\n"
                    "System Settings → Privacy & Security → Microphone"
                )
            self.root.after(0, self._on_recording_error, err_msg)

    def _on_recording_saved(self, wav_path: str):
        """Called after recording is saved. Starts transcription via same pipeline.
        Button states are managed by _start_transcription_with_path."""
        self.status_var.set("Transcribing recording...")
        self.lang_var.set("")
        self.timer_var.set("")

        # Set file_path so the save dialog uses the right name
        self.file_path = wav_path
        self.file_label.config(text=f"Recording: {Path(wav_path).name}", foreground="#059669")

        # Start transcription using the same pipeline as file uploads
        self._start_transcription_with_path(wav_path)

    def _on_recording_error(self, err_msg: str):
        """Called on recording error."""
        self.is_listening = False
        self._stop_timer()
        self.record_btn.config(text="Record", style="Record.TButton")
        self.select_btn.config(state=tk.NORMAL)
        self.transcribe_btn.config(state=tk.NORMAL)
        self.timer_var.set("")
        self.status_var.set(f"Error: {err_msg}")
        messagebox.showerror("Microphone Error", err_msg)

    def _clear_transcript(self):
        """Clear the transcript text area."""
        if self.is_listening or self.is_transcribing:
            return
        self.transcription_text = ""
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.config(state=tk.DISABLED)
        self.copy_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.status_var.set("Transcript cleared.")
        self.lang_var.set("")
        self.clear_btn.config(state=tk.DISABLED)

    # ── Timer ──────────────────────────────────────────────────────────

    def _start_timer(self):
        """Start the elapsed-time timer (ticks every second)."""
        self._update_timer()

    def _stop_timer(self):
        """Stop the elapsed-time timer."""
        if self._timer_job:
            try:
                self.root.after_cancel(self._timer_job)
            except Exception:
                pass
            self._timer_job = None

    def _update_timer(self):
        """Update the elapsed-time display."""
        if not self.is_listening:
            return
        elapsed = int(time.time() - self.live_start_time)
        mins, secs = divmod(elapsed, 60)
        self.timer_var.set(f"⏱  {mins:02d}:{secs:02d}")
        self._timer_job = self.root.after(1000, self._update_timer)

    # ── File selection ─────────────────────────────────────────────────

    def _select_file(self):
        if self.is_transcribing or self.is_listening:
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
        """Start transcription of the user-selected file."""
        if self.is_transcribing:
            return
        if not self.file_path:
            messagebox.showinfo("No file", "Please select an audio file first.")
            return
        self._start_transcription_with_path(self.file_path)

    def _start_transcription_with_path(self, file_path: str):
        """Start transcription for a given path (file or recording)."""
        if self.is_transcribing:
            return

        self.is_transcribing = True
        self._cancel_requested = False
        self._worker_gen = self._cancel_gen
        self._set_ui_busy(True)

        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert("1.0", "Transcribing… please wait.\n(This may take a few minutes for long files.)")
        self.text_widget.config(state=tk.DISABLED)

        self.progress["value"] = 0
        self._last_reported_pct = -1
        self.cancel_lbl.pack(side=tk.LEFT, padx=(2, 0), pady=2)
        self.worker_thread = threading.Thread(
            target=self._transcribe_worker,
            args=(file_path,),
            daemon=True,
        )
        self.worker_thread.start()

    def _transcribe_worker(self, file_path: str):
        try:
            result = transcribe_audio(
                file_path,
                progress_callback=self._on_progress,
                status_callback=self._on_status,
                cancel_check=lambda: self._cancel_requested,
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

    def _cancel_transcription(self):
        """Cancel the current file transcription immediately."""
        if not self.is_transcribing:
            return
        self._cancel_requested = True
        self._cancel_gen += 1  # invalidate any stale worker completions
        _log(f"Transcription cancel requested (gen={self._cancel_gen})")
        # Immediately clear the UI so cancellation is instant
        self.progress["value"] = 0
        self.cancel_lbl.pack_forget()
        self.is_transcribing = False
        self._set_ui_busy(False)
        self.status_var.set("Transcription cancelled.")
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert("1.0", "Transcription was cancelled.")
        self.text_widget.config(state=tk.DISABLED)
        # Stale worker thread will finish and call the completion handler,
        # which compares _cancel_gen vs _worker_gen to skip stale completions.

    def _on_transcription_done(self, result: dict):
        # Stale worker guard: if _cancel_gen changed since this worker started, skip
        if self._cancel_gen != self._worker_gen:
            return
        self.progress.stop()
        self.cancel_lbl.pack_forget()
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

        self.clear_btn.config(state=tk.NORMAL if text else tk.DISABLED)
        if text:
            self.copy_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            self.status_var.set("Transcription complete.")
        else:
            self.status_var.set("Transcription complete — no speech detected.")
            self.lang_var.set("")

    def _on_transcription_error(self, exc: Exception):
        # Stale worker guard: if _cancel_gen changed since this worker started, skip
        if self._cancel_gen != self._worker_gen:
            return
        self.progress.stop()
        self.cancel_lbl.pack_forget()
        self.is_transcribing = False
        self._set_ui_busy(False)

        err_msg = str(exc)
        if err_msg == "Cancelled":
            self.status_var.set("Transcription cancelled.")
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert("1.0", "Transcription was cancelled.")
            self.text_widget.config(state=tk.DISABLED)
            _log("Transcription cancelled by user")
            return

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
        self.record_btn.config(state=state if not self.is_listening else tk.DISABLED)

        # Keep copy/save enabled only when we have text
        if not busy and not self.transcription_text:
            self.copy_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)

    def _on_close(self):
        if self.is_transcribing or self.is_listening:
            msg = "Transcription in progress. Quit anyway?" if self.is_transcribing else "Recording in progress. Quit anyway?"
            if not messagebox.askyesno("Quit?", msg):
                return
        if self.is_listening:
            self.live_stop.set()
        self._stop_timer()
        self.root.destroy()


# ---------------------------------------------------------------------------
#  Startup log: write right after tkinter import so we know it worked
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
