# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller .spec for Audio Transcriber (macOS .app).

Uses COLLECT (onedir mode) so all Python .so / .dylib / .pyc files
sit alongside the executable on disk.  macOS LaunchServices can then
load them without needing a self-extraction step.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate site-packages so we can add explicit binary paths
# ---------------------------------------------------------------------------
SITE_PACKAGES = Path([p for p in sys.path if "site-packages" in p][0])

extra_binaries = []

def _add_binaries(src_dir, dest_subdir):
    """Add every .dylib and .so from *src_dir* to extra_binaries."""
    src = Path(src_dir)
    if not src.is_dir():
        return
    for f in src.iterdir():
        if f.is_file() and f.suffix in (".dylib", ".so"):
            extra_binaries.append((str(f), dest_subdir))

# ---------- onnxruntime C API ----------
_add_binaries(SITE_PACKAGES / "onnxruntime" / "capi", "onnxruntime/capi")

# ---------- ctranslate2 .dylibs ----------
_add_binaries(SITE_PACKAGES / "ctranslate2" / ".dylibs", "ctranslate2/.dylibs")

# ---------- PyAV .dylibs & root .so files ----------
_add_binaries(SITE_PACKAGES / "av" / ".dylibs", "av/.dylibs")
for f in (SITE_PACKAGES / "av").iterdir():
    if f.is_file() and f.suffix in (".so", ".dylib"):
        extra_binaries.append((str(f), "av"))

# ---------- tkinter .so (from python lib-dynload) ----------
# Anaconda keeps _tkinter.cpython-313-*.so inside lib-dynload
LIB_DYNLOAD = SITE_PACKAGES.parent / "lib-dynload"
if LIB_DYNLOAD.is_dir():
    for f in LIB_DYNLOAD.iterdir():
        if f.name.startswith("_tkinter") and f.suffix == ".so":
            extra_binaries.append((str(f), "."))

# ---------- Tcl / Tk framework dylibs (needed by _tkinter.so) ----------
# _tkinter.so links to @rpath/libtcl8.6.dylib and @rpath/libtk8.6.dylib.
# These live in the Anaconda lib directory and MUST be bundled alongside
# the executable so the dynamic linker finds them when launched from Finder.
_tcl_lib = Path("/opt/anaconda3/lib/libtcl8.6.dylib")
_tk_lib = Path("/opt/anaconda3/lib/libtk8.6.dylib")
if _tcl_lib.exists():
    extra_binaries.append((str(_tcl_lib), "."))
if _tk_lib.exists():
    extra_binaries.append((str(_tk_lib), "."))

# ---------- data files (needed at runtime, NOT collected automatically) ----------
extra_datas = []

# faster-whisper VAD model  (silero_vad_v6.onnx) — required for vad_filter=True
_vad_asset = SITE_PACKAGES / "faster_whisper" / "assets" / "silero_vad_v6.onnx"
if _vad_asset.exists():
    extra_datas.append((str(_vad_asset), "faster_whisper/assets"))

# yaml package Python source files — the C extension _yaml.so is collected
# automatically by PyInstaller, but the .py files (__init__.py, dumper.py,
# loader.py, etc.) are sometimes MISSING in the bundle, causing the
# "module 'yaml' has no attribute 'dump'" crash at runtime.
_yaml_pkg = SITE_PACKAGES / "yaml"
if _yaml_pkg.is_dir():
    for _f in _yaml_pkg.iterdir():
        if _f.is_file() and _f.suffix == ".py":
            extra_datas.append((str(_f), "yaml"))
    # Also add the _yaml C extension explicitly if not already bundled
    _yaml_so = _yaml_pkg / "_yaml.cpython-313-darwin.so"
    if _yaml_so.exists():
        extra_binaries.append((str(_yaml_so), "yaml"))

# ---------------------------------------------------------------------------
# Hidden imports
# ---------------------------------------------------------------------------
hidden_imports = [
    # faster-whisper
    "faster_whisper",
    "faster_whisper.transcribe",
    "faster_whisper.audio",
    "faster_whisper.feature_extractor",
    "faster_whisper.tokenizer",
    "faster_whisper.utils",
    "faster_whisper.vad",
    # ctranslate2
    "ctranslate2",
    "ctranslate2.specs",
    "ctranslate2.models",
    "ctranslate2.converters",
    "ctranslate2.extensions",
    # onnxruntime
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi.onnxruntime_inference_collection",
    "onnxruntime.capi.onnxruntime_validation",
    "onnxruntime.capi._pybind_state",
    # PyAV
    "av",
    "av._core",
    "av.container",
    "av.codec",
    "av.codec.context",
    "av.video",
    "av.audio",
    "av.audio.resampler",
    "av.subtitles",
    "av.filter",
    "av.sidedata",
    "av.utils",
    "av.buffer",
    "av.packet",
    "av.stream",
    "av.format",
    "av.frame",
    "av.plane",
    "av.dictionary",
    "av.logging",
    "av.bitstream",
    "av.error",
    "av.device",
    "av.opaque",
    "av.datasets",
    "av.about",
    # sounddevice (live microphone capture)
    "sounddevice",
    "sounddevice._sounddevice",
    "_sounddevice",
    # huggingface + tokenizers
    "huggingface_hub",
    "huggingface_hub.hub",
    "huggingface_hub.file_download",
    "huggingface_hub.snapshot_download",
    "huggingface_hub.repocard",
    "huggingface_hub.utils._fixes",
    "hf_xet",
    "tokenizers",
    "tokenizers.models",
    "tokenizers.pre_tokenizers",
    "tokenizers.trainers",
    "tokenizers.decoders",
    "tokenizers.normalizers",
    "tokenizers.processors",
    # yaml (fixes "module 'yaml' has no attribute 'dump'" crash)
    "yaml",
    "yaml._yaml",
    "_yaml",
    # general
    "numpy",
    "tqdm",
    "_tkinter",
    "tkinter",
]

excluded_imports = [
    "PyQt5", "PyQt5.QtCore", "PyQt5.QtGui", "PyQt5.QtWidgets",
    "PyQt5.sip", "PyQtWebEngine",
    "PySide6", "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
    "PySide6.QtNetwork",
    "QtAwesome", "qtconsole", "qtpy", "superqt",
    "matplotlib", "PIL", "cv2", "scipy", "pandas", "sklearn",
    "tensorflow", "torch", "torchvision", "jax",
    "soundfile", "librosa",
    "plotly", "plotly.figure_factory", "plotly.graph_objects",
    "IPython", "ipykernel", "jupyter", "notebook",
    # Anaconda extras that bloat the bundle
    "panel", "bokeh", "dask", "distributed", "astropy", "altair",
    "alabaster", "anaconda_catalogs", "arrow", "asynctest",
    "bottleneck", "conda", "conda_index", "conda_build", "conda_env",
    "conda_package_handling", "conda_repo_cli", "conda_repos",
    "conda_content_trust", "conda_package_streaming",
    "conda_smithy", "cookiecutter", "cytoolz",
    "dask_image", "datashader", "datashape", "diagrams",
    "docutils", "feedparser", "flask",
    "geopandas", "geopy", "guzzle_sphinx_theme",
    "holoviews", "hvplot", "intake", "ipython_genutils",
    "jedi", "joblib", "jupyter_client", "jupyter_core",
    "jupyter_events", "jupyter_server", "jupyterlab",
    "jupyterlab_pygments", "jupyterlab_server",
    "llvmlite", "locket",
    "matplotlib_inline", "mistune", "mpmath", "multipledispatch",
    "numba", "numexpr", "openpyxl",
    "partd", "patsy", "pdfminer", "pdfplumber",
    "pep8", "pexpect", "pickleshare", "pillow", "pip",
    "prometheus_client", "prompt_toolkit", "ptyprocess",
    "pulp", "pvlib", "pyarrow", "pybind11", "pycodestyle",
    "pycosat", "pycparser", "pyct", "pycurl",
    "pydantic_settings", "pydeck", "pyerfa", "pyflakes",
    "pygithub", "pygments", "pygraphviz", "pylint",
    "pympler", "pynvim", "pyodbc", "pyparsing",
    "pyrsistent", "pyshp", "pysocks", "pytables",
    "pytest", "pytest_remotedata", "python_dateutil",
    "python_jsonrpc_server", "python_lsp_jsonrpc",
    "python_lsp_server", "pytz", "pyviz_comms", "pywavelets",
    "pyzotero",
    "reportlab", "rope",
    "ruamel", "ruamel.yaml", "ruamel.yaml.clib",
    "ruamel_yaml_conda", "scikit_image", "scikit_learn",
    "seaborn", "send2trash", "shapely",
    "sherpa", "sip", "snowballstemmer",
    "sortedcontainers", "sphinx", "sphinxcontrib",
    "sphinx_copybutton", "sphinx_issues", "sphinx_panels",
    "sphinx_tabs", "spyder", "spyder_kernels",
    "sqlalchemy", "statsmodels", "sympy", "tabulate",
    "tbb", "tblib", "terminado", "testpath",
    "textdistance", "texttable", "threadpoolctl",
    "tinycss2", "tlz", "tomli_w",
    "toolz", "tornado", "traitlets",
    "trio", "twine", "tzdata",
    "ujson", "unicodecsv", "unidecode",
    "virtualenv", "wcwidth", "webencodings",
    "werkzeug", "wheel", "wrapt", "xlrd", "xlsxwriter",
    "xlwings", "xyzservices", "zarr", "zict",
    "zipp", "zstandard",
]

# ---------------------------------------------------------------------------
# Analysis  →  PYZ  →  EXE (exclude binaries)  →  COLLECT  →  BUNDLE
# ---------------------------------------------------------------------------
a = Analysis(
    ["transcriber.py"],
    pathex=[],
    binaries=extra_binaries,
    datas=extra_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_imports,
    noarchive=False,
    module_collection_mode={
        # 'py' = store as files only (bypass PYZ compression)
        # This prevents zlib 'Error -5 while decompressing data' errors
        # that occur when large model packages are compressed into the PYZ.
        "faster_whisper": "py",
        "ctranslate2": "py",
        "onnxruntime": "py",
        "av": "py",
        "huggingface_hub": "py",
        "tokenizers": "py",
        "numpy": "py",
        "tkinter": "py",
        "yaml": "py",
    },
)

pyz = PYZ(a.pure)

# EXE with exclude_binaries=True → the executable stays small,
# and all binaries (.so / .dylib) are placed alongside it via COLLECT.
exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="Audio Transcriber",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,           # ← no terminal window
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Audio Transcriber",
)

app = BUNDLE(
    coll,
    name="Audio Transcriber.app",
    icon=None,
    display_name="Audio Transcriber",
    version="1.0.0",
    bundle_identifier="com.audiotranscriber.app",
    info_plist={
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleVersion": "1.0.0",
        "NSHumanReadableCopyright": "© 2026 Audio Transcriber",
        "LSMinimumSystemVersion": "11.0",
        # LSUIElement intentionally OMITTED – user wants a normal app
        # with a Dock icon and standard window management.
        "NSHighResolutionCapable": True,
    },
)
