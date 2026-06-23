#!/usr/bin/env bash
# ===========================================================================
#  build.sh — Build Audio Transcriber as a standalone macOS .app
# ===========================================================================
#  Usage:
#    cd /Users/Prashant/My_Workspace/audio
#    chmod +x build.sh
#    ./build.sh
#
#  Output:
#    ./dist/Audio Transcriber.app
#
#  You can copy the .app to /Applications/ and launch it normally.
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Audio Transcriber — macOS .app Builder"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ---- 1. Verify Python ----
echo "→ Python version:"
python3 --version
echo ""

# ---- 2. Verify faster-whisper ----
echo "→ Checking faster-whisper…"
python3 -c "from faster_whisper import WhisperModel; print('   OK')" 2>&1 || {
    echo "   ERROR: faster-whisper not installed."
    echo "   Run:  pip3 install faster-whisper"
    exit 1
}

# ---- 3. Verify PyInstaller ----
echo "→ PyInstaller version: $(pyinstaller --version 2>&1)"
echo ""

# ---- 4. Clean previous build ----
echo "→ Cleaning previous build artifacts…"
rm -rf build dist __pycache__

# ---- 5. Build with .spec file ----
echo "→ Running PyInstaller (this may take several minutes)…"
echo ""

# Use the .spec file for reliable dependency handling
# NOTE: --windowed and --onedir are NOT allowed alongside a .spec file.
# Console mode and bundle structure are controlled inside the spec itself.
pyinstaller \
    --noconfirm \
    --clean \
    --log-level=INFO \
    build.spec

echo ""
echo "→ Build command completed."

# ---- 6. Post-processing: clean up & codesign ----
APP="dist/Audio Transcriber.app"

if [ -d "$APP" ]; then
    echo "→ Post-processing bundle…"

    # Remove .dist-info directories from Frameworks/ — they confuse codesign
    find "$APP/Contents/Frameworks" -type d -name '*.dist-info' -exec rm -rf {} + 2>/dev/null || true
    find "$APP/Contents/Resources" -type d -name '*.dist-info' -exec rm -rf {} + 2>/dev/null || true

    # Ad-hoc sign the whole bundle so macOS Gatekeeper accepts it
    echo "→ Signing bundle with ad-hoc signature…"
    codesign -s - --deep --force --timestamp=none "$APP" 2>&1 || {
        echo "   ⚠ Codesign warning (non-fatal): $?"
    }
fi

# ---- 7. Verify the .app was created ----
if [ -d "$APP" ]; then
    APP_SIZE_MB=$(du -sh "$APP" 2>/dev/null | cut -f1 || echo "?")
    echo "   App bundle size: $APP_SIZE_MB"
else
    echo "   WARNING: .app bundle not found at $APP"
fi

# ---- 8. Done ----
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Build complete!"
echo ""
echo "  App location:"
echo "    ./dist/Audio Transcriber.app"
echo ""
echo "  To install:"
echo "    cp -r \"$APP\" /Applications/"
echo ""
echo "  First launch will download the large-v3 model (~3 GB)."
echo "  After that, it runs fully offline."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
