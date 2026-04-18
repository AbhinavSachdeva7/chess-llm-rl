#!/usr/bin/env bash
# setup_stockfish.sh — Download and extract Stockfish binary
set -euo pipefail

STOCKFISH_DIR="./stockfish"
mkdir -p "$STOCKFISH_DIR"

OS="$(uname -s 2>/dev/null || echo Windows)"

if [[ "$OS" == "Linux" ]]; then
    FILENAME="stockfish-ubuntu-x86-64-avx2.tar"
    URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_18/${FILENAME}"
    DEST="$STOCKFISH_DIR/$FILENAME"

    if [[ -f "$STOCKFISH_DIR/stockfish" ]]; then
        echo "Stockfish binary already exists at $STOCKFISH_DIR/stockfish — skipping download."
        exit 0
    fi

    echo "Downloading Stockfish for Linux..."
    curl -L -o "$DEST" "$URL"

    echo "Extracting..."
    tar -xf "$DEST" -C "$STOCKFISH_DIR" --strip-components=1
    rm -f "$DEST"

    # Find binary and rename/move if needed
    SF_BIN="$(find "$STOCKFISH_DIR" -name 'stockfish*' -type f | head -n1)"
    if [[ "$SF_BIN" != "$STOCKFISH_DIR/stockfish" ]]; then
        mv "$SF_BIN" "$STOCKFISH_DIR/stockfish"
    fi

    chmod +x "$STOCKFISH_DIR/stockfish"
    echo "Stockfish installed at $STOCKFISH_DIR/stockfish"

elif [[ "$OS" == "Windows_NT" || "$OS" == "Windows" || "$OS" =~ MINGW || "$OS" =~ MSYS || "$OS" =~ CYGWIN ]]; then
    FILENAME="stockfish-windows-x86-64-avx2.zip"
    URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_18/${FILENAME}"
    DEST="$STOCKFISH_DIR/$FILENAME"

    if [[ -f "$STOCKFISH_DIR/stockfish.exe" ]]; then
        echo "Stockfish binary already exists at $STOCKFISH_DIR/stockfish.exe — skipping download."
        exit 0
    fi

    echo "Downloading Stockfish for Windows..."
    curl -L -o "$DEST" "$URL"

    echo "Extracting..."
    # Use PowerShell to unzip
    powershell -NoProfile -Command "Expand-Archive -Path '${DEST}' -DestinationPath '${STOCKFISH_DIR}' -Force"
    rm -f "$DEST"

    # Find the .exe and move it to the top-level stockfish dir
    SF_EXE="$(find "$STOCKFISH_DIR" -name 'stockfish*.exe' -type f | head -n1)"
    if [[ -n "$SF_EXE" && "$SF_EXE" != "$STOCKFISH_DIR/stockfish.exe" ]]; then
        mv "$SF_EXE" "$STOCKFISH_DIR/stockfish.exe"
        # Remove empty subdirectories left by extraction
        find "$STOCKFISH_DIR" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
    fi

    echo "Stockfish installed at $STOCKFISH_DIR/stockfish.exe"

else
    echo "ERROR: Unsupported OS: $OS" >&2
    exit 1
fi
