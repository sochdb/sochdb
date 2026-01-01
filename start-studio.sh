#!/bin/bash
# ToonDB Studio Launcher
# Starts the ToonDB Studio as a Tauri desktop application

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDIO_DIR="$SCRIPT_DIR/toondb-studio"

echo "🎨 ToonDB Studio Launcher"
echo "========================="
echo ""

# Check if we're in the right directory
if [ ! -d "$STUDIO_DIR" ]; then
    echo "❌ Error: toondb-studio directory not found at $STUDIO_DIR"
    echo "   Make sure you're running this script from the toondb root directory."
    exit 1
fi

cd "$STUDIO_DIR"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    echo ""
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed."
    echo "   Please install Node.js and npm first."
    exit 1
fi

# Kill any existing process on port 1420
PORT=1420
if lsof -ti:$PORT > /dev/null 2>&1; then
    echo "⚠️  Port $PORT is in use. Killing existing process..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null
    sleep 1
    echo "✅ Port $PORT is now free."
    echo ""
fi

# Check for --web flag to run web-only mode
if [ "$1" == "--web" ]; then
    echo "🌐 Starting ToonDB Studio (Web Mode)..."
    echo ""
    echo "   Local:   http://localhost:$PORT/"
    echo ""
    echo "   Press Ctrl+C to stop the server."
    echo ""
    npm run dev
else
    echo "🚀 Starting ToonDB Studio (Tauri Desktop App)..."
    echo ""
    echo "   This will compile Rust and launch the desktop app."
    echo "   Press Ctrl+C to stop."
    echo ""
    npm run tauri dev
fi
