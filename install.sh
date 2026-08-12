#!/bin/bash
set -e

echo "========================================"
echo "AI CARTOON VIDEO CONVERTER - Installer"
echo "========================================"
echo ""

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

echo "Detected OS: $OS"
echo ""

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "WARNING: FFmpeg not found. Attempting to install..."
    if [[ "$OS" == "linux" ]]; then
        if command -v apt-get &> /dev/null; then
            apt-get update -qq && apt-get install -y -qq ffmpeg
        elif command -v yum &> /dev/null; then
            yum install -y ffmpeg
        elif command -v pacman &> /dev/null; then
            pacman -S --noconfirm ffmpeg
        else
            echo "ERROR: Could not install FFmpeg automatically. Please install it manually."
            exit 1
        fi
    elif [[ "$OS" == "macos" ]]; then
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo "ERROR: Please install FFmpeg manually (brew install ffmpeg)"
            exit 1
        fi
    else
        echo "ERROR: Please install FFmpeg manually and add it to PATH"
        exit 1
    fi
fi

FFMPEG_VERSION=$(ffmpeg -version | head -n1)
echo "FFmpeg found: $FFMPEG_VERSION"
echo ""

# Create virtual environment (optional)
if [ "$1" == "--venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    source venv/bin/activate
    PYTHON_CMD="python"
fi

# Upgrade pip
echo "Upgrading pip..."
$PYTHON_CMD -m pip install --quiet --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
$PYTHON_CMD -m pip install --quiet -r requirements.txt

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "To start the converter, run:"
echo "  python run.py"
echo ""
echo "For help:"
echo "  python run.py --help"
echo ""
echo "Google Colab Quick Start:"
echo "  !bash install.sh"
echo "  !python run.py"
echo ""
