#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the Keystroke-GRU project.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# System packages:
#   python3-tk        -> tkinter GUIs (collect_data.py, login.py)
#   python3-dev + build-essential -> compile evdev (a Linux dep of pynput)
#   xvfb              -> headless X server so the GUI apps can run in Cloud Agents
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends \
  python3-tk \
  python3-dev \
  build-essential \
  xvfb

# Python dependencies. The base image marks the system interpreter as PEP 668
# "externally managed", so install into it explicitly; the VM is isolated.
pip3 install --break-system-packages -r requirements.txt
