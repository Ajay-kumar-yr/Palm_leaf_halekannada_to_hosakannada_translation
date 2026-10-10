#!/bin/bash
# Run the SETU demo UI in the FOREGROUND of this script.
#
# It must not be detached with setsid/nohup: when the launching command
# exits and nothing else is running in the distro, WSL shuts the whole
# instance down a few seconds later and takes the app with it -- which
# looked exactly like a silent crash (0-byte log, no process).
cd /mnt/d/major_proj/Palm_leaf_halekannada_to_hosakannada_translation || exit 1
export PYTHONPATH=src
exec "$HOME/setu-venv/bin/python" -u -m setu.demo.app "$@"
