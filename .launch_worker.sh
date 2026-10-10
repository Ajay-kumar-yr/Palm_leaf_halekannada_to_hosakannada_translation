#!/bin/bash
# Palmira worker: its own conda env (Python 3.7), held in the foreground.
# It must run from ~/Palmira, which holds configs/, pretrained/ and
# predictor.py. Loading the model costs ~20s, which is why it is a
# long-lived worker and not a per-upload subprocess.
source "$HOME/miniconda3/etc/profile.d/conda.sh" || exit 1
conda activate palmira || exit 1
cd "$HOME/Palmira" || exit 1
exec python -u /mnt/d/major_proj/Palm_leaf_halekannada_to_hosakannada_translation/data/raw_corpus/palmira_worker.py
