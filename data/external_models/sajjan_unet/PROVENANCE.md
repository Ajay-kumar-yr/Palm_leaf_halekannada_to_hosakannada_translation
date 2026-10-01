Source: https://github.com/sajjanvsl/U-Net-CNN-for-binarization-of-Historical-Kannada-Handwritten-Palm-Leaf-Manuscripts
(MIT licensed, see LICENSE in this folder)

S.P. Sajjan et al.'s U-Net binarizer -- the same model used to produce
HKHPL's own Ground_Truth_images. Used here for the real-photo inference
pipeline only (Palmira -> cut lines -> this -> CRNN); never applied to
synthetic S1/S2 images. See src/setu/demo/binarize.py.

`unet_best_weights.pth` (97,924,917 bytes,
sha256 aef2dc11584ba489217cf7f41642583059046f2341efda1178b4867238186f44)
is gitignored like every other model checkpoint in this repo -- re-download
from the `main` branch of the source repo above (root-level
`unet_best_weights.pth`) if missing.
