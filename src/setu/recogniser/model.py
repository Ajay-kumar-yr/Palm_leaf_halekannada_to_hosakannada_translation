"""The CRNN recogniser (roadmap v4 Track B, the only recogniser path).

Architecture per CLAUDE.md: 5 conv blocks (32->256 channels, pool in
height only after block 3), one BiLSTM (256 hidden units), CTC output
over the WX vocabulary + blank. Target size 3-5M parameters. Input height
64, variable width.

Blocks 3-5 use two conv layers each (rather than one) to reach the
3-5M-parameter target while keeping the "5 conv blocks", "32->256
channels", and "one BiLSTM" constraints from CLAUDE.md exactly as
specified -- a single conv per block landed at ~2.1M params, under
target; doubling the convs in the deeper, most-channel-heavy blocks
closes the gap without adding a second recurrent layer or changing the
channel range.

Blocks 1-3 pool both height and width (2x2); blocks 4-5 pool height only
(2x1), preserving the width/time resolution CTC decodes over -- this is
what "pool in height only after block 3" means. A final adaptive pool
collapses whatever height remains (64 / 2^5 = 2) to exactly 1 before the
recurrent stage.
"""

from __future__ import annotations

import torch
from torch import nn

from setu.data import wx

NUM_CLASSES = len(wx.VOCAB) + 1  # + CTC blank at index 0
RNN_HIDDEN = 256
WIDTH_DOWNSAMPLE = 8  # blocks 1-3 each halve width; blocks 4-5 do not


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, rnn_hidden: int = RNN_HIDDEN):
        super().__init__()

        self.block1 = nn.Sequential(_conv_bn_relu(1, 32), nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(_conv_bn_relu(32, 64), nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(
            _conv_bn_relu(64, 128), _conv_bn_relu(128, 128), nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            _conv_bn_relu(128, 256), _conv_bn_relu(256, 256), nn.MaxPool2d((2, 1), (2, 1))
        )
        self.block5 = nn.Sequential(
            _conv_bn_relu(256, 256), _conv_bn_relu(256, 256), nn.MaxPool2d((2, 1), (2, 1))
        )
        self.collapse_height = nn.AdaptiveAvgPool2d((1, None))

        self.rnn = nn.LSTM(
            input_size=256, hidden_size=rnn_hidden, num_layers=1,
            bidirectional=True, batch_first=False,
        )
        self.classifier = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 1, H=64, W) -> log_probs: (T, B, num_classes) for CTC."""
        x = self.block1(images)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.collapse_height(x)  # (B, C=256, 1, T)

        b, c, h, t = x.shape
        assert h == 1, f"expected height collapsed to 1, got {h}"
        x = x.squeeze(2).permute(2, 0, 1)  # (T, B, C)

        x, _ = self.rnn(x)  # (T, B, 2*hidden)
        logits = self.classifier(x)  # (T, B, num_classes)
        return logits.log_softmax(dim=2)

    def output_length(self, input_width: int) -> int:
        """Time steps CTC decodes over, for a given input image width."""
        return input_width // WIDTH_DOWNSAMPLE


def expand_classifier(state_dict: dict, num_classes: int) -> dict:
    """Load a checkpoint trained on a smaller vocabulary into a model with
    more output classes (setu.data.wx.EXTENDED_VOCAB adds Kannada digits).

    Only the final Linear grows. Its existing rows are copied across
    unchanged -- the extension appends symbols, so every old class index
    still means what it meant -- and the new rows start from the same
    initialisation a fresh Linear would use. Everything the conv stack
    and the LSTM learned is kept intact.

    Returns the state dict unchanged when the sizes already match, so
    callers can apply it unconditionally.
    """
    w, b = state_dict.get("classifier.weight"), state_dict.get("classifier.bias")
    if w is None or w.shape[0] == num_classes:
        return state_dict
    if w.shape[0] > num_classes:
        raise ValueError(
            f"checkpoint has {w.shape[0]} output classes, more than the {num_classes} "
            f"asked for -- refusing to drop classes"
        )
    ref = nn.Linear(w.shape[1], num_classes)
    new_w, new_b = ref.weight.data.clone(), ref.bias.data.clone()
    new_w[: w.shape[0]] = w
    new_b[: b.shape[0]] = b
    out = dict(state_dict)
    out["classifier.weight"], out["classifier.bias"] = new_w, new_b
    return out


def greedy_decode(
    log_probs: torch.Tensor, input_lengths: torch.Tensor | None = None
) -> list[list[int]]:
    """CTC greedy (best-path) decode: (T, B, C) log-probs -> per-batch class
    index sequences with blanks removed and repeats collapsed.

    `input_lengths` (one real length per batch item) must be passed for any
    batch with more than one distinct image width: batched images are
    padded to the same width, and without truncating each path to its own
    real length first, decode reads into the zero-padding region, which the
    model was never trained to produce anything meaningful for (CTCLoss
    itself respects input_lengths internally; decode does not unless told
    to).
    """
    best_path = log_probs.argmax(dim=2).transpose(0, 1)  # (B, T)
    decoded: list[list[int]] = []
    for b, path in enumerate(best_path.tolist()):
        if input_lengths is not None:
            path = path[: int(input_lengths[b])]
        seq: list[int] = []
        prev = None
        for idx in path:
            if idx != 0 and idx != prev:  # 0 is the blank index
                seq.append(idx)
            prev = idx
        decoded.append(seq)
    return decoded


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
