"""The modernizer: a small encoder-decoder transformer trained from
scratch (CLAUDE.md: "~6 encoder + 6 decoder layers, hidden size 256, 4
heads, ~20M params -- NOT ByT5-small. ByT5's Kannada-script pretraining
doesn't transfer to WX-romanised input, and it costs ~10x the training
time for that reason").

Encoder input: WX-symbol ids (setu.data.wx.VOCAB, ~56 symbols + blank/pad)
-- for B3/B4 this is where the recogniser's argmax output or the soft
bridge's blended embeddings plug in; for S3/S2 pretraining and fine-tuning
it's the WX-encoding of plain old-Kannada text (no recogniser involved
yet). Decoder output: characters from a data-built CharVocab (real modern
Kannada text, see setu.modernizer.vocab) -- generation is standard
teacher-forced cross-entropy at train time, greedy/autoregressive at
inference.

hidden=256, heads=4, feedforward=2048, 6+6 layers lands at ~19-20M
trainable parameters (see count_parameters) -- feedforward=1024 (the more
"default" 4x-hidden choice) undershoots to ~13M, so feedforward is widened
instead of adding a third recurrent/attention stack, keeping the layer
counts and hidden size exactly as CLAUDE.md specifies.
"""

from __future__ import annotations

import torch
from torch import nn

from setu.data import wx

HIDDEN = 256
HEADS = 4
FEEDFORWARD = 2048
ENCODER_LAYERS = 6
DECODER_LAYERS = 6
DROPOUT = 0.1
MAX_LEN = 2048  # generous cap on sequence length for positional embeddings

ENCODER_VOCAB_SIZE = len(wx.VOCAB) + 1  # + PAD at index 0 (WX indices already reserve 0 for CTC blank;
# reused as PAD here too -- the modernizer never sees a "blank" concept, so the same index is fine.)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, hidden: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, B, hidden) -> adds a learned position embedding per step."""
        t = x.shape[0]
        positions = torch.arange(t, device=x.device)
        return x + self.embedding(positions).unsqueeze(1)


class Modernizer(nn.Module):
    def __init__(self, output_vocab_size: int, pad_id: int, bos_id: int, eos_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self._embed_scale = HIDDEN**0.5  # standard transformer embedding scaling (Vaswani et al.)
        self.src_embed = nn.Embedding(ENCODER_VOCAB_SIZE, HIDDEN, padding_idx=0)
        self.tgt_embed = nn.Embedding(output_vocab_size, HIDDEN, padding_idx=pad_id)
        self.pos_embed = PositionalEmbedding(MAX_LEN, HIDDEN)

        # norm_first=True (pre-LN): PyTorch's default (post-LN) is notoriously
        # unstable for a deep (6+6 layer) transformer trained from scratch
        # without a learning-rate warmup schedule -- this was the actual
        # cause of the memorize-check getting stuck at loss ~3.3 for 350+
        # epochs (the mask-dtype fix above was real but insufficient on its
        # own). Pre-LN is the standard fix used by most from-scratch
        # transformer training recipes for exactly this reason.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN, nhead=HEADS, dim_feedforward=FEEDFORWARD, dropout=DROPOUT, norm_first=True
        )
        # Pre-LN leaves the final residual stream un-normalized, so a final
        # LayerNorm after the stack is standard practice (not optional).
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=ENCODER_LAYERS, norm=nn.LayerNorm(HIDDEN))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=HIDDEN, nhead=HEADS, dim_feedforward=FEEDFORWARD, dropout=DROPOUT, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=DECODER_LAYERS, norm=nn.LayerNorm(HIDDEN))

        self.output_proj = nn.Linear(HIDDEN, output_vocab_size)

    def encode(self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """src_ids: (T_src, B) -> memory: (T_src, B, hidden). The B3 (argmax)
        path: a single discrete WX symbol per frame, looked up in src_embed."""
        return self.encode_embeds(self.src_embed(src_ids), src_key_padding_mask)

    def encode_embeds(self, src_embeds: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """src_embeds: (T_src, B, hidden), already-looked-up or already-blended
        -> memory: (T_src, B, hidden). The B4 (soft bridge) path plugs in here
        directly with confidence-blended embeddings (setu.bridge.soft_bridge)
        instead of a single discrete symbol's lookup -- same encoder either
        way, only what feeds it differs, per CLAUDE.md: "same recogniser, same
        data, same modernizer training procedure -- only the interface
        differs" between B3 and B4."""
        x = self.pos_embed(src_embeds * self._embed_scale)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """tgt_ids: (T_tgt, B) -> logits: (T_tgt, B, vocab)."""
        t = tgt_ids.shape[0]
        # Boolean, not generate_square_subsequent_mask's float -inf mask:
        # mixing a float attn_mask with a boolean key_padding_mask triggers
        # PyTorch's "mismatched key_padding_mask and attn_mask" deprecation
        # path, which merges them incorrectly in this torch version and was
        # the actual cause of the modernizer memorize-check failing to
        # converge (loss stuck ~3.3 for 350+ epochs) -- both masks must be
        # the same (boolean, True = masked-out) type.
        causal_mask = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=tgt_ids.device), diagonal=1
        )
        x = self.pos_embed(self.tgt_embed(tgt_ids) * self._embed_scale)
        out = self.decoder(
            x, memory, tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(out)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_in_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced training forward. src_ids: (T_src, B), tgt_in_ids
        (decoder input, i.e. targets shifted right with BOS prepended):
        (T_tgt, B) -> logits: (T_tgt, B, vocab)."""
        memory = self.encode(src_ids, src_key_padding_mask)
        return self.decode(tgt_in_ids, memory, tgt_key_padding_mask, src_key_padding_mask)

    @torch.no_grad()
    def greedy_generate(
        self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor, max_len: int
    ) -> list[list[int]]:
        """src_ids: (T_src, B) -> per-batch generated id sequences (no BOS/EOS)."""
        device = src_ids.device
        batch = src_ids.shape[1]
        memory = self.encode(src_ids, src_key_padding_mask)

        generated = torch.full((1, batch), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for _ in range(max_len):
            tgt_padding_mask = torch.zeros(batch, generated.shape[0], dtype=torch.bool, device=device)
            logits = self.decode(generated, memory, tgt_padding_mask, src_key_padding_mask)
            next_ids = logits[-1].argmax(dim=-1)  # (B,)
            next_ids = torch.where(finished, torch.full_like(next_ids, self.pad_id), next_ids)
            generated = torch.cat([generated, next_ids.unsqueeze(0)], dim=0)
            finished = finished | (next_ids == self.eos_id)
            if bool(finished.all()):
                break

        sequences = generated[1:].transpose(0, 1).tolist()  # drop BOS, (B, T)
        out = []
        for seq in sequences:
            if self.eos_id in seq:
                seq = seq[: seq.index(self.eos_id)]
            out.append(seq)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
