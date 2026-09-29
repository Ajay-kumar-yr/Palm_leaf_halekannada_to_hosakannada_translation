"""CLAUDE.md rule 3: before any training run longer than 10 minutes, prove
the model can memorise 8 examples to near-zero error first. This is the
modernizer's version of setu.recogniser.memorize_check -- same rationale,
applied to the encoder-decoder transformer instead of the CTC recogniser.

8 short real old/modern pairs (drawn from the S3 corpus, not invented) are
used to train the Modernizer to memorisation and checked by greedy
autoregressive decoding against the target modern text.

Usage:
    python -m setu.modernizer.memorize_check
"""

from __future__ import annotations

import sys

import torch
from torch import nn

from setu.data import wx
from setu.eval.metrics import cer
from setu.modernizer.model import Modernizer
from setu.modernizer.vocab import build_vocab
from setu.runlog import finish_run, start_run

SEED = 0

PAIRS = [
    ("ಅಯ್ಯೋ ನನಗೇನೂ ಬೇಡ", "ಅಯ್ಯೋ ನನಗೇನೂ ಬೇಡ"),
    ("ಇದು ಐಕ್ಯಸ್ಥಲಿಯ ಅನುಭವ ಕಥನ", "ಇದು ಐಕ್ಯಸ್ಥಲಿಯ ಅನುಭವ ಕಥನ"),
    ("ಎನ್ದರೆ ಹೌದು ಸ್ವಲ್ಪ ಕಷ್ಟ", "ಎಂದರೆ ಹೌದು ಸ್ವಲ್ಪ ಕಷ್ಟ"),
    ("ಎನ್ದು ಒನ್ದು ಪ್ರಶ್ನೆ", "ಎಂದು ಒಂದು ಪ್ರಶ್ನೆ"),
    ("ಎನ್ದು ಕೇಳುವವರೂ ಇದ್ದಾರೆ", "ಎಂದು ಕೇಳುವವರೂ ಇದ್ದಾರೆ"),
    ("ಎನ್ನುವ ಸಮಯ ಇನ್ದು", "ಎನ್ನುವ ಸಮಯ ಇಂದು"),
    ("ಏನಾದರಾಗಲೀ ನೋಡೇ ಬಿಡೋಣ", "ಏನಾದರಾಗಲೀ ನೋಡೇ ಬಿಡೋಣ"),
    ("ಸಾಧಕನೆ ಪರಿಕಿಸಿ ನೀ ನೋಡು", "ಸಾಧಕನೆ ಪರಿಕಿಸಿ ನೀ ನೋಡು"),
]

MAX_EPOCHS = 400
TARGET_CER = 0.0
LOG_EVERY = 25


def build_batch(vocab, device: torch.device):
    src_seqs = [[wx.SYMBOL_TO_INDEX[s] for s in wx.encode(old)] for old, _ in PAIRS]
    tgt_seqs = [vocab.encode(modern) for _, modern in PAIRS]

    max_src = max(len(s) for s in src_seqs)
    max_tgt_in = max(len(s) for s in tgt_seqs) + 1  # + BOS

    src_ids = torch.zeros(max_src, len(PAIRS), dtype=torch.long)
    src_pad_mask = torch.ones(len(PAIRS), max_src, dtype=torch.bool)
    tgt_in_ids = torch.full((max_tgt_in, len(PAIRS)), vocab.pad_id, dtype=torch.long)
    tgt_out_ids = torch.full((max_tgt_in, len(PAIRS)), vocab.pad_id, dtype=torch.long)
    tgt_pad_mask = torch.ones(len(PAIRS), max_tgt_in, dtype=torch.bool)

    for i, seq in enumerate(src_seqs):
        src_ids[: len(seq), i] = torch.tensor(seq, dtype=torch.long)
        src_pad_mask[i, : len(seq)] = False

    for i, seq in enumerate(tgt_seqs):
        tgt_in = [vocab.bos_id] + seq
        tgt_out = seq + [vocab.eos_id]
        tgt_in_ids[: len(tgt_in), i] = torch.tensor(tgt_in, dtype=torch.long)
        tgt_out_ids[: len(tgt_out), i] = torch.tensor(tgt_out, dtype=torch.long)
        tgt_pad_mask[i, : len(tgt_in)] = False

    return (
        src_ids.to(device), src_pad_mask.to(device),
        tgt_in_ids.to(device), tgt_out_ids.to(device), tgt_pad_mask.to(device),
    )


def main() -> None:
    torch.manual_seed(SEED)  # CLAUDE.md rule 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = build_vocab([modern for _, modern in PAIRS])
    model = Modernizer(len(vocab), vocab.pad_id, vocab.bos_id, vocab.eos_id).to(device)
    ce_loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    src_ids, src_pad_mask, tgt_in_ids, tgt_out_ids, tgt_pad_mask = build_batch(vocab, device)
    max_gen_len = max(len(vocab.encode(m)) for _, m in PAIRS) + 5

    run_dir = start_run(
        "modernizer_memorize_check",
        {
            "seed": SEED,
            "pairs": [{"old": o, "modern": m} for o, m in PAIRS],
            "max_epochs": MAX_EPOCHS,
            "target_cer": TARGET_CER,
            "vocab_size": len(vocab),
            "device": str(device),
        },
    )

    final_epoch, final_loss, final_cer, predictions = None, None, None, None
    model.train()
    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad()
        logits = model(src_ids, tgt_in_ids, src_pad_mask, tgt_pad_mask)
        loss = ce_loss(logits.reshape(-1, logits.shape[-1]), tgt_out_ids.reshape(-1))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            generated = model.greedy_generate(src_ids, src_pad_mask, max_gen_len)
        predictions = [vocab.decode(seq) for seq in generated]
        cers = [cer(modern, pred) for (_, modern), pred in zip(PAIRS, predictions)]
        mean_cer = sum(cers) / len(cers)

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}  mean CER {mean_cer:.4f}")

        if mean_cer <= TARGET_CER:
            final_epoch, final_loss, final_cer = epoch, loss.item(), mean_cer
            break
    else:
        final_epoch, final_loss, final_cer = MAX_EPOCHS, loss.item(), mean_cer

    passed = final_cer <= TARGET_CER
    results = {
        "passed": passed,
        "epoch_reached": final_epoch,
        "final_loss": final_loss,
        "final_mean_cer": final_cer,
        "per_pair": [
            {"old": o, "target_modern": m, "predicted_modern": p, "cer": cer(m, p)}
            for (o, m), p in zip(PAIRS, predictions)
        ],
    }
    finish_run(run_dir, results)

    print(f"\nRun folder: {run_dir}")
    if passed:
        print(f"PASS -- memorised all 8 examples (CER {final_cer:.4f}) by epoch {final_epoch}.")
    else:
        print(
            f"FAIL -- CER {final_cer:.4f} after {final_epoch} epochs, did not reach "
            f"{TARGET_CER}. Per CLAUDE.md rule 3: something is broken -- do NOT start a real "
            f"pretraining run until this passes."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
