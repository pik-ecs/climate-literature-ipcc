"""Tune WG III binary rules against AR6-cited ground truth.

Positives: documents cited by WG III chapters (wg3_n > 0). Recall on them is
optimistic for every rule (topic shares are estimated from the same
citations), but rule-vs-rule comparison is fair. Contrast set: documents
cited by WG I but not WG III — literature the IPCC did engage with, filed
elsewhere; we want rules that don't flag these.

Rules evaluated from cached passes (wg3_score, wg3_coverage,
wg3_threshold_examples — rerun those first if the model changed):
  product >= t      score = Σ_t loading · wg3_share (continuous, cut swept)
  max-loading >= m  the old soft rule (cut swept)
  argmax            of the shrunk three-WG content scores
  argmax-br         base-rate corrected: argmax of score - corpus mean
  argmax+margin     base-rate corrected, top-two margin >= 0.05

    uv run python scripts/wg3_tune.py
"""

import numpy as np
import pandas as pd

from climate_literature.constants import IPCC_DATA

Z_SCORE = np.load("/tmp/wg3_score.npz", allow_pickle=True)  # ids, score (product)
Z_DOC = np.load("/tmp/wg3_doc_wg_scores.npz", allow_pickle=True)  # ids, S (N,3)
Z_MAX = np.load("/tmp/wg3_threshold_stat.npz", allow_pickle=True)  # ids, mx

ids = pd.Index(Z_SCORE["ids"])
assert (Z_DOC["ids"] == ids.to_numpy()).all() and (Z_MAX["ids"] == ids.to_numpy()).all()

prod = Z_SCORE["score"].astype(np.float64)
S = Z_DOC["S"].astype(np.float64)
mx = Z_MAX["mx"].astype(np.float64)

dwg = pd.read_parquet(IPCC_DATA / "ipcc_doc_wg.parquet").set_index("item_id")
dwg = dwg.reindex(ids).fillna(0.0)
w1, w2, w3 = (dwg[f"wg{n}_n"].to_numpy() for n in (1, 2, 3))
pos = w3 > 0
neg1 = (w1 > 0) & ~pos
print(f"positives (wg3-cited): {pos.sum():,}   wg1-only contrast: {neg1.sum():,}  ")


def row(name, flag):
    print(
        f"{name:22s} {flag.sum():>9,} {flag.mean():>6.1%}  "
        f"{flag[pos].mean():>6.1%}  {flag[neg1].mean():>6.1%}"
    )


argmax = S.argmax(axis=1) + 1
corrected = (S - S.mean(axis=0)).argmax(axis=1) + 1
margin = np.sort(S, axis=1)[:, 2] - np.sort(S, axis=1)[:, 1]

print(f"{'rule':22s} {'flagged':>9} {'share':>6}  {'recall3':>7}  {'flag_wg1':>8}")
for t in (0.10, 0.20, 0.30, 1 / 3, 0.40, 0.50):
    row(f"product >= {t:.3f}", prod >= t)
for m in (0.05, 0.10, 0.15, 0.20):
    row(f"max-loading >= {m:.2f}", mx >= m)
row("argmax == 3", argmax == 3)
row("argmax-br == 3", corrected == 3)
row("argmax-br margin>=.05", (corrected == 3) & (margin >= 0.05))

print("\nrecall on wg3-cited vs flag rate on wg1-only, product cut fine grid:")
for t in np.arange(0.15, 0.55, 0.025):
    f = prod >= t
    print(
        f"  t={t:.3f}  flagged {f.sum():>8,}  recall3 {f[pos].mean():.1%}  "
        f"wg1-flag {f[neg1].mean():.1%}"
    )
