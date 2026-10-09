"""Non-linear (convex) WG III scoring: do high-share topics deserve more?

The shipped score is linear in topic wg3_share: Σ_t w_t · s_t, w = renorm
loadings. A document whose loadings sit on low-share topics (AFOLU forestry,
wg3_share ~0.2) can therefore never score high even when it is squarely
mitigation policy. Hypothesis: a convex moment score

    g_r = Σ w s^r / Σ w s^(r-1)   (r=2,3; r=1 is the linear score)

upweights a document's strongly-WG III topics relative to its weak ones, so
"a little of a very WG III topic" moves the score more. This script streams
the K200 loadings once, caches the moments, and benchmarks each rule against
both ground truths: AR6-cited positives (WG III-cited, contrast WG I-only)
and the mitigation policy classifier, at each rule's natural cut (share of
corpus flagged held near the shipped 1/3-linear operating point).

Also prints, per document group, the loading profile over wg3_share bins:
where policy-positive-missed / AR6-WG3-cited / AR6-WG1-cited documents put
their mass.

RESULT (Oct 2026): hypothesis rejected. At the matched operating point
(32.2% of the dated corpus flagged), g2@0.421 gives AR6 recall 62.4% / WG I
flag 2.2% / policy coverage 83.7% vs linear@1/3's 62.0% / 1.2% / 83.7% —
identical recall and policy coverage at twice the WG I contamination; g3 is
worse on every axis. The policy misses put only ~21% of loading above share
0.45 (AR6-WG III-cited positives: ~35%), so convexity lifts them past a
matched cut only by admitting more WG I/II literature. Keep linear @ 1/3;
policy membership is better handled by the classifier's own verdict than by
reweighting the citation-share signal.

Needs the loadings shards; writes /tmp/wg3_gamma.npz.

    uv run python scripts/wg3_gamma.py
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from climate_literature.constants import IPCC_DATA, TABLES_DIR
from climate_literature.reporting.plots import (
    LAST_COMPLETE_YEAR,
    RELEVANCE_THRESHOLD,
    load_prediction_columns,
)
from climate_literature.topics.wg import DT_DIR, WG3_CUT, K

CACHE = "/tmp/wg3_gamma.npz"
BINS = [0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 1.0]
LINEAR_CUT = WG3_CUT


def stream_moments(s3: np.ndarray) -> dict[str, np.ndarray]:
    """Per-document Σ w s^r for r=0..3 over renormalised loadings, in shards."""
    parts = {"ids": [], "m0": [], "m1": [], "m2": [], "m3": []}
    for f in sorted(DT_DIR.glob("shard=*/*.parquet")):
        cols = ["item_id"] + [f"topic_{t}" for t in range(K)]
        df = pq.read_table(f, columns=cols).to_pandas().set_index("item_id")
        V = df.to_numpy(copy=True).astype(np.float64)
        V /= np.maximum(V.sum(axis=1, keepdims=True), 1e-9)
        P = parts
        P["ids"].append(df.index.to_numpy())
        P["m0"].append(V.sum(axis=1))
        P["m1"].append(V @ s3)
        P["m2"].append(V @ s3**2)
        P["m3"].append(V @ s3**3)
    return {k: np.concatenate(v) for k, v in parts.items()}


def g(z: dict, r: int) -> np.ndarray:
    """Moment score g_r = m_r / m_{r-1} (r=1 linear)."""
    return z[f"m{r}"] / np.maximum(z[f"m{r - 1}"], 1e-12)


def main() -> None:
    share = pd.read_csv(TABLES_DIR / "topic_wg_a200.csv").set_index("topic")
    s3 = share["wg3_share"].reindex(range(K)).to_numpy(np.float64)
    import os

    if os.path.exists(CACHE):
        z = dict(np.load(CACHE, allow_pickle=True))
    else:
        z = stream_moments(s3)
        np.savez(CACHE, **z)  # ty: ignore[invalid-argument-type]  # kwargs-form savez
    ids = pd.Index(z["ids"])
    lin = g(z, 1)

    # ground truths, aligned to the score index
    ipcc = pd.read_parquet(IPCC_DATA / "ipcc_doc_wg.parquet").set_index("item_id")
    ipcc = ipcc.reindex(ids).fillna(0.0)
    pos3 = (ipcc["wg3_n"] > 0).to_numpy()  # AR6 WG III-cited
    ctrl1 = ((ipcc["wg1_n"] > 0) & (ipcc["wg3_n"] == 0)).to_numpy()

    pred = load_prediction_columns(["item_id", "publication_year", "relevant"])
    pred = pred.drop_duplicates("item_id").set_index("item_id").reindex(ids)
    dated = (
        (pred["publication_year"].astype("float64") <= LAST_COMPLETE_YEAR)
        .fillna(False)
        .to_numpy()
    )
    pol = (pred["relevant"].to_numpy(np.float64) > RELEVANCE_THRESHOLD) & dated

    print(f"scored docs: {len(ids):,}   dated ≤{LAST_COMPLETE_YEAR}: {dated.sum():,}")
    print(f"AR6 ground truth: WG III-cited {pos3.sum():,}, WG I-only {ctrl1.sum():,}")
    print(
        f"policy positives: {pol.sum():,}  (linear@{LINEAR_CUT:.2f} covers "
        f"{(pol & (lin >= LINEAR_CUT)).sum() / pol.sum():.1%})"
    )

    # operating points: g_r cuts chosen to flag ~the same corpus share as
    # linear@1/3 (32% dated), plus the shipped point for reference
    base = (lin >= LINEAR_CUT) & dated
    target = base.sum() / dated.sum()
    print(f"\nlinear@1/3 flags {target:.1%} of dated corpus; matched-cut rules:")
    print(
        f"{'rule':>10} {'cut':>6} {'corpus':>7} {'AR6-recall':>11} "
        f"{'WG1-flag':>9} {'pol-cov':>8}"
    )
    rules = {"linear": g(z, 1)}
    for r in (2, 3):
        gr = g(z, r)
        rules[f"g{r}"] = gr
        # bisect the cut matching the dated-corpus flag share
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            frac = ((gr >= mid) & dated).sum() / dated.sum()
            lo, hi = (mid, hi) if frac > target else (lo, mid)
        cut = (lo + hi) / 2
        sel = (gr >= cut) & dated
        print(
            f"{f'g{r}':>10} {cut:6.3f} {sel.sum() / dated.sum():7.1%} "
            f"{(pos3 & (gr >= cut)).sum() / pos3.sum():11.1%} "
            f"{(ctrl1 & (gr >= cut)).sum() / ctrl1.sum():9.1%} "
            f"{(pol & (gr >= cut)).sum() / pol.sum():8.1%}"
        )
        # also evaluate g_r at the linear cut 1/3 (convex scores run low)
        sel = (gr >= LINEAR_CUT) & dated
        print(
            f"{f'g{r}':>10} {LINEAR_CUT:6.3f} {sel.sum() / dated.sum():7.1%} "
            f"{(pos3 & (gr >= LINEAR_CUT)).sum() / pos3.sum():11.1%} "
            f"{(ctrl1 & (gr >= LINEAR_CUT)).sum() / ctrl1.sum():9.1%} "
            f"{(pol & (gr >= LINEAR_CUT)).sum() / pol.sum():8.1%}"
        )
    sel = base
    print(
        f"{'linear':>10} {LINEAR_CUT:6.3f} {target:7.1%} "
        f"{(pos3 & (lin >= LINEAR_CUT)).sum() / pos3.sum():11.1%} "
        f"{(ctrl1 & (lin >= LINEAR_CUT)).sum() / ctrl1.sum():9.1%} "
        f"{(pol & (lin >= LINEAR_CUT)).sum() / pol.sum():8.1%}"
    )

    # where each group's loading mass sits in wg3_share space:
    # mean Σ_{s in bin} w_t  (needs the shards again — use cached moments?
    # no: recompute per-bin loading on a sampled subset of shards)
    binsum = np.zeros((3, len(BINS) - 1))
    sel_groups = [
        ("policy∧lin-miss", pol & (lin < LINEAR_CUT)),
        ("AR6-WG3-cited", pos3),
        ("AR6-WG1-only", ctrl1),
    ]
    selm = [sel for _, sel in sel_groups]
    for f in sorted(DT_DIR.glob("shard=*/*.parquet")):
        cols = ["item_id"] + [f"topic_{t}" for t in range(K)]
        df = pq.read_table(f, columns=cols).to_pandas().set_index("item_id")
        pos = ids.get_indexer(df.index.to_numpy())
        V = df.to_numpy(copy=True).astype(np.float64)
        V /= np.maximum(V.sum(axis=1, keepdims=True), 1e-9)
        for gi, sel in enumerate(selm):
            rows = np.where(sel[pos])[0]
            if not len(rows):
                continue
            W = V[rows]
            for bi in range(len(BINS) - 1):
                inb = (s3 >= BINS[bi]) & (s3 < BINS[bi + 1])
                binsum[gi, bi] += W[:, inb].sum()
    print("\nmean loading by wg3_share bin (rows: groups):")
    hdr = "  ".join(f"{BINS[i]:.2f}-{BINS[i + 1]:.2f}" for i in range(len(BINS) - 1))
    print(f"{'group':>16} {hdr}")
    for (name, sel), row in zip(sel_groups, binsum, strict=True):
        print(f"{name:>16} " + "  ".join(f"{v / max(sel.sum(), 1):5.2f}" for v in row))


if __name__ == "__main__":
    main()
