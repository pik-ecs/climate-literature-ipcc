"""Policy-relevant documents the WG III content score misses at the 1/3 cut.

The 2024 policy classifier targets mitigation policy (the WGIII sectors), so
its positives should mostly clear the WG III content cut. This script sizes
the disagreement — policy-relevant but score < WG3_CUT — shows how close the
misses sit to the cut, what recall of them each candidate cut would give
(alongside the corpus-wide wg3_relevant count each cut costs), and samples
misses across their score range for hand-judging.

RESULT (Oct 2026): 17,318 policy positives (16.3%) sit below the 1/3 cut,
score quantiles hugging it from under (p05 0.27). Hand-judging 12 spread
across their range: ~6 clear WG III (REDD+, GHG inventories, CDM credits,
Kyoto economics, carbon-footprint analysis), 3 borderline, 3 not WG III
(Arctic physical science; one non-climate policy paper scoring 0.99 on the
policy classifier — its noise floor). No lower cut rescues them cleanly
(0.25 → 97% policy coverage but 52% of the corpus flagged; see
scripts/wg3_gamma.py for why convex weights fare no better).

Needs /tmp/wg3_score.npz and /tmp/wg3_threshold_stat.npz (see wg3_score.py).

    uv run python scripts/wg3_policy_misses.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from climate_literature.reporting.plots import (
    LAST_COMPLETE_YEAR,
    RELEVANCE_THRESHOLD,
    SECTOR_PREFIX,
    load_prediction_columns,
    sector_columns,
)
from climate_literature.topics.wg import WG3_CUT

N_SHOW = 12
CAND = [0.15, 0.20, 0.25, WG3_CUT]

Z = np.load("/tmp/wg3_score.npz", allow_pickle=True)
ids = pd.Index(Z["ids"])
prod = Z["score"].astype(np.float64)
ZM = np.load("/tmp/wg3_threshold_stat.npz", allow_pickle=True)
assert (ZM["ids"] == ids.to_numpy()).all()
top_idx = ZM["top_idx"]

spec = importlib.util.spec_from_file_location(
    "wte", Path(__file__).with_name("wg3_threshold_examples.py")
)
assert spec is not None and spec.loader is not None
wte = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wte)

pred = load_prediction_columns(
    ["item_id", "publication_year", "relevant", *sector_columns()]
).drop_duplicates("item_id")
pred = pred.set_index("item_id").reindex(ids)
have = pred["publication_year"].notna().to_numpy()
dated = (
    (pred["publication_year"].astype("float64") <= LAST_COMPLETE_YEAR)
    .fillna(False)
    .to_numpy()
)
pol = (
    (pred["relevant"].to_numpy(np.float64) > RELEVANCE_THRESHOLD)
    & have
    & dated
    & ~np.isnan(prod)
)
sector = pred[[c for c in pred.columns if c.startswith(SECTOR_PREFIX)]]
sec_argmax = sector.dropna(how="all").idxmax(axis=1)

miss = pol & (prod < WG3_CUT)
hit = pol & (prod >= WG3_CUT)
print(
    f"policy-relevant (scored, ≤{LAST_COMPLETE_YEAR}): {pol.sum():,}   "
    f"missed at {WG3_CUT:.3f}: {miss.sum():,}  ({miss.sum() / pol.sum():.1%})"
)
print(
    "score of policy positives (quantiles):",
    np.round(np.quantile(prod[pol], [0.05, 0.1, 0.25, 0.5, 0.75]), 3),
)

print("\ncut  recall-of-policy  wg3_relevant corpus-wide")
n_corpus = int((have & dated).sum())
prod_dated = pd.Series(prod, index=ids)[have & dated]
for c in CAND:
    rec = (pol & (prod >= c)).sum() / pol.sum()
    n_cut = int((prod_dated >= c).sum())
    print(
        f"{c:5.2f}       {rec:6.1%}            {n_cut:,} "
        f"({n_cut / n_corpus:.0%} of corpus)"
    )

topic_wg = pd.read_csv("report/tables/topic_wg_a200.csv").set_index("topic")
prim = topic_wg["primary_wg"].to_numpy()
share3 = topic_wg["wg3_share"].to_numpy()
dom = top_idx[:, 0]
print(
    "\nmean wg3_share of dominant topic: "
    f"hits {share3[dom[hit]].mean():.2f}, misses {share3[dom[miss]].mean():.2f}"
)

mi = np.where(miss)[0]
order = mi[np.argsort(prod[mi], kind="stable")]
pick = order[np.linspace(0, len(order) - 1, N_SHOW, dtype=int)]
metas = wte.fetch_meta(list(ids.to_numpy()[pick]))
tw = wte.top_words()
print(f"\n{N_SHOW} sampled misses (policy-relevant, score < {WG3_CUT:.3f}):")
for i in pick:
    title, text = metas.get(ids[i], ("<no title>", ""))
    sec = sec_argmax.get(ids[i], "")
    sec = sec.removeprefix(SECTOR_PREFIX)[2:] if isinstance(sec, str) and sec else sec
    print(
        f"\n  score={prod[i]:.3f}  relevant={pred['relevant'].iloc[i]:.2f}"
        f"  sector={sec}  {str(title)[:100]}"
    )
    print(f"    abstract: {' '.join(str(text).split())[:220]}...")
    for j in range(3):
        t = top_idx[i, j]
        print(
            f"    topic {t:3d} wg={prim[t]} wg3_share={share3[t]:.2f}  {tw.get(t, '')}"
        )
