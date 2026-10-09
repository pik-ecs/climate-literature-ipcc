"""Inspect WG III-cited documents that the product score misses at the adopted cut.

Prints the score distribution of the positives, the dominant-topic WG mix of
hits vs misses (mechanism), and sampled misses with title, abstract and their
top topics (loading x topic wg3_share) to see where their mass sits.

Needs /tmp/wg3_score.npz and /tmp/wg3_threshold_stat.npz (see wg3_score.py).

    uv run python scripts/wg3_misses.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from climate_literature.constants import IPCC_DATA, TABLES_DIR
from climate_literature.topics.wg import WG3_CUT

T = WG3_CUT
N_SHOW = 12

Z = np.load("/tmp/wg3_score.npz", allow_pickle=True)
ids = pd.Index(Z["ids"])
prod = Z["score"].astype(np.float64)
ZM = np.load("/tmp/wg3_threshold_stat.npz", allow_pickle=True)
assert (ZM["ids"] == ids.to_numpy()).all()
top_idx, top_val = ZM["top_idx"], ZM["top_val"].astype(np.float64)

spec = importlib.util.spec_from_file_location(
    "wte", Path(__file__).with_name("wg3_threshold_examples.py")
)
assert spec is not None and spec.loader is not None
wte = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wte)

dwg = pd.read_parquet(IPCC_DATA / "ipcc_doc_wg.parquet").set_index("item_id")
dwg = dwg.reindex(ids).fillna(0.0)
w3 = dwg["wg3_n"].to_numpy()
pos = w3 > 0
miss = pos & (prod < T)
hit = pos & (prod >= T)

print(
    f"positives: {pos.sum():,}   misses at t={T:.3f}: {miss.sum():,}  "
    f"(recall {hit.sum() / pos.sum():.1%})"
)
print(
    "product score of positives (quantiles):",
    np.round(np.quantile(prod[pos], [0.1, 0.25, 0.5, 0.75, 0.9]), 3),
)

topic_wg = pd.read_csv(TABLES_DIR / "topic_wg_a200.csv").set_index("topic")
prim = topic_wg["primary_wg"].to_numpy()
share3 = topic_wg["wg3_share"].to_numpy()
dom = top_idx[:, 0]
print("\ndominant topic primary_wg of positives:")
for name, g in (("hits", hit), ("misses", miss)):
    print(
        f"  {name:7s} {pd.Series(prim[dom[g]]).value_counts().sort_index().to_dict()}"
    )
print(
    f"  mean wg3_share of dominant topic: hits {share3[dom[hit]].mean():.2f}, "
    f"misses {share3[dom[miss]].mean():.2f}"
)

# sample misses spread across their score range
mi = np.where(miss)[0]
order = mi[np.argsort(prod[mi], kind="stable")]
pick = order[np.linspace(0, len(order) - 1, N_SHOW, dtype=int)]
metas = wte.fetch_meta(list(ids.to_numpy()[pick]))
tw = wte.top_words()
print(f"\n{N_SHOW} sampled misses (wg3-cited, product < {T:.3f}), low to high score:")
for i in pick:
    title, text = metas.get(ids[i], ("<no title>", ""))
    print(f"\n  score={prod[i]:.3f}  wg3_cites={int(w3[i])}  {str(title)[:105]}")
    print(f"    abstract: {' '.join(str(text).split())[:200]}...")
    for j in range(3):
        t = top_idx[i, j]
        print(
            f"    topic {t:3d} w={top_val[i, j]:.3f} wg={prim[t]} "
            f"wg3_share={share3[t]:.2f}  {tw.get(t, '')}"
        )
