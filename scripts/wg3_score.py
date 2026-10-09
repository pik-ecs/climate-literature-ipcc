"""Continuous WG III score per document: score = Σ_t loading · wg3_share(topic).

Rows of the K200 loadings are renormalised to sum to 1, so the score is the
WG III content fraction of a document — the topic-weighted average of the
per-topic shares from report/tables/topic_wg_a200.csv. Prints the
distribution, how the binary count moves across candidate cuts, per-year
mean scores, and rescores the near-threshold examples from
scripts/wg3_threshold_examples.py for eyeballing.

    uv run python scripts/wg3_score.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from climate_literature.constants import PREDICTIONS_DATA, TABLES_DIR, TOPICS_DATA

TAG = "K200_a0.0"
K = 200
DT_DIR = TOPICS_DATA / "doc_topics" / TAG
CACHE = Path("/tmp/wg3_score.npz")
CUTS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]


def load_scores():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return pd.Index(z["ids"]), z["score"]
    share = pd.read_csv(TABLES_DIR / "topic_wg_a200.csv").set_index("topic")
    s3 = share["wg3_share"].reindex(range(K)).to_numpy(np.float64)
    ids, sc = [], []
    for f in sorted(DT_DIR.glob("shard=*/*.parquet")):
        cols = ["item_id"] + [f"topic_{t}" for t in range(K)]
        df = pq.read_table(f, columns=cols).to_pandas().set_index("item_id")
        V = df.to_numpy(copy=True).astype(np.float64)
        V /= np.maximum(V.sum(axis=1, keepdims=True), 1e-9)
        ids.append(df.index)
        sc.append(V @ s3)
    ids, score = pd.Index(np.concatenate(ids)), np.concatenate(sc)
    np.savez_compressed(CACHE, ids=ids.to_numpy(), score=score)
    return ids, score


def year_table():
    """item_id -> publication_year from the prediction partitions."""
    frames = [
        pq.read_table(f, columns=["item_id", "publication_year"]).to_pandas()
        for f in PREDICTIONS_DATA.rglob("*.parquet")
    ]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("item_id")
    return df.set_index("item_id")["publication_year"]


def main():
    ids, score = load_scores()
    n = len(ids)
    print(
        f"corpus: {n:,} documents   score: mean={score.mean():.3f} "
        f"median={np.median(score):.3f} max={score.max():.3f}"
    )

    hist, edges = np.histogram(score, bins=np.arange(0.0, 0.601, 0.02))
    print("\ndocs per 0.02 bin of the WG III content fraction:")
    for i, c in enumerate(hist):
        print(f"  [{edges[i]:4.2f},{edges[i + 1]:4.2f})  {c:>6,}")

    print("\nWG III document count vs cut:")
    for t in CUTS:
        k = int((score >= t).sum())
        print(f"  ≥{t:5.2f}  {k:>8,}  ({k / n:.1%})")

    yr = year_table().reindex(ids).to_numpy(dtype="float64")
    ok = (yr >= 1985) & (yr <= 2024)
    mean_tab = pd.DataFrame({"y": yr[ok], "s": score[ok]}).groupby("y")["s"].mean()
    print("\nmean WG III content fraction by publication year:")
    for y, v in mean_tab.items():
        print(f"  {int(y)}  {v:.3f}")

    # rescore the twelve examples from the threshold-examples script
    spec = importlib.util.spec_from_file_location(
        "wte", Path(__file__).with_name("wg3_threshold_examples.py")
    )
    assert spec is not None and spec.loader is not None
    wte = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wte)
    s_ids, mx, mx_topic, top_idx, top_val, topic_wg = wte.stream_statistic()
    tw = wte.top_words()
    pos = pd.Series(np.arange(len(s_ids)), index=s_ids)
    score_lookup = pd.Series(score, index=ids)

    orph = wte.ORPHAN
    for label, lo, hi in [("BELOW", 0.09, orph), ("ABOVE", orph, 0.11)]:
        sel_ids = s_ids[(mx >= lo) & (mx < hi)]
        sel_mx = mx[(mx >= lo) & (mx < hi)]
        order = np.argsort(sel_mx, kind="stable")
        picks = sel_ids[order[np.linspace(0, len(order) - 1, wte.N_EACH, dtype=int)]]
        metas = wte.fetch_meta(list(picks))
        print(f"\n=== examples previously {label} the 0.10 max-loading cut ===")
        for doc in picks:
            p = pos[doc]
            title = str(metas.get(doc, ("<no title>", ""))[0])
            print(f"\n  score={score_lookup[doc]:.3f}  max={mx[p]:.4f}  {title[:100]}")
            t = mx_topic[p]
            print(
                f"    top wg3 topic {t}: load={top_val[p].max():.3f} "
                f"wg3_share={topic_wg['wg3_share'][t]:.2f}  {tw.get(t, '')}"
            )


if __name__ == "__main__":
    main()
