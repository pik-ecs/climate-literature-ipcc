"""Eyeball WG III soft-membership examples near the 0.10 threshold.

The soft3 statistic per document is its max renormalised loading over the
WG III-primary topics (climate_literature.topics.wg). This script:

  1. streams that statistic over the K200_a0.0 loadings,
  2. shows how the wg3_relevant count moves across candidate thresholds,
  3. prints sampled documents just below / just above 0.10 with title,
     abstract snippet, and their top topics (with WG labels) for eyeballing.

    uv run python scripts/wg3_threshold_examples.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from climate_literature.constants import PREDICTIONS_DATA, TABLES_DIR, TOPICS_DATA

TAG = "K200_a0.0"
K = 200
DT_DIR = TOPICS_DATA / "doc_topics" / TAG
TOPIC_WG = TABLES_DIR / "topic_wg_a200.csv"
ORPHAN = 0.1
THRESHOLDS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
N_EACH = 6  # examples to show per side of the threshold


CACHE = Path("/tmp/wg3_threshold_stat.npz")


def stream_statistic():
    """Per-doc max renormalised loading on WG III topics + top-3 topics."""
    topic_wg = pd.read_csv(TOPIC_WG).set_index("topic")
    wg3 = topic_wg.index[topic_wg["primary_wg"] == 3].to_numpy()
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return (
            pd.Index(z["ids"]),
            z["mx"],
            z["mx_topic"],
            z["top_idx"],
            z["top_val"],
            topic_wg,
        )
    ids, mx, mx_topic = [], [], []
    top_idx, top_val = [], []
    for f in sorted(DT_DIR.glob("shard=*/*.parquet")):
        cols = ["item_id"] + [f"topic_{t}" for t in range(K)]
        df = pq.read_table(f, columns=cols).to_pandas().set_index("item_id")
        V = df.to_numpy(copy=True).astype(np.float32)
        V /= np.maximum(V.sum(axis=1, keepdims=True), 1e-9)
        sub = V[:, wg3]
        am = sub.argmax(axis=1)
        ids.append(df.index)
        mx.append(sub[np.arange(len(sub)), am])
        mx_topic.append(wg3[am])
        i3 = np.argpartition(-V, kth=3, axis=1)[:, :3]
        rows = np.arange(len(V))[:, None]
        v3 = V[rows, i3]
        order = np.argsort(-v3, axis=1)
        top_idx.append(i3[rows, order])
        top_val.append(v3[rows, order])
    np.savez_compressed(
        CACHE,
        ids=np.concatenate(ids),
        mx=np.concatenate(mx),
        mx_topic=np.concatenate(mx_topic),
        top_idx=np.vstack(top_idx),
        top_val=np.vstack(top_val),
    )
    return (
        pd.Index(np.concatenate(ids)),
        np.concatenate(mx),
        np.concatenate(mx_topic),
        np.vstack(top_idx),
        np.vstack(top_val),
        topic_wg,
    )


def top_words():
    words = pd.read_parquet(
        DT_DIR.parent.parent / "models" / TAG / "topic_words.parquet"
    )
    words = words.sort_values(["topic", "score"], ascending=[True, False])
    head = words.groupby("topic").cumcount() < 6
    return words[head].groupby("topic")["term"].apply(" ".join)


def fetch_meta(sample_ids):
    """Title/abstract for sampled ids, scanning prediction partitions' keys.

    Scores store item_id as raw 16-byte UUID binary, predictions as arrow
    uuid — match on the UUID.bytes form.
    """

    def as_bytes(u):  # arrow uuid arrives as UUID via to_pylist, bytes via to_pandas
        return u if isinstance(u, bytes) else u.bytes

    want = set(sample_ids)
    found = {}
    for part in sorted(PREDICTIONS_DATA.glob("source_file=*")):
        if not want:
            break
        f = next(part.glob("*.parquet"))
        keys = [
            u.bytes
            for u in pq.read_table(f, columns=["item_id"]).column("item_id").to_pylist()
        ]
        if want.isdisjoint(keys):
            continue
        df = pq.read_table(f, columns=["item_id", "title", "text"]).to_pandas()
        df["idb"] = [as_bytes(u) for u in df["item_id"]]
        for _, r in df[df["idb"].isin(want)].iterrows():
            found[r.idb] = (r.title, r.text)
            want.discard(r.idb)
    return found


def main():
    ids, mx, mx_topic, top_idx, top_val, topic_wg = stream_statistic()
    n = len(ids)
    print(f"corpus: {n:,} documents\n")

    print("wg3_relevant count vs threshold (ORPHAN currently 0.10):")
    hist, edges = np.histogram(mx, bins=np.arange(0.0, 0.301, 0.01))
    for t in THRESHOLDS:
        k = int((mx >= t).sum())
        print(f"  ≥{t:5.3f}  {k:>8,}  ({k / n:.1%})")
    print("\ndocs per 0.01 bin of the decision statistic (0–0.30):")
    for i, c in enumerate(hist):
        lo = edges[i]
        mark = "  <-- ORPHAN" if abs(lo - ORPHAN) < 1e-9 else ""
        print(f"  [{lo:4.2f},{lo + 0.01:4.2f})  {c:>6,}{mark}")

    below = ids[(mx >= 0.09) & (mx < ORPHAN)]
    bmx = mx[(mx >= 0.09) & (mx < ORPHAN)]
    above = ids[(mx >= ORPHAN) & (mx < 0.11)]
    amx = mx[(mx >= ORPHAN) & (mx < 0.11)]
    print(
        f"\njust-below band [0.09,0.10): {len(below):,} docs; "
        f"just-above [0.10,0.11): {len(above):,}"
    )

    def stride_pick(arr, k):
        return arr[np.linspace(0, len(arr) - 1, k, dtype=int)]

    # deterministic spread across each band (linspace, not random)
    order_b = np.argsort(bmx, kind="stable")
    order_a = np.argsort(amx, kind="stable")
    picks = [
        ("BELOW (excluded)", stride_pick(order_b, N_EACH), below, bmx),
        ("ABOVE (included)", stride_pick(order_a, N_EACH), above, amx),
    ]

    tw = top_words()
    pos = pd.Series(np.arange(n), index=ids)
    prim = topic_wg["primary_wg"]
    share = topic_wg["wg3_share"]

    for label, sel, band_ids, band_mx in picks:
        meta = fetch_meta(list(band_ids[sel]))
        print(f"\n=== {label} ===")
        for i in sel:
            doc = band_ids[i]
            p = pos[doc]
            title, text = meta.get(doc, ("<title not found>", ""))
            t = band_mx[i]
            print(f"\n  score={t:.4f}  {str(title)[:110]}")
            snippet = " ".join(str(text).split())[:240]
            print(f"    abstract: {snippet}...")
            for j in range(3):
                topic = top_idx[p, j]
                print(
                    f"    topic {topic:3d} w={top_val[p, j]:.3f} wg={prim[topic]} "
                    f"wg3_share={share[topic]:.2f}  {tw.get(topic, '')}"
                )


if __name__ == "__main__":
    main()
