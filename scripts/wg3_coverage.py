"""Document-level WG classifier + IPCC coverage ("misses") analysis.

Intensity classifier: score_wg(doc) = Σ_t renormalised loading · shrunk
share(t, wg), where topic shares are shrunk toward a FLAT 1/3 prior with a
pseudo-count (flat, not corpus-wide: shrinking toward the corpus share would
import IPCC's own coverage bias — the very thing the 2020 paper says not to
assume). A topic with 20 loading-weighted citations contributes mostly
"no evidence", a topic with 500 contributes its empirical share. WG(doc) =
argmax — no threshold. Weakly-evidenced documents show up as small margins
between scores, reported for audit.

Coverage analysis: for each WG, what fraction of the documents classified
into it did the AR6 actually cite, and how many citations did they get —
the modern form of the 2020 claim that misses are larger on the WG III side.

Writes report/tables/wgiii_coverage_a200.csv (year x class x own-WG
coverage). Run assign first (needs the n_docs_wg* evidence columns).

EXPLORATORY: committed as audit evidence; the AR6-misses finding it supports
(misses larger on the WG III side) is not yet in the write-up, and nothing in
the reporting pipeline consumes the coverage table.

    uv run python scripts/wg3_coverage.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from climate_literature.constants import IPCC_DATA, TABLES_DIR, TOPICS_DATA

TAG = "K200_a0.0"
K = 200
DT_DIR = TOPICS_DATA / "doc_topics" / TAG
A_PRIOR = 50.0  # pseudo-citations of flat prior per topic
CACHE = Path("/tmp/wg3_doc_wg_scores.npz")
OUT = TABLES_DIR / "wgiii_coverage_a200.csv"


def shrunk_shares(a: float) -> np.ndarray:
    t = pd.read_csv(TABLES_DIR / "topic_wg_a200.csv").set_index("topic")
    S = t[["score_wg1", "score_wg2", "score_wg3"]].reindex(range(K)).to_numpy(float)
    return (S + a / 3) / (S.sum(axis=1, keepdims=True) + a)


def doc_scores() -> tuple[pd.Index, np.ndarray]:
    """(ids, (N, 3) content scores under the A_PRIOR-shrunk shares)."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return pd.Index(z["ids"]), z["S"]
    shares = shrunk_shares(A_PRIOR)
    ids, SS = [], []
    for f in sorted(DT_DIR.glob("shard=*/*.parquet")):
        cols = ["item_id"] + [f"topic_{t}" for t in range(K)]
        df = pq.read_table(f, columns=cols).to_pandas().set_index("item_id")
        V = df.to_numpy(copy=True).astype(np.float64)
        V /= np.maximum(V.sum(axis=1, keepdims=True), 1e-9)
        ids.append(df.index)
        SS.append(V @ shares)
    ids, S = pd.Index(np.concatenate(ids)), np.vstack(SS).astype(np.float32)
    np.savez_compressed(CACHE, ids=ids.to_numpy(), S=S)
    return ids, S


def main():
    spec = importlib.util.spec_from_file_location(
        "ws", Path(__file__).with_name("wg3_score.py")
    )
    assert spec is not None and spec.loader is not None
    ws = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ws)

    ids, S = doc_scores()
    cls = S.argmax(axis=1) + 1  # 1, 2, 3
    srt = np.sort(S, axis=1)
    margin = srt[:, 2] - srt[:, 1]

    t = pd.read_csv(TABLES_DIR / "topic_wg_a200.csv")
    if "n_docs_wg3" in t.columns:
        print("topic evidence audit — distinct cited docs per owning WG:")
        for wg in (1, 2, 3):
            g = t[t.primary_wg == wg][f"n_docs_wg{wg}"]
            print(
                f"  WG {wg} topics: median {g.median():5.0f}  min {g.min():4d}  "
                f"topics with <50: {(g < 50).sum()}"
            )
    else:
        print("run `topics.wg assign` first for the n_docs_wg* evidence columns")

    print(f"\ncorpus: {len(ids):,} documents  (flat-prior pseudo-count a={A_PRIOR})")
    lab = pd.Series(cls).value_counts().sort_index()
    print(f"documents per class WG: {lab.to_dict()}")
    for thr in (0.02, 0.05):
        print(
            f"  margin < {thr}: {(margin < thr).mean():.1%} of corpus "
            "(weak claim between top two WGs)"
        )

    dwg = pd.read_parquet(IPCC_DATA / "ipcc_doc_wg.parquet").set_index("item_id")
    dwg = dwg.reindex(ids)
    cites = dwg[["wg1_n", "wg2_n", "wg3_n"]].fillna(0.0).to_numpy(np.float32)
    cited_any = dwg["n_refs"].notna().to_numpy()

    years = ws.year_table().reindex(ids).to_numpy(dtype="float64")

    print("\ncoverage by class (what AR6 actually cited):")
    print(
        f"{'class':>6} {'docs':>9} {'cited_own':>10} {'cites/doc':>10} "
        f"{'cited_any':>10} {'refs/doc':>9}"
    )
    for wg in (1, 2, 3):
        g = cls == wg
        own = cites[g, wg - 1]
        print(
            f"{wg:>6} {g.sum():>9,} {(own > 0).mean():>10.1%} "
            f"{own.mean():>10.2f} {cited_any[g].mean():>10.1%} "
            f"{dwg['n_refs'].fillna(0).to_numpy()[g].mean():>9.2f}"
        )

    ok = pd.Series((years >= 1985) & (years <= 2024) & (cls > 0))
    tab = pd.DataFrame(
        {
            "year": pd.Series(years).astype("Int64")[ok],
            "class_wg": pd.Series(cls)[ok],
            "cited_own": (cites[np.arange(len(cls)), cls - 1] > 0)[ok.to_numpy()],
        }
    )
    yr = tab.groupby(["class_wg", "year"]).agg(
        docs=("cited_own", "size"), cited_own=("cited_own", "sum")
    )
    yr["share_cited_own"] = yr.cited_own / yr.docs
    yr.reset_index().to_csv(OUT, index=False)
    piv = yr["share_cited_own"].unstack("class_wg").round(3)
    print("\nshare of class documents cited by their own WG, by year:")
    print(piv.to_string())
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
