"""Topic → Working Group assignment and the resulting document counts.

The 2020 method, redone on K200_a0.0. ipcc_refs.match tells us, for each
corpus document, how many AR6 WG I / II / III chapters cited it (or the
document it matches). Here those per-document citations are aggregated over
topic loadings:

    wg_score(topic, wg) = Σ_docs H[doc, topic] · citations(doc, wg)

and each topic's `primary_wg` is the argmax — the same rule as `primary_wg`
in the 2019 topics.csv. A stricter `wg3_share` (WG III's fraction of the
topic's total IPCC citations) supports a "typically cited by WG III" cut
beyond bare argmax.

`assign` writes report/tables/topic_wg_a200.csv; `counts` writes
report/tables/wgiii_doc_counts_a200.csv with the headline document counts:

  hard   dominant topic (argmax loading) is a WG III-primary topic
  soft   ≥ 0.10 loading on any WG III-primary topic (K200's audit threshold)
  strict hard/soft over the stricter WG III topics (wg3_share > 0.5)

counts also splits by publication year and cross-tabs against the
policy-relevance classifier and its sector argmax (reporting/numbers.py
convention) as a convergence check. Topic labels come from AR6-era citations,
so post-2021 years describe documents sitting in topics the AR6 WG III cited,
not fresh citations.

Run from the repo root:

    uv run python -m climate_literature.topics.wg assign
    uv run python -m climate_literature.topics.wg counts
"""

import numpy as np
import pandas as pd
import typer

from climate_literature.constants import IPCC_DATA, TABLES_DIR, TOPICS_DATA

app = typer.Typer(help="Assign topics to IPCC working groups over K200 loadings.")

TAG = "K200_a0.0"
K = 200
DT_DIR = TOPICS_DATA / "doc_topics" / TAG
TOPIC_WG = TABLES_DIR / "topic_wg_a200.csv"
COUNTS = TABLES_DIR / "wgiii_doc_counts_a200.csv"
DOC_WG = IPCC_DATA / "ipcc_doc_wg.parquet"  # DVC-tracked, see ipcc_refs
ORPHAN = 0.1  # same soft-membership threshold as the K200 audit


def _load_loadings() -> pd.DataFrame:
    """The full 1.44M × 200 doc-topic matrix (float32), item_id-indexed."""
    shards = sorted(DT_DIR.glob("shard=*/*.parquet"))
    assert shards, f"no scores under {DT_DIR}"
    cols = ["item_id"] + [f"topic_{t}" for t in range(K)]
    H = pd.concat(
        pd.read_parquet(f, columns=cols).set_index("item_id") for f in shards
    ).astype(np.float32)
    return H


def _top_words(top_n: int = 8) -> pd.Series:
    words = pd.read_parquet(
        DT_DIR.parent.parent / "models" / TAG / "topic_words.parquet"
    )
    words = words.sort_values(["topic", "score"], ascending=[True, False])
    head = words.groupby("topic").cumcount() < top_n
    return words[head].groupby("topic")["term"].apply(" ".join)


def _load_doc_wg() -> pd.DataFrame:
    return pd.read_parquet(DOC_WG).set_index("item_id")


def _load_meta() -> pd.DataFrame:
    """Per-document publication_year + policy-classifier columns.

    Sector-score columns only exist in some prediction partitions, so
    reporting.plots.load_prediction_columns (per-file schema check) is the
    right loader — same one the WGIII growth numbers use.
    """
    from climate_literature.reporting.plots import (
        load_prediction_columns,
        sector_columns,
    )

    meta = load_prediction_columns(
        ["item_id", "publication_year", "relevant", *sector_columns()]
    )
    return meta.drop_duplicates("item_id").set_index("item_id")


@app.command()
def assign() -> None:
    """Score topics by citing WG and label primary_wg -> topic_wg_a200.csv."""
    H = _load_loadings()
    dwg = _load_doc_wg().reindex(H.index)
    cited = dwg["n_refs"].notna().to_numpy()
    V = H.to_numpy()
    Vc = V[cited]
    W = dwg[["wg1_n", "wg2_n", "wg3_n"]].fillna(0.0).to_numpy(dtype=np.float32)[cited]
    S = Vc.T @ W  # (K, 3): topic loadings weighted by per-doc WG citations
    mass = V.sum(axis=0)
    n_cited_docs = (Vc > 0).sum(axis=0)

    row_sums = S.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = S / row_sums[:, None]
    primary = np.where(row_sums > 0, S.argmax(axis=1) + 1, 0)
    out = pd.DataFrame(
        {
            "topic": np.arange(K),
            "top_words": pd.Series(_top_words()).reindex(range(K)).fillna(""),
            "docs_equiv": mass.round(0).astype(int),
            "n_cited_docs": n_cited_docs,
            "ipcc_cites_total": row_sums.astype(int),
            "score_wg1": S[:, 0].astype(int),
            "score_wg2": S[:, 1].astype(int),
            "score_wg3": S[:, 2].astype(int),
            "wg1_share": shares[:, 0],
            "wg2_share": shares[:, 1],
            "wg3_share": shares[:, 2],
            "primary_wg": primary,
        }
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TOPIC_WG, index=False)

    label = out["primary_wg"].value_counts().sort_index()
    typer.echo(f"topics per WG: {label.to_dict()}  (0 = no IPCC citations)")
    typer.echo(
        f"topics with <30 cited docs (label weak): {(out['n_cited_docs'] < 30).sum()}"
    )
    wg3 = out[out["primary_wg"] == 3].sort_values("docs_equiv", ascending=False)
    typer.echo(f"\ntop WG3-primary topics ({len(wg3)} total):")
    for _, r in wg3.head(10).iterrows():
        typer.echo(
            f"  T{r.topic:3d} {r.docs_equiv:7,d} docs  "
            f"wg3_share={r.wg3_share:.2f}  {r.top_words}"
        )
    typer.echo(f"-> {TOPIC_WG}")


def _dominant_and_membership(
    H: pd.DataFrame, wg3_topics: np.ndarray, wg3_strict: np.ndarray
) -> dict[str, np.ndarray]:
    V = H.to_numpy()
    # renormalize rows before thresholding: raw NMF rows do not sum to 1
    # (the audit script does the same); argmax is scale-invariant per row
    V = V / np.maximum(V.sum(axis=1, keepdims=True), 1e-9)
    dom = V.argmax(axis=1)
    prim = pd.read_csv(TOPIC_WG).set_index("topic")["primary_wg"].to_numpy()
    dom_wg = prim[dom]
    soft3 = (V[:, wg3_topics] >= ORPHAN).any(axis=1)
    soft3s = (V[:, wg3_strict] >= ORPHAN).any(axis=1)
    hard3s = np.isin(dom, wg3_strict)
    return {
        "dom_wg": dom_wg,
        "hard3": dom_wg == 3,
        "hard3_strict": hard3s,
        "soft3": soft3,
        "soft3_strict": soft3s,
    }


@app.command()
def counts() -> None:
    """Count documents in WG-labeled topics -> wgiii_doc_counts_a200.csv."""
    H = _load_loadings()
    topic_wg = pd.read_csv(TOPIC_WG)
    wg3 = topic_wg.loc[topic_wg["primary_wg"] == 3, "topic"].to_numpy()
    wg3s = topic_wg.loc[
        (topic_wg["primary_wg"] == 3) & (topic_wg["wg3_share"] > 0.5), "topic"
    ].to_numpy()
    m = _dominant_and_membership(H, wg3, wg3s)
    n = len(H)

    typer.echo(f"corpus: {n:,} documents")
    for key in ("hard3", "hard3_strict", "soft3", "soft3_strict"):
        typer.echo(f"  {key:13s} {m[key].sum():>8,}  ({m[key].mean():.1%})")
    dom_counts = pd.Series(m["dom_wg"]).value_counts().sort_index()
    typer.echo(f"dominant-topic docs per WG: {dom_counts.to_dict()}")

    # year split + policy cross-tab
    meta = _load_meta().reindex(H.index)
    years = meta["publication_year"].to_numpy(dtype="float64")
    relevant = (meta["relevant"].to_numpy(dtype="float64") > 0.5).astype(bool)
    sector_cols = [c for c in meta.columns if c.startswith("8 - ")]
    sec = meta[sector_cols].to_numpy(dtype="float32")
    sec_argmax = pd.Series(
        [
            sector_cols[i].split(". ")[1] if len(sector_cols) else ""
            for i in range(len(sector_cols))
        ]
    )

    year_tab = pd.DataFrame(
        {
            "year": pd.Series(years, dtype="float64").astype("Int64"),
            "hard3": m["hard3"],
            "soft3": m["soft3"],
            "relevant": relevant,
        }
    )
    yr = (
        year_tab[year_tab["year"].between(1990, 2025)]
        .groupby("year")
        .agg(
            docs=("hard3", "size"), wg3_hard=("hard3", "sum"), wg3_soft=("soft3", "sum")
        )
    )
    yr["wg3_hard_share"] = yr["wg3_hard"] / yr["docs"]

    typer.echo("\npolicy-relevance convergence (classifier of Callaghan et al. 2024):")
    typer.echo(
        f"  relevant share within hard3 set:    {relevant[m['hard3']].mean():.1%}"
    )
    typer.echo(f"  relevant share corpus-wide:     {relevant.mean():.1%}")
    if len(sector_cols):
        # argmax only over policy-relevant hard3 documents: the sector scores
        # are only trustworthy where the classifier's cascade engaged (the
        # rule used in reporting/numbers.py)
        sec_names = sec_argmax[sec.argmax(axis=1)]
        sel = m["hard3"] & relevant
        typer.echo(f"\n  sector argmax over relevant∩hard3 documents ({sel.sum():,}):")
        typer.echo(pd.Series(sec_names[sel]).value_counts().to_string())

    out = {
        "corpus_docs": [n],
        "hard3": [int(m["hard3"].sum())],
        "hard3_strict": [int(m["hard3_strict"].sum())],
        "soft3": [int(m["soft3"].sum())],
        "soft3_strict": [int(m["soft3_strict"].sum())],
    }
    pd.DataFrame(out).to_csv(COUNTS, index=False)
    yr.reset_index().to_csv(
        COUNTS.with_name("wgiii_docs_by_year_a200.csv"), index=False
    )
    typer.echo(f"-> {COUNTS} (+ yearly table)")


if __name__ == "__main__":
    app()
