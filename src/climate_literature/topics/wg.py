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
report/tables/wgiii_doc_counts_a200.csv with the headline document counts.
WG III membership is granted by either of two complementary instruments: a
content score from IPCC citations,

    score(doc) = Σ_t renormalised loading(doc, t) · wg3_share(t)

cut at 1/3 (more WG III content than an equal three-way split — an absolute
anchor independent of this corpus's citation mix; against AR6-cited ground
truth this cut recovers 62% of WG III-cited documents while flagging only
1.2% of WG I-cited literature, scripts/wg3_tune.py), OR the mitigation-policy
classifier of Callaghan et al. 2024. The union matters because the score
sees subject vocabulary as the IPCC cites it, and application-focused
mitigation papers (REDD+, GHG accounting, carbon markets) sit in forest/land
topics WG II also cites: the cut alone misses 16% of classifier positives
(scripts/wg3_policy_misses.py), and no convex reweighting of the score
recovers them without doubling WG I contamination (scripts/wg3_gamma.py).
The union lifts AR6 recall to 68% at a 2.7% WG I flag rate. The yearly sheet
includes one column per definition (`wg3_score_cut`, `policy`,
`policy_or_wg3`, `policy_and_wg3`) so sheet users can pick an instrument;
`policy_or_wg3` is the report headline.

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
WG3_CUT = 1 / 3  # more WG III content than an equal three-way split of a doc


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
    # real evidence per topic per WG: distinct cited docs with positive
    # loading on the topic (ipcc_cites_total below is loading-weighted and
    # fractional — a magnitude, not a count)
    n_docs_wg = (Vc > 0).T.astype(np.float32) @ (W > 0).astype(np.float32)  # (K, 3)

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
            "n_docs_wg1": n_docs_wg[:, 0].round(0).astype(int),
            "n_docs_wg2": n_docs_wg[:, 1].round(0).astype(int),
            "n_docs_wg3": n_docs_wg[:, 2].round(0).astype(int),
            "ipcc_cites_total": row_sums.round(0).astype(int),
            "score_wg1": S[:, 0].round(0).astype(int),
            "score_wg2": S[:, 1].round(0).astype(int),
            "score_wg3": S[:, 2].round(0).astype(int),
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


def _score_and_dominant(H: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """WG III content score per document, and its dominant topic's WG."""
    V = H.to_numpy()
    # renormalize rows first: raw NMF rows do not sum to 1 (the audit script
    # does the same), so the score is a proper content fraction
    V = V / np.maximum(V.sum(axis=1, keepdims=True), 1e-9)
    topic_wg = pd.read_csv(TOPIC_WG).set_index("topic")
    dom_wg = topic_wg["primary_wg"].to_numpy()[V.argmax(axis=1)]
    share3 = topic_wg["wg3_share"].reindex(range(K)).to_numpy(dtype=np.float64)
    return V @ share3, dom_wg


@app.command()
def counts() -> None:
    """Count documents in WG-labeled topics -> wgiii_doc_counts_a200.csv."""
    H = _load_loadings()
    score3, dom_wg = _score_and_dominant(H)
    score_cut = score3 >= WG3_CUT

    # year split + policy cross-tab
    meta = _load_meta().reindex(H.index)
    years = meta["publication_year"].to_numpy(dtype="float64")
    policy = (meta["relevant"].to_numpy(dtype="float64") > 0.5).astype(bool)

    # membership by either instrument: the citation-share score sees content
    # through the IPCC's own citing lens; the mitigation-policy classifier
    # catches application-focused papers whose vocabulary WG II also cites
    # (scripts/wg3_policy_misses.py; convex score variants win nothing:
    # scripts/wg3_gamma.py)
    wg3_rel = score_cut | policy
    n = len(H)

    typer.echo(f"corpus: {n:,} documents")
    typer.echo(f"  score-cut     {score_cut.sum():>8,}  ({score_cut.mean():.1%})")
    typer.echo(f"  wg3_relevant  {wg3_rel.sum():>8,}  ({wg3_rel.mean():.1%})")
    dom_counts = pd.Series(dom_wg).value_counts().sort_index()
    typer.echo(f"dominant-topic docs per WG: {dom_counts.to_dict()}")
    sector_cols = [c for c in meta.columns if c.startswith("8 - ")]
    sec = meta[sector_cols].to_numpy(dtype="float32")
    sec_argmax = pd.Series(
        [
            sector_cols[i].split(". ")[1] if len(sector_cols) else ""
            for i in range(len(sector_cols))
        ]
    )

    # one column per definition, so sheet users pick an instrument:
    # wg3_score_cut is content score >= 1/3, policy is the classifier,
    # plus their union (the report headline) and intersection
    year_tab = pd.DataFrame(
        {
            "year": pd.Series(years, dtype="float64").astype("Int64"),
            "wg3_score_cut": score_cut,
            "policy": policy,
            "policy_or_wg3": wg3_rel,
            "policy_and_wg3": score_cut & policy,
        }
    )
    # one row per publication year, all years, plus a blank-year row for
    # records with no publication year; rows sum to corpus_docs, and the
    # readers in reporting/ apply the 1985..LAST_COMPLETE_YEAR report window
    yr = year_tab.groupby("year", dropna=False).agg(
        docs=("policy_or_wg3", "size"),
        wg3_score_cut=("wg3_score_cut", "sum"),
        policy=("policy", "sum"),
        policy_or_wg3=("policy_or_wg3", "sum"),
        policy_and_wg3=("policy_and_wg3", "sum"),
    )
    yr["policy_or_wg3_share"] = yr["policy_or_wg3"] / yr["docs"]

    typer.echo("\npolicy-relevance convergence (classifier of Callaghan et al. 2024):")
    typer.echo(f"  policy share within wg3_relevant set: {policy[wg3_rel].mean():.1%}")
    typer.echo(f"  policy share corpus-wide:           {policy.mean():.1%}")
    if len(sector_cols):
        # argmax over policy-relevant documents: the sector scores are only
        # trustworthy where the classifier's cascade engaged (the rule used
        # in reporting/numbers.py)
        sec_names = sec_argmax[sec.argmax(axis=1)]
        sel_all = wg3_rel & policy
        # the echo mirrors the report window (matches the figure); the sheet
        # includes every publication year, like the yearly table
        sel_win = sel_all & pd.Series(years).between(1985, 2025).fillna(False)
        typer.echo(
            f"\n  sector argmax over policy documents, 1985-2025 ({sel_win.sum():,}):"
        )
        typer.echo(pd.Series(sec_names[sel_win]).value_counts().to_string())

        # sector x year counts; shares are one divide away in a spreadsheet,
        # and raw counts keep the audit trail explicit
        sy = pd.DataFrame(
            {
                "year": pd.array(years[sel_all], dtype="Int64"),  # <NA> = none
                "sector": np.asarray(sec_names[sel_all], dtype=object),
            }
        )
        sy["sector"] = sy["sector"].fillna("unclassified")
        pivot = (
            sy.groupby(["year", "sector"], observed=True, dropna=False)
            .size()
            .unstack("sector", fill_value=0)
        )
        pivot = pivot[pivot.sum().sort_values(ascending=False).index]
        pivot["policy_total"] = pivot.sum(axis=1)
        sec_out = COUNTS.with_name("wgiii_sector_by_year_a200.csv")
        pivot.reset_index().to_csv(sec_out, index=False)
        typer.echo(f"-> {sec_out} (sector x year counts)")

    out = {
        "corpus_docs": [n],
        "wg3_score_cut": [int(score_cut.sum())],
        "policy": [int(policy.sum())],
        "policy_or_wg3": [int(wg3_rel.sum())],
        "policy_and_wg3": [int((score_cut & policy).sum())],
    }
    pd.DataFrame(out).to_csv(COUNTS, index=False)
    yr.reset_index().to_csv(
        COUNTS.with_name("wgiii_docs_by_year_a200.csv"), index=False
    )
    typer.echo(f"-> {COUNTS} (+ yearly table)")


if __name__ == "__main__":
    app()
