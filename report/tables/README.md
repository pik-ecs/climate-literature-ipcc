# report/tables — data dictionary

All sheets describe the K200_a0.0 release (200-topic NMF over ~1.4M
climate-relevant Scopus records; see `report/topic_selection_memo.md` for how
the model was selected). Corpus = one row per unique `item_id` in
`data/predictions`.

## WG III document counts

**`wgiii_docs_by_year_a200.csv`** — one row per publication year (1985–2025).

| column | meaning |
| --- | --- |
| `docs` | corpus records with that cover year |
| `wg3_relevant` | WG III-relevant: topic content score ≥ 1/3 **or** policy-relevant (`climate_literature.topics.wg`) |
| `policy` | classified climate-policy-relevant by the Callaghan et al. 2024 (mitigation) classifier, `relevant` > 0.5 |
| `wg3_policy_relevant` | WG III-relevant ∩ policy-relevant — since the union rule this is by construction **equal to `policy`**, kept for continuity with older shares of this file |
| `wg3_relevant_share` | `wg3_relevant / docs` |

⚠️ `policy` and `wg3_policy_relevant` are counts of *mitigation-policy*
papers — a strict subset of `wg3_relevant`. They are **not** the marginal
count of "all papers of WG III interest"; don't compare `policy` to the
`wg3_relevant` prose totals.

**`wgiii_doc_counts_a200.csv`** — headline totals (`corpus_docs`,
`wg3_relevant`) over the whole corpus, all years.

**`wgiii_sector_by_year_a200.csv`** — one row per publication year (1987–2025),
one column per WG III sector plus `unclassified` if ever present, counts of
policy-relevant papers whose sector argmax is that sector; `policy_total` is
the row sum and equals the `policy` column of `wgiii_docs_by_year_a200.csv`.
Sector = argmax over the seven sector scores (cascade in
`classify/predict.py`), so only policy-relevant papers appear. Divide a row
by `policy_total` for shares; shares are the safer read across years, since
raw counts track Scopus coverage growth.

**`wgiii_coverage_a200.csv`** — exploratory (see `scripts/wg3_coverage.py`;
not consumed by the pipeline): one row per (WG class, year) from a
three-way shrunk classifier; `docs` classified into `class_wg`, `cited_own`
how many the AR6 cited, `share_cited_own` = IPCC coverage of the class.

## Topic-level sheets

**`topic_wg_a200.csv`** — one row per topic: `top_words`, `docs_equiv`
(corpus loading mass), IPCC citation evidence (`n_cited_docs`,
`n_docs_wg1/2/3` citing documents, `ipcc_cites_total`, `score_wg*`
citation-weighted, `wg1/2/3_share` shares, `primary_wg` argmax).
`wg3_share` feeds the document content score in `topics.wg`.

## Model-selection / audit sheets (K-selection era, various K)

`baseline_*_vs_2019.csv`, `ladder_a0.0.csv`, `coherence*.csv`,
`audit_*.csv`, `ipcc_ar6_spotcheck.csv` — artifacts of the K=80–220 selection
and audit runs, retained as evidence; the selected release is K200_a0.0.
Not updated after model changes unless the audit is deliberately rerun.
