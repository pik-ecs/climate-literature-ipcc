# report/tables — data dictionary

All sheets describe the K200_a0.0 release (200-topic NMF over ~1.4M
climate-relevant Scopus records; see `report/topic_selection_memo.md` for how
the model was selected). Corpus = one row per unique `item_id` in
`data/predictions`.

## WG III document counts

**`wgiii_docs_by_year_a200.csv`** — one row per publication year. All years are
included, plus a blank-year row for records with no publication year; the rows
sum to `corpus_docs`. The report numbers use the window 1985–2025, applied in
`reporting/`. WG III-relevance has two instruments; each combination has its
own column (rule and validation: `climate_literature.topics.wg`).

| column | meaning |
| --- | --- |
| `docs` | corpus records with that cover year |
| `wg3_score_cut` | content score ≥ 1/3: loading-weighted average of per-topic WG III citation shares from AR6 reference lists |
| `policy` | classified climate-policy-relevant by the Callaghan et al. 2024 (mitigation) classifier, `relevant` > 0.5 |
| `policy_or_wg3` | either instrument flags the paper — the definition quoted in the report text |
| `policy_and_wg3` | both instruments flag the paper — the strict overlap |
| `policy_or_wg3_share` | `policy_or_wg3 / docs` |

The instruments measure different things and disagree in both directions:
`wg3_score_cut` sees subject vocabulary as the IPCC cites it (and misses
application-focused mitigation papers whose vocabulary WG II also cites),
while `policy` is task-specific and has its own false positives. The report
text uses `policy_or_wg3`; prose totals should not be compared against
`policy` or `policy_and_wg3` columns.

**`wgiii_doc_counts_a200.csv`** — the same columns as corpus totals, all
years, plus `corpus_docs`.

**`wgiii_sector_by_year_a200.csv`** — one row per publication year (all years;
blank year = no publication year), one column per WG III sector plus
`unclassified` if ever present, counts of policy-relevant papers whose sector
argmax is that sector; `policy_total` is the row sum and equals the `policy`
column of `wgiii_docs_by_year_a200.csv`.
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
