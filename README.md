# climate-literature

This repo describes the size and make-up of the climate literature,
by re-applying the methods of Callaghan et al. (2020) 
to ~1.4M climate-relevant Scopus records (1985–2025)
with a 200-topic NMF model (K200_a0.0), 
policy-relevance and sector classifiers, 
and Working Group shares per topic from the IPCC reference lists. 

## Reproducing the figures and numbers

```
uv sync
dvc pull    # data/ (corpus predictions, topic scores, IPCC citations) is DVC-tracked

uv run python -m climate_literature.topics.wg counts
uv run python -m climate_literature.reporting.plots build-all
uv run python -m climate_literature.reporting.numbers
```

Tables are generated in report/tables/ (columns defined in report/tables/README.md),
figures in figures/. 
The report text in report/wgiii_lit_growth.md is generated from the template in
src/climate_literature/reporting/templates/wgiii_lit_growth.md.j2 by running
the numbers command above.

## Rebuilding the upstream data

The commands above use model artifacts committed to data version control,
which are produced by:

```
uv run python -m climate_literature.topics.train sweep    # NMF over a grid of topic counts, on a sample
uv run python -m climate_literature.topics.compare        # stability across topic counts
uv run python -m climate_literature.topics.coherence      # coherence scan
uv run python -m climate_literature.topics.train apply    # chosen run -> whole corpus
uv run python -m climate_literature.topics.ipcc_refs      # IPCC reference lists -> cited docs
uv run python -m climate_literature.topics.wg assign      # topic -> WG citation shares
uv run python -m climate_literature.classify.predict      # policy/sector scores -> data/predictions
```

Computationally intensive stages are run on the PIK HPC, with Slurm wrappers in scripts/. 
The choice of 200 topics is documented in report/topic_selection_memo.md, 
with the supporting sheets in report/tables/, and the WG III-relevance rule and its
validation in src/climate_literature/topics/wg.py and scripts/wg3_*.py.

## Dev

Run pre-commit hooks to check code quality with

```sh
uvx prek run --all-files    # ruff, ty
```
