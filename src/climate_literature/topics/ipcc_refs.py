"""IPCC citation provenance for the K200 corpus — the 2020 WG-assignment redo.

The 2020 paper assigned every topic a `primary_wg` by aggregating which
working-group reports cited its documents (Django IPCCRef rows tagged AR×WG in
the 2019 project). Our 2026 corpus is a bulk Scopus download with no IPCC
citation provenance at all, so the citation side has to be rebuilt. Instead of
re-parsing report PDFs, we use Karpova (2026), a CC0 database of 188,576
references parsed from 26 IPCC reports (AR2–AR6 assessment reports plus most
special reports), each reference tagged with its WG (I/II/III/SR), report,
chapter, year and — where available — DOI. The AR6 slice is 57k unique
references with 78% DOI coverage.

    Karpova, A. (2026). Database of references from Intergovernmental Panel on
    Climate Change (IPCC) reports in the years 1995-2022 [Data set].
    DataverseNO. https://doi.org/10.18710/8K2STL

This module:

  fetch — download the dataset files, verify pinned sha256 checksums, write a
    provenance manifest (data/ipcc/raw/ is gitignored; the manifest in
    data/ipcc/ is git-tracked).

  build — normalize the AR6 slice into data/ipcc/ar6_refs.parquet: one row per
    unique reference (id_unif), DOI repaired (the parser left whitespace
    breaks in some), per-WG citing-chapter counts, first author/year/journal.

  match — join references to corpus documents. Corpus side = data/predictions
    (item_id, scopus_id, title, publication_year) plus DOIs from the raw
    Scopus JSONL (cached in data/ipcc/raw/scopus_doi.parquet). Tiers: exact
    DOI; exact normalized title with year ±1; rapidfuzz token_set_ratio ≥ 90
    with year ±2 over a rare-token-blocked candidate pool. Outputs to
    report/tables: ipcc_ar6_matches.parquet (ref → item_id, auditable),
    ipcc_doc_wg.parquet (per-document WG citation counts, the input to the
    topic scoring in wg.py), ipcc_ar6_spotcheck.csv (sample for eyeballing).

Downstream (same package, wg.py): score each K200 topic by the WG mix of the
literature it contains, assign primary_wg, and count documents per WG.

Run from the repo root:

    uv run python -m climate_literature.topics.ipcc_refs fetch
    uv run python -m climate_literature.topics.ipcc_refs build
    uv run python -m climate_literature.topics.ipcc_refs match
"""

import hashlib
import json
import random
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
import typer
from rapidfuzz import process
from rapidfuzz.fuzz import token_set_ratio

from climate_literature.constants import IPCC_DATA, PREDICTIONS_DATA, TABLES_DIR

app = typer.Typer(help="Acquire IPCC reference-list data (DataverseNO).")


@app.callback()
def main() -> None:
    """Keep subcommand syntax stable as more commands (build, match) are added."""


DATASET_DOI = "doi:10.18710/8K2STL"
DATASET_LANDING = f"https://dataverse.no/dataset.xhtml?persistentId={DATASET_DOI}"
DATASET_VERSION = "1.1"  # released 2026-09-03; cite v1.1 explicitly

RAW_DIR = IPCC_DATA / "raw"
MANIFEST = IPCC_DATA / "fetch_manifest.json"
AR6_REFS = IPCC_DATA / "ar6_refs.parquet"
DOI_MAP = RAW_DIR / "scopus_doi.parquet"
CORPUS_INDEX = RAW_DIR / "corpus_index.parquet"
# binary MB-scale artifacts live in data/ipcc/ (DVC-tracked); the human-
# reviewable CSVs stay git-tracked under report/tables
MATCHES = IPCC_DATA / "ipcc_ar6_matches.parquet"
DOC_WG = IPCC_DATA / "ipcc_doc_wg.parquet"
SPOTCHECK = TABLES_DIR / "ipcc_ar6_spotcheck.csv"

# Dataverse datafile ids from the v1.1 file listing; access URLs redirect to
# the storage S3 objects, so pin the checksum instead of the URL contents.
FILES = {
    "IPCC_database.csv": {
        "file_id": 295940,
        "sha256": "a677be90b6529fbd30a70e5fce6b9525cd14b8f6437603354239d5208b43244e",
        "bytes": 75036588,
    },
    "ReadMe.txt": {
        "file_id": 295948,
        "sha256": "683465e471b0cba23308e87bbfca2ef5769aba990dddfe5c5c8025110ee3793f",
        "bytes": 4910,
    },
    "Summary_over_included_reports.pdf": {
        "file_id": 295943,
        "sha256": "15639646af658f35e23610eb70b45e9790a3836ec810dce6206632803fc5aa1b",
        "bytes": 282374,
    },
}

AR6_PREFIXES = ("AR6_WGI", "AR6_WGII", "AR6_WGIII")
WG_CODE = {"I": "wg1", "II": "wg2", "III": "wg3"}

DOI_RE = re.compile(r"^10\.[^/\s]+/[^\s]+$")
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


# --- normalization ---------------------------------------------------------


def _clean_doi(value) -> str | None:
    """Dataverse DOI cells contain whitespace-mangled DOIs and stray URLs."""
    if not isinstance(value, str):
        return None
    s = re.sub(r"\s+", "", value).lower().rstrip(".,;")
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    return s if DOI_RE.match(s) else None


def _norm_title(value) -> str | None:
    """Lowercase, de-accented, alphanumeric-only title key for joining."""
    if not isinstance(value, str):
        return None
    s = unicodedata.normalize("NFKD", value)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _year(value) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    m = YEAR_RE.search(str(value))
    return int(m.group(0)) if m else None


# --- fetch -----------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "climate-literature-ipcc"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            out.write(chunk)
    tmp.rename(dest)


@app.command()
def fetch(
    force: bool = typer.Option(False, help="Re-download even if checksums match."),
) -> None:
    """Download the pinned DataverseNO files into data/ipcc/raw/, checksum-verified."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, spec in FILES.items():
        dest = RAW_DIR / name
        url = f"https://dataverse.no/api/access/datafile/{spec['file_id']}"
        if dest.exists() and not force and _sha256(dest) == spec["sha256"]:
            typer.echo(f"cached  {name}")
        else:
            typer.echo(f"fetch   {name} <- {url}")
            _download(url, dest)
            digest = _sha256(dest)
            if digest != spec["sha256"]:
                typer.secho(
                    f"checksum mismatch for {name}: got {digest}, pinned "
                    f"{spec['sha256']} — dataset may have changed version",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=1)
        entries[name] = {
            "dataverse_file_id": spec["file_id"],
            "sha256": spec["sha256"],
            "bytes": dest.stat().st_size,
        }
    manifest = {
        "dataset": DATASET_LANDING,
        "doi": DATASET_DOI,
        "version": DATASET_VERSION,
        "license": "CC0 1.0",
        "citation": (
            "Karpova, A., 2026, 'Database of references from Intergovernmental "
            "Panel on Climate Change (IPCC) reports in the years 1995-2022', "
            "DataverseNO, V1.1, CC0-1.0, doi:10.18710/8K2STL"
        ),
        "retrieved": date.today().isoformat(),
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    typer.echo(f"manifest -> {MANIFEST}")


# --- build: normalized AR6 reference table ---------------------------------


def _load_raw() -> pd.DataFrame:
    raw = pd.read_csv(
        RAW_DIR / "IPCC_database.csv", sep=";", dtype=str, encoding="utf-8-sig"
    )
    raw = raw[raw["Report"].notna()].rename(
        columns={"Journal _name_stand": "journal_std"}
    )
    return raw


def load_ar6_refs(rebuild: bool = False) -> pd.DataFrame:
    """One row per unique AR6 reference (id_unif), normalized."""
    if AR6_REFS.exists() and not rebuild:
        return pd.read_parquet(AR6_REFS)
    raw = _load_raw()
    ar6 = raw[raw["Report"].str.startswith(AR6_PREFIXES)].copy()
    ar6["doi_clean"] = ar6["DOI"].map(_clean_doi)
    ar6["pub_year"] = ar6["Year"].map(_year)
    ar6["title_norm"] = ar6["Title"].map(_norm_title)
    for code, col in WG_CODE.items():
        ar6[col] = (ar6["WG"] == code).astype("int64")
    ar6["chapter"] = ar6["Report"] + " ~ " + ar6["Chapter_name"].astype(str)

    def _first_valid(s: pd.Series):
        s = s.dropna()
        return s.iloc[0] if len(s) else pd.NA

    refs = (
        ar6.sort_values(["id_unif", "Report", "Chapter_name"])
        .groupby("id_unif", sort=True)
        .agg(
            title=("Title", "first"),
            title_norm=("title_norm", "first"),
            journal=("journal_std", _first_valid),
            first_author=("First_author", "first"),
            pub_year=("pub_year", _first_valid),
            doi=("doi_clean", _first_valid),
            wg1_n=("wg1", "sum"),
            wg2_n=("wg2", "sum"),
            wg3_n=("wg3", "sum"),
            n_chapters=("chapter", "nunique"),
            wgs=("WG", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index()
    )
    IPCC_DATA.mkdir(parents=True, exist_ok=True)
    refs.to_parquet(AR6_REFS, index=False)
    return refs


@app.command()
def build() -> None:
    """Write data/ipcc/ar6_refs.parquet: normalized, deduplicated AR6 references."""
    refs = load_ar6_refs(rebuild=True)
    typer.echo(f"{len(refs)} unique AR6 references -> {AR6_REFS}")
    typer.echo(f"  with clean DOI: {refs['doi'].notna().mean():.1%}")
    typer.echo(
        "  cited by: "
        + ", ".join(
            f"{k} {v} ({v / len(refs):.1%})"
            for k, v in refs["wgs"].value_counts().items()
        )
    )


# --- corpus index ----------------------------------------------------------


def _build_doi_map() -> pd.DataFrame:
    """scopus_id -> doi from the raw Scopus API JSONL (one scan, cached)."""
    if DOI_MAP.exists():
        return pd.read_parquet(DOI_MAP)
    rows = []
    from climate_literature.constants import RAW_DATA

    for f in sorted(RAW_DATA.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                rec = json.loads(line)
                doi = _clean_doi(rec.get("prism:doi"))
                if doi:
                    rows.append((rec.get("eid") or rec.get("dc:identifier", ""), doi))
    df = pd.DataFrame(rows, columns=["scopus_id", "doi"]).drop_duplicates("scopus_id")
    df.to_parquet(DOI_MAP, index=False)
    return df


def load_corpus_index(refresh: bool = False) -> pd.DataFrame:
    """item_id, scopus_id, doi, title_norm, pub_year for all corpus documents."""
    if CORPUS_INDEX.exists() and not refresh:
        return pd.read_parquet(CORPUS_INDEX)
    parts = [
        pd.read_parquet(
            p, columns=["item_id", "scopus_id", "title", "publication_year"]
        )
        for p in sorted(PREDICTIONS_DATA.glob("*/*.parquet"))
    ]
    docs = pd.concat(parts).drop_duplicates("item_id")
    docs["doi"] = docs["scopus_id"].map(
        _build_doi_map().drop_duplicates("scopus_id").set_index("scopus_id")["doi"]
    )
    docs["title_norm"] = docs["title"].map(_norm_title)
    docs["pub_year"] = docs["publication_year"].map(_year)
    docs = docs.drop(columns=["title", "publication_year"])
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(CORPUS_INDEX, index=False)
    return docs


# --- match -----------------------------------------------------------------


def _year_ok(ref_year, doc_year, tol: int) -> bool:
    if ref_year is None or pd.isna(ref_year):
        return True
    if doc_year is None or pd.isna(doc_year):
        return True
    return abs(int(ref_year) - int(doc_year)) <= tol


def _match(refs: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame:
    """Assign each corpus document to at most one reference, tier by tier.

    Returns one row per (reference, corpus document) assignment. A reference
    may match several documents when the corpus holds duplicate Scopus
    records; corpus documents are assigned once, DOI tier first.
    """
    refs = refs.sort_values("id_unif").reset_index(drop=True)
    # plain lists, not frames: per-row .iloc on 1.4M-row columns is the hot
    # path here and pandas scalar access dominates the runtime otherwise
    n_ci = len(ci)
    titles = ci["title_norm"].tolist()
    years = ci["pub_year"].tolist()
    item_ids = ci["item_id"].tolist()
    r_titles = refs["title_norm"].tolist()
    r_years = refs["pub_year"].tolist()
    # ref fields carry the `ipcc_` prefix in the output so they never collide
    # with the corpus columns (`corpus_…`) on the same row
    r_fields = {
        "id_unif": refs["id_unif"].tolist(),
        "ipcc_title": refs["title"].tolist(),
        "ipcc_year": refs["pub_year"].tolist(),
        "ipcc_doi": refs["doi"].tolist(),
        "first_author": refs["first_author"].tolist(),
        "wgs": refs["wgs"].tolist(),
        "wg1_n": refs["wg1_n"].tolist(),
        "wg2_n": refs["wg2_n"].tolist(),
        "wg3_n": refs["wg3_n"].tolist(),
        "n_chapters": refs["n_chapters"].tolist(),
    }
    assigned = [-1] * n_ci  # ci row -> ref position
    matched_refs: set[int] = set()
    out_rows: list[dict] = []

    def _take(ref_pos: int, rows, tier: str, score: float) -> None:
        """Assign the still-free candidate documents to reference `ref_pos`."""
        free = [r for r in rows if assigned[r] == -1]
        if not free:
            return
        for r in free:
            assigned[r] = ref_pos
        matched_refs.add(ref_pos)
        row = {c: vals[ref_pos] for c, vals in r_fields.items()}
        row["item_ids"] = [item_ids[r] for r in free]
        row["corpus_years"] = [years[r] for r in free]
        row["tier"] = tier
        row["score"] = score
        out_rows.append(row)

    # tier 1: exact DOI
    doi_rows: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(ci["doi"].tolist()):
        if isinstance(d, str):
            doi_rows[d].append(i)
    for pos, d in enumerate(refs["doi"].tolist()):
        if isinstance(d, str) and d in doi_rows:
            _take(pos, doi_rows[d], "doi", 100.0)
    typer.echo(f"  doi tier: {len(matched_refs)} refs")

    # tier 2: exact normalized title, year ±1
    title_rows: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(titles):
        if isinstance(t, str):
            title_rows[t].append(i)
    for pos, t in enumerate(r_titles):
        if pos in matched_refs or not isinstance(t, str) or t not in title_rows:
            continue
        cand = [
            r
            for r in title_rows[t]
            if assigned[r] == -1 and _year_ok(r_years[pos], years[r], 1)
        ]
        if cand:
            _take(pos, cand, "title", 95.0)
    typer.echo(f"  title tier: {len(matched_refs)} refs total now")

    # tier 3: rapidfuzz over a rare-token-blocked candidate pool
    dfreq: Counter = Counter(
        tok for t in titles if isinstance(t, str) for tok in set(t.split(" "))
    )
    postings: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(titles):
        if not isinstance(t, str):
            continue
        for tok in set(t.split(" ")):
            if len(tok) >= 3 and dfreq[tok] <= 1500:
                postings[tok].append(i)
    n_before = len(matched_refs)
    for pos, qt in enumerate(r_titles):
        if pos in matched_refs or not isinstance(qt, str):
            continue
        qtoks = [t for t in qt.split(" ") if len(t) >= 3]
        # token_set_ratio gives 100 to any title CONTAINING a short query
        # ("Dengue" ⊂ "early warning system for dengue outbreak …"), and AR6
        # grey-lit entries love one-word titles. Require several query tokens
        # and full query-token coverage in the candidate title; blocking then
        # uses only tokens that are rare enough to be selective.
        if len(qtoks) < 3:
            continue
        blocking = [t for t in qtoks if t in postings]
        if not blocking:
            continue
        blocking.sort(key=lambda t: dfreq[t])
        cand = set()
        for t in blocking[:3]:
            cand.update(postings[t])
        cand = {
            r for r in cand if assigned[r] == -1 and _year_ok(r_years[pos], years[r], 2)
        }
        if not cand:
            continue
        cand_list = sorted(cand)
        qset = set(qtoks)
        scores = process.extract(
            qt,
            [titles[r] for r in cand_list],
            scorer=token_set_ratio,
            score_cutoff=90.0,
            limit=None,
        )
        ok = [
            (choice, s, i)
            for choice, s, i in scores
            if qset.issubset(set(choice.split(" ")))
        ]
        if not ok:
            continue
        best = max(s for _, s, _ in ok)
        winners = sorted(i for _, s, i in ok if s == best)
        _take(pos, [cand_list[i] for i in winners], "fuzzy", float(best))
        if pos % 10_000 == 0:
            typer.echo(f"  fuzzy tier: ref {pos} ({len(matched_refs)} matched)")
    typer.echo(f"  fuzzy tier done: +{len(matched_refs) - n_before} refs")
    return pd.DataFrame(out_rows)


@app.command()
def match(
    refresh_index: bool = typer.Option(
        False, help="Rebuild the cached corpus DOI/title index."
    ),
) -> None:
    """Match AR6 references to corpus documents; write tables + a spot-check dump."""
    refs = load_ar6_refs()
    ci = load_corpus_index(refresh=refresh_index)
    typer.echo(
        f"matching {len(refs)} references against {len(ci)} corpus documents"
        f" (corpus DOI coverage {ci['doi'].notna().mean():.1%})"
    )
    matches = _match(refs, ci)
    matches = matches.rename(
        columns={"item_ids": "item_id", "corpus_years": "corpus_year"}
    ).explode(["item_id", "corpus_year"], ignore_index=True)
    matches["corpus_title"] = matches["item_id"].map(
        ci.set_index("item_id")["title_norm"]
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(MATCHES, index=False)
    per_item = matches.groupby("item_id")[["wg1_n", "wg2_n", "wg3_n"]].sum()
    per_item["n_refs"] = matches.groupby("item_id")["id_unif"].nunique()
    per_item = per_item[["n_refs", "wg1_n", "wg2_n", "wg3_n"]].reset_index()
    per_item.to_parquet(DOC_WG, index=False)

    matched = matches["id_unif"].nunique()
    typer.echo(f"refs matched: {matched} / {len(refs)} ({matched / len(refs):.1%})")
    tiers = matches["tier"].value_counts()
    typer.echo(
        "  by tier: " + ", ".join(f"{t} {tiers[t]}" for t in ("doi", "title", "fuzzy"))
    )
    for wg, col in (("I", "wg1_n"), ("II", "wg2_n"), ("III", "wg3_n")):
        sub = refs[[wg in s.split(",") for s in refs["wgs"]]]
        hit = sub["id_unif"].isin(set(matches.loc[matches[col] > 0, "id_unif"]))
        typer.echo(
            f"  WG{wg}: {hit.sum()} / {len(sub)} refs matched ({hit.mean():.1%})"
        )
    typer.echo(f"corpus documents with ≥1 AR6 citation: {len(per_item)}")

    rng = random.Random(1)
    sample = []
    for _, grp in matches.groupby("tier"):
        idx = rng.sample(range(len(grp)), min(8, len(grp)))
        sample.extend(grp.iloc[idx].to_dict("records"))
    spot = pd.DataFrame(sample)[
        ["tier", "score", "ipcc_year", "ipcc_title", "corpus_year", "corpus_title"]
    ]
    spot.to_csv(SPOTCHECK, index=False)
    typer.echo(f"spot-check sample ({len(spot)} rows) -> {SPOTCHECK}")
    typer.echo(f"match table -> {MATCHES}; per-document WG counts -> {DOC_WG}")


if __name__ == "__main__":
    app()
