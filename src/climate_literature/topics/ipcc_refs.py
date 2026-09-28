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

Later steps (same package, wg.py) match these references against the corpus
(item_id via DOI, then title+year fuzzy as fallback) and stream the K200
doc-topic loadings to score each topic by citing WG.

Run from the repo root:

    uv run python -m climate_literature.topics.ipcc_refs fetch
"""

import hashlib
import json
import urllib.request
from datetime import date
from pathlib import Path

import typer

from climate_literature.constants import IPCC_DATA

app = typer.Typer(help="Acquire IPCC reference-list data (DataverseNO).")


@app.callback()
def main() -> None:
    """Keep subcommand syntax stable as more commands (build, match) are added."""


DATASET_DOI = "doi:10.18710/8K2STL"
DATASET_LANDING = f"https://dataverse.no/dataset.xhtml?persistentId={DATASET_DOI}"
DATASET_VERSION = "1.1"  # released 2026-09-03; cite v1.1 explicitly

RAW_DIR = IPCC_DATA / "raw"
MANIFEST = IPCC_DATA / "fetch_manifest.json"

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


if __name__ == "__main__":
    app()
