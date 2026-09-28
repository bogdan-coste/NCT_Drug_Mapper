"""
Batch therapeutic-category tester.

Reads Author's Usage + NCT Record ID rows from the Excel file, runs each
through the TherapeuticCategory pipeline, and writes a results CSV.

Usage:
    uv run batch_test.py kanct_drugs_20241219.xlsx [--limit N] [--output results.csv] [--start N]

Arguments:
    file        Path to the xlsx file (required)
    --limit     Max number of rows to process (default: all)
    --start     Row offset to start from, 0-based (default: 0)
    --output    Output CSV path (default: batch_results_<timestamp>.csv)
    --env       Path to .env file (default: .env in cwd)
"""

import argparse
import csv
import sys
import time
import traceback
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Error substrings that indicate an infrastructure failure (not a per-row
# data issue).  When one of these appears in a consecutive run of exceptions,
# the batch aborts instead of burning through the remaining rows.
_INFRA_ERROR_SUBSTRINGS = (
    "no connection could be made",   # WinError 10061 — port refused
    "connection refused",
    "connection reset",
    "remotely closed",
    "errno 111",                      # Linux connection refused
    "qdrant",                         # any explicit Qdrant error
    "winerror 10061",
    "winerror 10054",
)
_INFRA_CONSECUTIVE_ABORT = 3   # abort after this many consecutive infra errors


# ---------------------------------------------------------------------------
# Excel reader (no openpyxl needed)
# ---------------------------------------------------------------------------

def _col_index(ref: str) -> int:
    """Convert a cell reference like 'AB3' to a 0-based column index."""
    letters = "".join(ch for ch in ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def _read_xlsx(path: Path) -> list[dict[str, str]]:
    """Read all rows from the first sheet, returning list of dicts keyed by header."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with zipfile.ZipFile(path) as z:
        shared_raw = z.read("xl/sharedStrings.xml")
        sheet_raw = z.read("xl/worksheets/sheet1.xml")

    shared = ET.fromstring(shared_raw)
    strs: list[str] = [
        "".join(p.text or "" for p in si.findall(".//s:t", ns))
        for si in shared.findall(".//s:si", ns)
    ]

    sheet = ET.fromstring(sheet_raw)
    rows = sheet.findall(".//s:row", ns)

    def cell_value(c: ET.Element) -> str:
        v = c.find("s:v", ns)
        if v is None or v.text is None:
            return ""
        if c.get("t") == "s":
            idx = int(v.text)
            return strs[idx] if idx < len(strs) else ""
        return v.text.strip()

    if not rows:
        return []

    headers = [cell_value(c) for c in rows[0].findall("s:c", ns)]

    records: list[dict[str, str]] = []
    for row in rows[1:]:
        values: dict[int, str] = {
            _col_index(c.get("r", "A")): cell_value(c)
            for c in row.findall("s:c", ns)
        }
        record: dict[str, str] = {
            header: values.get(i, "")
            for i, header in enumerate(headers)
        }
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def _load_pipeline(env_path: Path):
    load_dotenv(env_path)

    from src.schemas.models import AppConfig
    from src.api.sapbert_server import SapBERTServer
    from src.embedder.sapbert_embedder import SapBERTEmbedder
    from src.retrieval.retriever import TherapeuticCategoryRetriever
    from src.vectordb.vector_store import VectorStore
    from src.ingestion.nct_loader import NCTLoader
    from src.llm.llm_client import LLMClient
    from src.processing.modules.thereapeutic_cathegory_module import TherapeuticCategory

    NCT_BASE_URL = "https://clinicaltrials.gov/api/v2"

    config = AppConfig.from_env()
    SapBERTServer(config)

    embedder = SapBERTEmbedder(config.sapbert)
    retriever = TherapeuticCategoryRetriever(embedder, VectorStore(config.qdrant))

    pipeline = TherapeuticCategory(
        nct_loader=NCTLoader(base_url=NCT_BASE_URL),
        definition_llm=LLMClient(config.llm),
        superclass_llm=LLMClient(config.llm),
        mapping_llm=LLMClient(config.llm),
        retriever=retriever,
        term_extractor_llm=LLMClient(config.llm),
    )
    return pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch therapeutic-category test")
    parser.add_argument("file", type=Path, help="Path to the xlsx file")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process")
    parser.add_argument("--start", type=int, default=0, help="Row offset (0-based)")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--env", type=Path, default=Path(".env"), help="Path to .env file")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or Path(f"batch_results_{timestamp}.csv")

    # -- read rows
    print(f"Reading {args.file} ...", flush=True)
    all_rows = _read_xlsx(args.file)
    print(f"  {len(all_rows)} data row(s) found.", flush=True)

    rows = all_rows[args.start:]
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"  Processing {len(rows)} row(s) (start={args.start}, limit={args.limit}).", flush=True)

    if not rows:
        print("Nothing to process.", flush=True)
        sys.exit(0)

    # -- load pipeline
    print("Loading pipeline ...", flush=True)
    try:
        pipeline = _load_pipeline(args.env)
    except Exception as exc:
        print(f"ERROR loading pipeline: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    print("Pipeline ready.\n", flush=True)

    # -- output CSV
    out_fields = [
        "row",
        "term",
        "nct_id",
        "curation_date",
        "status",           # ok | error
        "definition_source",
        "ncit_code",
        "ncit_preferred_name",
        "specific_category",
        "broad_category",
        "root_category",
        "mapped_term",
        "compatibility",
        "match_type",
        "reason",
        "error_message",
        "elapsed_s",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()

        ok_count = 0
        error_count = 0
        skip_count = 0
        consecutive_infra_errors = 0

        with tqdm(total=len(rows), unit="row", desc="Processing", file=sys.stderr) as pbar:
            for row_idx, row in enumerate(rows, start=args.start + 1):
                term = (row.get("Author's Usage") or "").strip()
                nct_raw = (row.get("NCT Record ID") or "").strip()
                curation_date = (row.get("Curation Date") or "").strip()

                pbar.set_postfix(ok=ok_count, err=error_count, skip=skip_count, refresh=False)

                if not term or not nct_raw:
                    tqdm.write(f"  [{row_idx:>4}] SKIP — missing term or NCT ID")
                    writer.writerow({
                        "row": row_idx,
                        "term": term,
                        "nct_id": nct_raw,
                        "curation_date": curation_date,
                        "status": "skip",
                        "error_message": "missing term or NCT ID",
                    })
                    csv_file.flush()
                    skip_count += 1
                    pbar.update(1)
                    continue

                nct_ids = [n.strip() for n in nct_raw.replace(";", ",").split(",") if n.strip()]

                pbar.set_description(f"{term[:30]}")
                tqdm.write(f"  [{row_idx:>4}] {term!r} | {', '.join(nct_ids)}")

                t0 = time.perf_counter()
                try:
                    result = pipeline.run(term=term, nct_ids=nct_ids)
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    exc_str = str(exc)
                    tqdm.write(f"          -> EXCEPTION: {exc_str}")
                    writer.writerow({
                        "row": row_idx,
                        "term": term,
                        "nct_id": nct_raw,
                        "curation_date": curation_date,
                        "status": "error",
                        "error_message": exc_str,
                        "elapsed_s": f"{elapsed:.1f}",
                    })
                    csv_file.flush()
                    error_count += 1
                    pbar.update(1)
                    # Detect infrastructure failures and abort early.
                    exc_lower = exc_str.lower()
                    if any(s in exc_lower for s in _INFRA_ERROR_SUBSTRINGS):
                        consecutive_infra_errors += 1
                        if consecutive_infra_errors >= _INFRA_CONSECUTIVE_ABORT:
                            tqdm.write(
                                f"\nABORTING: {consecutive_infra_errors} consecutive "
                                f"infrastructure errors — Qdrant or embedder is "
                                f"unreachable. Fix the service and resume with "
                                f"--start {row_idx}."
                            )
                            break
                    else:
                        consecutive_infra_errors = 0
                    continue

                elapsed = time.perf_counter() - t0
                consecutive_infra_errors = 0  # successful pipeline call resets counter

                if result.get("error"):
                    tqdm.write(f"          -> ERROR: {result['error']}")
                    writer.writerow({
                        "row": row_idx,
                        "term": term,
                        "nct_id": nct_raw,
                        "curation_date": curation_date,
                        "status": "error",
                        "error_message": result["error"],
                        "elapsed_s": f"{elapsed:.1f}",
                    })
                    csv_file.flush()
                    error_count += 1
                    pbar.update(1)
                    continue

                mapping = result.get("ontology_mapping", {})
                cat = result.get("category_labels", {})
                mapped = mapping.get("selected_term", "—")

                tqdm.write(
                    f"          -> {mapped!r}  "
                    f"[{result.get('definition_source', '?')}]  "
                    f"{elapsed:.1f}s"
                )

                writer.writerow({
                    "row": row_idx,
                    "term": term,
                    "nct_id": nct_raw,
                    "curation_date": curation_date,
                    "status": "ok",
                    "definition_source": result.get("definition_source", ""),
                    "ncit_code": result.get("ncit_code") or "",
                    "ncit_preferred_name": result.get("ncit_preferred_name") or "",
                    "specific_category": cat.get("specific_category", ""),
                    "broad_category": cat.get("broad_category", ""),
                    "root_category": cat.get("root_category", ""),
                    "mapped_term": mapped,
                    "compatibility": mapping.get("compatibility", ""),
                    "match_type": mapping.get("match_type", ""),
                    "reason": mapping.get("reason", ""),
                    "elapsed_s": f"{elapsed:.1f}",
                })
                csv_file.flush()
                ok_count += 1
                pbar.update(1)

    tqdm.write(f"\nDone. Results written to: {output_path}")


if __name__ == "__main__":
    main()
