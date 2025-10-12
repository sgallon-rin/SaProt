#!/usr/bin/env python
"""Prepare FOLD classification dataset for SaProt fine-tuning.

This script reads the SCOP split files (e.g. training.txt, validation.txt,
 test_fold.txt, test_family.txt, test_superfamily.txt), converts each listed
 domain PDB structure into SaProt AA+3Di tokens with Foldseek, assigns
 integer labels, and writes the dataset into LMDB shards expected by the
 SaProt finetuning pipeline.

$ python scripts/build_fold_lmdb.py   --foldseek bin/foldseek   --pdb-dir /raid_elmo/home/lr/shenjl/wangyi/protein-pocket/data_fixed/HomologyTAPE   --splits-dir /raid_elmo/home/lr/wangyi/Protein/HomologyTAPE   --label-column 1   --out-root LMDB/FOLD   --overwrite
Collected 1195 unique labels.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from utils.foldseek_util import get_struc_seq
from utils.generate_lmdb import jsonl2lmdb


def parse_split_file(path: Path, label_column: int) -> List[Tuple[str, str]]:
    """Parse a SCOP split file and return (domain_id, label_name) pairs."""
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if label_column >= len(fields):
                raise ValueError(
                    f"Line '{line}' in {path} has only {len(fields)} columns; "
                    f"label_column={label_column} is out of range."
                )

            domain_id = fields[0]
            label_name = fields[label_column]
            pairs.append((domain_id, label_name))

    return pairs


def collect_labels(split_to_pairs: Dict[str, List[Tuple[str, str]]]) -> Dict[str, int]:
    """Build a deterministic mapping from label string to numeric id."""
    labels = sorted({label for pairs in split_to_pairs.values() for _, label in pairs})
    return {label: idx for idx, label in enumerate(labels)}


def inspect_label_format(labels: Iterable[str]) -> None:
    """Print whether labels follow the pattern <letters>.<digits> (e.g., a.1)."""
    pattern = re.compile(r"^[A-Za-z]+\.[0-9]+$")
    mismatched = [label for label in labels if not pattern.match(label)]

    if mismatched:
        preview = ", ".join(mismatched[:5])
        print(
            f"[WARN] {len(mismatched)} labels do not match <letters>.<digits>. "
            f"Examples: {preview}"
        )
    else:
        print("[OK] All labels match the expected pattern <letters>.<digits> (e.g., a.1).")


def domain_to_chain_hint(domain_id: str) -> str | None:
    """Infer the chain identifier from a SCOP domain id if possible."""
    # SCOP domain ids typically look like d1r4ca_. The character after the
    # pdb code often encodes the chain. We use that heuristic when multiple
    # chains are returned from Foldseek.
    if len(domain_id) >= 7 and domain_id[0] == "d":
        chain_candidate = domain_id[5]
        if chain_candidate.isalpha():
            return chain_candidate.upper()
    return None


def extract_combined_sequence(
    foldseek_bin: Path,
    pdb_path: Path,
    chain_hint: str | None,
    foldseek_verbose: bool = False,
) -> str:
    """Run Foldseek on a PDB domain and return the AA+3Di combined sequence."""
    seq_dict = get_struc_seq(
        foldseek=str(foldseek_bin),
        path=str(pdb_path),
        chains=None,
        process_id=0,
        plddt_mask=False,
        foldseek_verbose=foldseek_verbose,
    )

    if not seq_dict:
        raise RuntimeError(f"Foldseek returned no sequences for {pdb_path}")

    if chain_hint and chain_hint in seq_dict:
        combined_seq = seq_dict[chain_hint][2]
    else:
        # Fall back to the first available chain.
        combined_seq = next(iter(seq_dict.values()))[2]

    return combined_seq


def build_split_jsonl(
    foldseek_bin: Path,
    pdb_dir: Path,
    pairs: Iterable[Tuple[str, str]],
    label_to_idx: Dict[str, int],
    jsonl_path: Path,
    foldseek_verbose: bool = False,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as writer:
        for domain_id, label_name in tqdm(list(pairs), desc=f"{jsonl_path.stem}"):
            pdb_path = pdb_dir / f"{domain_id}.pdb"
            if not pdb_path.exists():
                raise FileNotFoundError(f"Missing structure file: {pdb_path}")

            chain_hint = domain_to_chain_hint(domain_id)
            combined_seq = extract_combined_sequence(
                foldseek_bin=foldseek_bin,
                pdb_path=pdb_path,
                chain_hint=chain_hint,
                foldseek_verbose=foldseek_verbose,
            )

            sample = {
                "uid": domain_id,
                "seq": combined_seq,
                "label": label_to_idx[label_name],
            }
            writer.write(json.dumps(sample, ensure_ascii=False) + "\n")


def jsonl_to_lmdb(jsonl_path: Path, lmdb_dir: Path, overwrite: bool) -> None:
    if lmdb_dir.exists() and overwrite:
        shutil.rmtree(lmdb_dir)
    jsonl2lmdb(str(jsonl_path), str(lmdb_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FOLD LMDB dataset for SaProt.")
    parser.add_argument("--foldseek", required=True, type=Path, help="Path to foldseek binary")
    parser.add_argument("--pdb-dir", required=True, type=Path, help="Directory with domain PDB files")
    parser.add_argument("--splits-dir", required=True, type=Path, help="Directory with SCOP split txt files")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=[
            "training.txt",
            "validation.txt",
            "test_fold.txt",
            "test_superfamily.txt",
            "test_family.txt",
        ],
        help="Split files to process (relative to splits-dir)",
    )
    parser.add_argument(
        "--label-column",
        type=int,
        default=1,
        help="Zero-based index of the column to use as the label (default: 1)",
    )
    parser.add_argument("--out-root", required=True, type=Path, help="Output directory root for LMDB shards")
    parser.add_argument("--keep-jsonl", action="store_true", help="Keep intermediate jsonl files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing LMDB directories")
    parser.add_argument("--foldseek-verbose", action="store_true", help="Print Foldseek output")
    args = parser.parse_args()

    per_split_label_col = {
        "training.txt": 1,
        "validation.txt": 1,
        "test_family.txt": 1,
        "test_fold.txt": 2,
        "test_superfamily.txt": 2,
    }

    split_pairs: Dict[str, List[Tuple[str, str]]] = {}
    for split_name in args.splits:
        split_path = args.splits_dir / split_name
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        label_col = per_split_label_col.get(split_name, args.label_column)
        split_pairs[split_name] = parse_split_file(split_path, label_col)

    label_to_idx = collect_labels(split_pairs)
    print(f"Collected {len(label_to_idx)} unique labels.")
    inspect_label_format(label_to_idx.keys())

    for split_name, pairs in split_pairs.items():
        jsonl_path = args.out_root / "jsonl" / f"{split_name}.jsonl"
        build_split_jsonl(
            foldseek_bin=args.foldseek,
            pdb_dir=args.pdb_dir,
            pairs=pairs,
            label_to_idx=label_to_idx,
            jsonl_path=jsonl_path,
            foldseek_verbose=args.foldseek_verbose,
        )

        lmdb_dir = args.out_root / "foldseek" / split_name.replace(".txt", "")
        jsonl_to_lmdb(jsonl_path, lmdb_dir, overwrite=args.overwrite)

        if not args.keep_jsonl:
            jsonl_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
