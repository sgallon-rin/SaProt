#!/usr/bin/env python
"""Prepare Reaction classification dataset for SaProt fine-tuning.

The Reaction dataset provides split lists (training/validation/test) where
each line contains a PDB id and chain in the form ``4xi6.A``. The label for
each chain is stored separately in ``chain_functions.txt`` with lines like
``4fae.B,247``. This script converts those resources into LMDB shards that
the SaProt classification datamodule can consume.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from utils.foldseek_util import get_struc_seq  # noqa: E402
from utils.generate_lmdb import jsonl2lmdb  # noqa: E402


def load_chain_labels(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "," in line:
                chain_id, label = [part.strip() for part in line.split(",", 1)]
            else:
                parts = [part.strip() for part in line.split() if part.strip()]
                if len(parts) != 2:
                    raise ValueError(f"Cannot parse line '{line}' in {path}")
                chain_id, label = parts

            mapping[chain_id] = label

    return mapping


def parse_split_file(path: Path) -> List[str]:
    chains: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            chains.append(line)

    return chains


def collect_labels(mapping: Dict[str, str], chains: Iterable[str]) -> Dict[str, int]:
    labels = sorted({mapping[chain] for chain in chains})
    return {label: idx for idx, label in enumerate(labels)}


def locate_structure_file(pdb_dir: Path, chain_id: str, extensions: List[str]) -> Path:
    base_id, _, chain = chain_id.partition(".")
    candidates = []

    for ext in extensions:
        candidates.append(pdb_dir / f"{chain_id}{ext}")
        candidates.append(pdb_dir / f"{chain_id.replace('.', '_')}{ext}")
        candidates.append(pdb_dir / f"{base_id}{ext}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find structure file for {chain_id}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def extract_combined_sequence(
    foldseek_bin: Path,
    pdb_path: Path,
    chain: str,
    process_id: int,
    foldseek_verbose: bool,
) -> str:
    target_chain = chain.upper() if chain else "A"

    seq_dict = get_struc_seq(
        foldseek=str(foldseek_bin),
        path=str(pdb_path),
        chains=[target_chain],
        process_id=process_id,
        plddt_mask=False,
        foldseek_verbose=foldseek_verbose,
    )

    if target_chain not in seq_dict:
        # Retry without restricting to a specific chain so we can inspect all outputs.
        seq_dict = get_struc_seq(
            foldseek=str(foldseek_bin),
            path=str(pdb_path),
            chains=None,
            process_id=process_id,
            plddt_mask=False,
            foldseek_verbose=foldseek_verbose,
        )

    if not seq_dict:
        raise KeyError(
            f"Foldseek returned no chains for {pdb_path}."
        )

    if target_chain in seq_dict:
        return seq_dict[target_chain][2]

    # Fall back to the first chain and warn so users can review later.
    fallback_chain, value = next(iter(seq_dict.items()))
    print(
        f"[WARN] Chain '{target_chain}' not found in {pdb_path.name}, "
        f"using chain '{fallback_chain}' instead."
    )
    return value[2]


def build_split_jsonl(
    foldseek_bin: Path,
    pdb_dir: Path,
    chains: Iterable[str],
    chain_to_label: Dict[str, str],
    label_to_idx: Dict[str, int],
    jsonl_path: Path,
    extensions: List[str],
    foldseek_verbose: bool,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as writer:
        for chain_id in tqdm(list(chains), desc=f"{jsonl_path.stem}"):
            if chain_id not in chain_to_label:
                raise KeyError(f"Label not found for chain {chain_id} in chain_functions file")

            pdb_path = locate_structure_file(pdb_dir, chain_id, extensions)
            base_id, _, chain = chain_id.partition(".")
            combined_seq = extract_combined_sequence(
                foldseek_bin=foldseek_bin,
                pdb_path=pdb_path,
                chain=chain or "A",
                process_id=os.getpid(),
                foldseek_verbose=foldseek_verbose,
            )

            sample = {
                "uid": chain_id,
                "seq": combined_seq,
                "label": label_to_idx[chain_to_label[chain_id]],
            }

            writer.write(json.dumps(sample, ensure_ascii=False) + "\n")


def jsonl_to_lmdb(jsonl_path: Path, lmdb_dir: Path, overwrite: bool) -> None:
    if lmdb_dir.exists() and overwrite:
        shutil.rmtree(lmdb_dir)
    jsonl2lmdb(str(jsonl_path), str(lmdb_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Reaction LMDB dataset for SaProt.")
    parser.add_argument("--foldseek", required=True, type=Path, help="Path to foldseek binary")
    parser.add_argument("--pdb-dir", required=True, type=Path, help="Directory with PDB/mmCIF files")
    parser.add_argument("--splits-dir", required=True, type=Path, help="Directory containing split txt files")
    parser.add_argument("--train", default="training.txt", help="Training split filename")
    parser.add_argument("--valid", default="validation.txt", help="Validation split filename")
    parser.add_argument("--test", default="test.txt", help="Test split filename")
    parser.add_argument("--label-file", required=True, type=Path, help="chain_functions.txt path")
    parser.add_argument("--out-root", required=True, type=Path, help="Output directory root for LMDB shards")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".pdb", ".cif", ".ent"],
        help="Structure file extensions to try when locating PDB files",
    )
    parser.add_argument("--keep-jsonl", action="store_true", help="Keep intermediate jsonl files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing LMDB directories")
    parser.add_argument("--foldseek-verbose", action="store_true", help="Print Foldseek output")
    args = parser.parse_args()

    chain_to_label = load_chain_labels(args.label_file)

    split_paths = {
        "train": args.splits_dir / args.train,
        "valid": args.splits_dir / args.valid,
        "test": args.splits_dir / args.test,
    }

    for name, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")

    train_chains = parse_split_file(split_paths["train"])
    valid_chains = parse_split_file(split_paths["valid"])
    test_chains = parse_split_file(split_paths["test"])

    all_chains = train_chains + valid_chains + test_chains
    missing_labels = [chain for chain in all_chains if chain not in chain_to_label]
    if missing_labels:
        raise KeyError(
            "Labels missing for the following chains: " + ", ".join(missing_labels[:10])
            + (" ..." if len(missing_labels) > 10 else "")
        )

    label_to_idx = collect_labels(chain_to_label, all_chains)
    print(f"Collected {len(label_to_idx)} unique labels.")

    splits = {
        "train": train_chains,
        "valid": valid_chains,
        "test": test_chains,
    }

    for split_name, chains in splits.items():
        jsonl_path = args.out_root / "jsonl" / f"{split_name}.jsonl"
        build_split_jsonl(
            foldseek_bin=args.foldseek,
            pdb_dir=args.pdb_dir,
            chains=chains,
            chain_to_label=chain_to_label,
            label_to_idx=label_to_idx,
            jsonl_path=jsonl_path,
            extensions=args.extensions,
            foldseek_verbose=args.foldseek_verbose,
        )

        lmdb_dir = args.out_root / "foldseek" / split_name
        jsonl_to_lmdb(jsonl_path, lmdb_dir, overwrite=args.overwrite)

        if not args.keep_jsonl:
            jsonl_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
