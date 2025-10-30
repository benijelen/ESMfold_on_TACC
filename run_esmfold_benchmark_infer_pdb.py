#!/usr/bin/env python3
import argparse
import os
import re
import time
import resource
from pathlib import Path

from transformers import EsmForProteinFolding, AutoTokenizer
import torch

def parse_fasta(path: Path):
    records = []
    header = None
    seq_lines = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(seq_lines).replace(" ", "").replace("\t", "")
                    records.append((header, seq))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
    if header is not None:
        seq = "".join(seq_lines).replace(" ", "").replace("\t", "")
        records.append((header, seq))
    return records

def safe_slug(s: str, maxlen: int = 180) -> str:
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s[:maxlen].strip("_")

def human_time(seconds: float) -> str:
    if seconds >= 60:
        m = int(seconds // 60)
        s = int(round(seconds - 60 * m))
        return f"{m}min {s}s"
    return f"{seconds:.2f} s"

def main():
    ap = argparse.ArgumentParser(description="Benchmark ESMFold infer_pdb over a FASTA file.")
    ap.add_argument("--fasta", required=True, help="Path to FASTA with sequences")
    ap.add_argument("--outdir", default="./esmfold_bench_out", help="Directory to write PDBs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="cuda or cpu")
    args = ap.parse_args()

    fasta_path = Path(args.fasta)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load model/tokenizer once
    device = torch.device(args.device)
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval().to(device)
    _ = AutoTokenizer.from_pretrained("facebook/esmfold_v1")  # not used here, but ok to keep

    torch.set_grad_enabled(False)

    records = parse_fasta(fasta_path)
    if not records:
        raise SystemExit(f"No sequences found in FASTA: {fasta_path}")

    for header, seq in records:
        seq = seq.strip().upper().replace(" ", "")
        gene_name = header.split(" | ", 1)[0]  # keep the original header left of " | "
        print(f">{gene_name}")

        # --- timing start ---
        ru_start = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.time()

        pdb_str = model.infer_pdb(seq)

        wall = time.time() - t0
        ru_end = resource.getrusage(resource.RUSAGE_SELF)
        user_cpu = ru_end.ru_utime - ru_start.ru_utime
        sys_cpu = ru_end.ru_stime - ru_start.ru_stime
        total_cpu = user_cpu + sys_cpu
        # --- timing end ---

        # Write PDB (outside the timed block to keep timing true to inference)
        fname = safe_slug(gene_name) + ".pdb"
        with open(outdir / fname, "w") as f:
            f.write(pdb_str)

        # Print summary (Colab-style)
        print(f"length {len(seq)}")
        print(f"CPU times: user {human_time(user_cpu)}, sys: {human_time(sys_cpu)}, total: {human_time(total_cpu)}")
        print(f"Wall time: {human_time(wall)}")
        print()  # blank line between entries

if __name__ == "__main__":
    main()
