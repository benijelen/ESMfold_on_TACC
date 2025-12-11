#!/usr/bin/env python3

import argparse
import os
import re
import time
import torch
from Bio import SeqIO
from transformers import EsmForProteinFolding, AutoTokenizer

# Hard cutoff to avoid GPU OOM on giant ORFs
MAX_LEN = 1024

def safe_name(s, maxlen=200):
    """
    Sanitize sequence IDs for filesystem safety.
    Keeps alnum, dot, dash, underscore.
    Truncates long IDs.
    """
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s[:maxlen]

def fold_fasta(fasta_path, outdir):
    """
    Iterate through sequences in a FASTA, fold each (unless too long),
    and write PDB files. Skips already-folded IDs.
    """

    os.makedirs(outdir, exist_ok=True)

    print("Loading ESMFold model ...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    # Put model on GPU if available, and turn off training behavior
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    for rec in SeqIO.parse(fasta_path, "fasta"):
        raw_id = rec.id
        seq = str(rec.seq)
        seqlen = len(seq)

        seq_id = safe_name(raw_id)
        pdb_path = os.path.join(outdir, f"{seq_id}.pdb")

        # 1. Skip work we've already done
        if os.path.exists(pdb_path):
            print(f"✔ Already exists, skipping: {seq_id}")
            continue

        # 2. Skip monsters that will OOM
        if seqlen > MAX_LEN:
            print(f"⛔ Skipping {seq_id}: length {seqlen} aa (> {MAX_LEN})")
            continue

        # Status logging
        print(f"\n>{seq_id}")
        print(f"length {seqlen}")

        start_t = time.time()
        try:
            with torch.no_grad():
                pdb_str = model.infer_pdb(seq)
        except Exception as e:
            print(f"❌ Error folding {seq_id}: {e}")
            continue

        wall = time.time() - start_t
        print(f"Wall time: {wall:.2f} s")

        # Write PDB
        with open(pdb_path, "w") as fh:
            fh.write(pdb_str)

        print(f"✅ Saved: {pdb_path}")

def main():
    ap = argparse.ArgumentParser(
        description="Run ESMFold on every protein in a FASTA and write PDBs."
    )
    ap.add_argument(
        "--fasta",
        required=True,
        help="Path to input FASTA (.faa)"
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Directory to write .pdb outputs"
    )
    args = ap.parse_args()

    fold_fasta(args.fasta, args.outdir)

if __name__ == "__main__":
    main()
