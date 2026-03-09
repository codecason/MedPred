from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import get_config
from .data import Rna3DDataset, build_target_residue_index, collate_batch, load_labels_csv, load_sequences_csv
from .model import MedPredModel


def clip_coords(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, -999.999, 9999.999)


def load_model(ckpt_path: str, device: torch.device) -> MedPredModel:
    cfg = get_config()
    model = MedPredModel(cfg.model).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_for_split(split: str, ckpt_path: str, out_path: str) -> None:
    """
    split: 'validation' or 'test'
    Produces a sample_submission-style CSV with 5 coordinate sets per residue.
    """
    cfg = get_config()
    data_dir = cfg.data.data_dir
    seq_path = os.path.join(data_dir, f"{split}_sequences.csv")

    sequences = load_sequences_csv(seq_path)

    # For submission template, we need all IDs from train_labels.csv
    train_labels_path = os.path.join(data_dir, cfg.data.train_labels)
    train_labels = load_labels_csv(train_labels_path)

    # Build a template using train_labels format
    template = train_labels[["ID", "resname", "resid"]].copy()

    # We'll ignore real chain/copy for now (optional in submission)
    # and fill x_i,y_i,z_i for i in 1..5 from the same prediction

    # Dataset without labels
    ds = Rna3DDataset(sequences, None, None)
    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_batch,
    )

    device = torch.device(cfg.training.device)
    model = load_model(ckpt_path, device)

    # Mapping from target_id to predicted coords (L,3)
    preds = {}
    all_entries = list(sequences.values())

    with torch.no_grad():
        idx_offset = 0
        for batch in loader:
            x = batch["x"].to(device)
            lengths = batch["lengths"].to(device)
            coords = model(x, lengths)  # (B,L,3)
            coords_np = coords.cpu().numpy()
            for i in range(coords_np.shape[0]):
                entry = all_entries[idx_offset + i]
                L = lengths[i].item()
                preds[entry.target_id] = coords_np[i, :L, :]
            idx_offset += coords_np.shape[0]

    # Fill submission rows
    coord_cols: List[str] = []
    # Determine max L in predictions for column construction
    max_L = max(v.shape[0] for v in preds.values()) if preds else 0
    for i in range(1, 6):
        for j in range(max_L):
            coord_cols.extend([f"x_{i}_{j+1}", f"y_{i}_{j+1}", f"z_{i}_{j+1}"])

    # Initialize all coordinate columns with NaN
    for c in coord_cols:
        template[c] = np.nan

    # Assign coordinates per residue
    for idx, row in template.iterrows():
        full_id = row["ID"]
        target_id, resid_str = full_id.rsplit("_", 1)
        resid = int(resid_str) - 1  # to 0-based index
        if target_id not in preds:
            continue
        coords = preds[target_id]
        if 0 <= resid < coords.shape[0]:
            x, y, z = coords[resid]
            for i in range(1, 6):
                base = (resid) * 3
                template.loc[idx, f"x_{i}_{resid+1}"] = x
                template.loc[idx, f"y_{i}_{resid+1}"] = y
                template.loc[idx, f"z_{i}_{resid+1}"] = z

    # Clip values
    for c in coord_cols:
        template[c] = clip_coords(template[c].to_numpy(dtype=float))

    template.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--out", type=str, default="submission.csv")
    args = parser.parse_args()

    predict_for_split(args.split, args.ckpt, args.out)


if __name__ == "__main__":
    main()

