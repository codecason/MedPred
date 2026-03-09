from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3}


def one_hot_encode_sequence(seq: str) -> np.ndarray:
    """Simple A/C/G/U one-hot encoding."""
    arr = np.zeros((len(seq), len(NUC_TO_IDX)), dtype=np.float32)
    for i, ch in enumerate(seq):
        idx = NUC_TO_IDX.get(ch, None)
        if idx is not None:
            arr[i, idx] = 1.0
    return arr


@dataclass
class SequenceEntry:
    target_id: str
    sequence: str
    temporal_cutoff: str
    description: str
    stoichiometry: str
    all_sequences: str
    ligand_ids: str
    ligand_smiles: str


def load_sequences_csv(path: str) -> Dict[str, SequenceEntry]:
    df = pd.read_csv(path)
    entries: Dict[str, SequenceEntry] = {}
    for _, row in df.iterrows():
        entries[row["target_id"]] = SequenceEntry(
            target_id=row["target_id"],
            sequence=row["sequence"],
            temporal_cutoff=row.get("temporal_cutoff", ""),
            description=row.get("description", ""),
            stoichiometry=row.get("stoichiometry", ""),
            all_sequences=row.get("all_sequences", ""),
            ligand_ids=row.get("ligand_ids", ""),
            ligand_smiles=row.get("ligand_SMILES", ""),
        )
    return entries


def load_labels_csv(path: str) -> pd.DataFrame:
    """Load train/validation labels (per-residue C1' coordinates)."""
    return pd.read_csv(path)


def build_target_residue_index(labels: pd.DataFrame) -> Dict[str, List[Tuple[int, int]]]:
    """
    Build mapping: target_id -> list of (resid, row_index) for that residue.

    ID column is of the form '{target_id}_{resid}' with resid 1-based.
    """
    mapping: Dict[str, List[Tuple[int, int]]] = {}
    for idx, row in labels.iterrows():
        full_id: str = row["ID"]
        # split only at last underscore in case target_id contains '_'
        target_id, resid_str = full_id.rsplit("_", 1)
        resid = int(resid_str)
        mapping.setdefault(target_id, []).append((resid, idx))
    return mapping


class Rna3DDataset(Dataset):
    """
    Minimal Dataset:
    - X: one-hot encoded sequence (L, 4)
    - y: flattened C1' coordinates (L * 3) for one chosen structure (x_1,y_1,z_1,...)

    This ignores multiple experimental conformations for simplicity and
    uses the first structure (x_1,y_1,z_1,...) as target.
    """

    def __init__(
        self,
        sequences: Dict[str, SequenceEntry],
        labels: Optional[pd.DataFrame],
        target_residue_index: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    ) -> None:
        self.sequences = list(sequences.values())
        self.labels = labels
        self.target_residue_index = target_residue_index or {}

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.sequences[idx]
        seq = entry.sequence
        x = one_hot_encode_sequence(seq)  # (L,4)
        x_tensor = torch.from_numpy(x)  # float32

        item: Dict[str, torch.Tensor] = {
            "target_id": torch.tensor(idx, dtype=torch.long),
            "x": x_tensor,
        }

        if self.labels is not None and entry.target_id in self.target_residue_index:
            rows = sorted(self.target_residue_index[entry.target_id], key=lambda t: t[0])
            # assume resid from 1..L and use first set of coords x_1,y_1,z_1,...
            coord_cols = [c for c in self.labels.columns if c.startswith("x_1") or c.startswith("y_1") or c.startswith("z_1")]
            coords_list: List[np.ndarray] = []
            for resid, df_idx in rows:
                row = self.labels.loc[df_idx, coord_cols].to_numpy(dtype=np.float32)
                coords_list.append(row)
            y = np.stack(coords_list, axis=0)  # (L, 3)
            y_tensor = torch.from_numpy(y.reshape(-1))  # (L*3,)
            item["y"] = y_tensor

        return item


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Simple padding collate function for variable-length sequences."""
    xs = [b["x"] for b in batch]
    max_len = max(x.shape[0] for x in xs)
    feat_dim = xs[0].shape[1]
    padded = torch.zeros(len(xs), max_len, feat_dim, dtype=xs[0].dtype)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, : x.shape[0]] = x

    out: Dict[str, torch.Tensor] = {
        "x": padded,
        "lengths": lengths,
    }

    if "y" in batch[0]:
        ys = [b["y"] for b in batch]
        max_y = max(y.shape[0] for y in ys)
        y_padded = torch.zeros(len(ys), max_y, dtype=ys[0].dtype)
        for i, y in enumerate(ys):
            y_padded[i, : y.shape[0]] = y
        out["y"] = y_padded

    return out

