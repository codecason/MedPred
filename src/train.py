from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import get_config
from .data import Rna3DDataset, build_target_residue_index, collate_batch, load_labels_csv, load_sequences_csv
from .model import MedPredModel, coords_to_flat


def make_dataloaders() -> Tuple[DataLoader, DataLoader]:
    cfg = get_config()
    data_dir = cfg.data.data_dir

    train_seq_path = os.path.join(data_dir, cfg.data.train_sequences)
    train_labels_path = os.path.join(data_dir, cfg.data.train_labels)
    val_seq_path = os.path.join(data_dir, cfg.data.validation_sequences)

    train_seqs = load_sequences_csv(train_seq_path)
    train_labels = load_labels_csv(train_labels_path)
    index = build_target_residue_index(train_labels)

    val_seqs = load_sequences_csv(val_seq_path)

    train_ds = Rna3DDataset(train_seqs, train_labels, index)
    val_ds = Rna3DDataset(val_seqs, None, None)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_batch,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: MedPredModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        lengths = batch["lengths"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        coords = model(x, lengths)  # (B,L,3)
        y_pred = coords_to_flat(coords, lengths)

        # match y size (it may be shorter due to variable L)
        min_dim = min(y_pred.size(1), y.size(1))
        loss = criterion(y_pred[:, :min_dim], y[:, :min_dim])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def main() -> None:
    cfg = get_config()
    train_loader, _ = make_dataloaders()

    device = torch.device(cfg.training.device)
    model = MedPredModel(cfg.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(cfg.training.num_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{cfg.training.num_epochs} - train_loss: {loss:.4f}")
        ckpt_path = os.path.join("checkpoints", f"model_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()

