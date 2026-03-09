import os
from dataclasses import dataclass


@dataclass
class DataConfig:
    data_dir: str = os.path.join("data")
    train_sequences: str = "train_sequences.csv"
    train_labels: str = "train_labels.csv"
    validation_sequences: str = "validation_sequences.csv"
    test_sequences: str = "test_sequences.csv"
    msa_dir: str = os.path.join("MSA")


@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1e-4
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    num_workers: int = 2


@dataclass
class ModelConfig:
    vocab_size: int = 4  # A,C,G,U
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1


@dataclass
class ExperimentConfig:
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()


def get_config() -> ExperimentConfig:
    return ExperimentConfig()

