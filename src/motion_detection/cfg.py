from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    momentum: float
    weight_decay: float
    log_interval: int
    save_interval: int
