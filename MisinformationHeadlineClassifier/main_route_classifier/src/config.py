import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    model_name: str = 'bert-base-uncased'
    num_labels: int = 3
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-05
    num_epochs: int = 20
    weight_decay: float = 0.01
    dropout_rate: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = 'cosine'
    min_lr_ratio: float = 0.1
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    data_dir: str = 'Dataset'
    results_dir: str = 'results'
    model_save_path: str = 'results/best_model'
    label_names: List[str] = field(default_factory=lambda: ['central_route', 'peripheral_route', 'neutral_route'])
    field_mapping: dict = field(default_factory=lambda: {'central_route': 'framework1_feature1', 'peripheral_route': 'framework1_feature2', 'neutral_route': 'framework1_feature3'})

    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
