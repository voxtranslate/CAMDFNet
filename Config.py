import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

# ============================================================================
# CONFIGURATION SYSTEM
# ============================================================================
@dataclass
class DatasetConfig:
    """Dataset configuration with defaults"""
    data_root: str = "/kaggle/input/plantvillage-dataset/color"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    batch_size: int = 16  # Increased from 1 to avoid batch norm issues
    num_workers: int = 4
    pin_memory: bool = True
    mask_method: str = "ensemble"
    augment: bool = True
    image_size: int = 512
    use_advanced_augmentation: bool = True
    augment_prob: float = 0.5
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.3
    color_jitter_hue: float = 0.15
    rotation_degrees: int = 30
    use_tta: bool = True
    tta_transforms: int = 5
    preserve_original_size_eval: bool = True

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    name: str = "camdfnet_base"
    num_classes: int = None  # Will be set automatically from dataset
    hidden_dim: int = 768
    use_fpn: bool = True
    use_se: bool = True
    use_sarm: bool = True
    dropout_rate: float = 0.1
    pretrained: bool = True
    return_attention_maps: bool = True
    save_all_activations: bool = True

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 10
    gradient_clip: float = 1.0
    early_stopping_patience: int = 50
    mixed_precision: bool = True
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    augmentation_prob: float = 0.5
    split: str = 'test'

@dataclass
class LossConfig:
    """Loss function configuration"""
    lambda_dice: float = 1.0
    lambda_bce: float = 1.0
    lambda_cls: float = 1.0
    lambda_mask_reg: float = 0.5
    lambda_aux: float = 0.5
    use_focal: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    use_class_weights: bool = True
    label_smoothing: float = 0.2

@dataclass
class SystemConfig:
    """System-level configuration"""
    device: str = "cuda"
    seed: int = 42
    checkpoint_dir: str = "./runnings"
    log_dir: str = "./logs"
    output_dir: str = "./results"
    save_frequency: int = 25
    verbose: bool = True
    experiment_name: str = "camdfnet"
    save_all_intermediates: bool = True

@dataclass
class CAMDFNetConfig:
    """Complete system configuration"""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'CAMDFNetConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(
            dataset=DatasetConfig(**data.get('dataset', {})),
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {})),
            loss=LossConfig(**data.get('loss', {})),
            system=SystemConfig(**data.get('system', {}))
        )
    
    def to_yaml(self, path: str):
        data = {
            'dataset': asdict(self.dataset),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'loss': asdict(self.loss),
            'system': asdict(self.system)
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
